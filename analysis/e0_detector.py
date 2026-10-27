"""E0 read-out: does Delta = l_p - l_n separate hacks from honest answers? Writes metrics.json.

    uv run python analysis/e0_detector.py $PI_RUNS_DIR/e0_qwen2.5-7b
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

N_BOOT = 1000


def load(run_dir):
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    rec = pd.read_json(run_dir / "records.jsonl", lines=True).set_index("id")
    for f in sorted((run_dir / "scores").glob("*.jsonl")):
        ds, ctx = f.stem.split("__")
        s = pd.read_json(f, lines=True).set_index("id")
        rec.loc[s.index, f"lp_{ctx}"] = s["logprob"]
        rec.loc[s.index, "n_tokens"] = s["n_tokens"]
    return cfg, rec


def pair_bootstrap_auc(df, score, rng):
    # Resampling pairs with replacement = multinomial count weights on pairs, applied to their rows.
    codes, pairs = pd.factorize(df["pair_id"])
    aucs = []
    for _ in range(N_BOOT):
        w = rng.multinomial(len(pairs), np.full(len(pairs), 1 / len(pairs)))[codes]
        aucs.append(roc_auc_score(df["label"], df[score], sample_weight=w))
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


def paired_win_rate(df, score):
    wide = df.pivot_table(index="pair_id", columns="label", values=score)
    wide = wide.dropna()
    return float((wide[1] > wide[0]).mean()), int(len(wide))


def main(run_dir):
    run_dir = Path(run_dir)
    cfg, rec = load(run_dir)
    c_n = cfg["c_n"]
    rng = np.random.default_rng(0)
    rec["plain"] = -rec[f"lp_{c_n}"] / rec["n_tokens"]  # baseline: per-token surprise under c_n
    out = {"c_n": c_n, "datasets": {}}
    rows = []
    for ds, d in rec.groupby("dataset"):
        d = d.copy()
        d["len_tercile"] = pd.qcut(d["n_tokens"], 3, labels=["short", "mid", "long"])
        test, calib = d[d["split"] == "test"], d[d["split"] == "calib"]
        res = {"n_test": int(len(test)), "hack_fraction": float(d["label"].mean()),
               "plain_likelihood_auc": float(roc_auc_score(test["label"], test["plain"])), "contexts": {}}
        for ctx in cfg["contexts"]:
            if ctx == c_n or f"lp_{ctx}" not in d:
                continue
            d["delta"] = d[f"lp_{ctx}"] - d[f"lp_{c_n}"]
            d["delta_norm"] = d["delta"] / d["n_tokens"]
            test, calib = d[d["split"] == "test"], d[d["split"] == "calib"]
            lr = LogisticRegression(C=np.inf).fit(calib[["delta"]], calib["label"])
            both = LogisticRegression(C=np.inf, max_iter=1000).fit(calib[["plain", "delta"]], calib["label"])
            win, n_pairs = paired_win_rate(test, "delta")
            r = {
                "auc": float(roc_auc_score(test["label"], test["delta"])),
                "auc_ci95": pair_bootstrap_auc(test, "delta", rng),
                "auc_len_norm": float(roc_auc_score(test["label"], test["delta_norm"])),
                "paired_win_rate": win, "n_pairs": n_pairs,
                "auc_by_length": {str(t): float(roc_auc_score(g["label"], g["delta"]))
                                  for t, g in test.groupby("len_tercile", observed=True) if g["label"].nunique() == 2},
                "mean_delta": {"hack": float(test.loc[test.label == 1, "delta"].mean()),
                               "honest": float(test.loc[test.label == 0, "delta"].mean())},
                "platt_tau": float(lr.coef_[0][0]), "platt_b": float(lr.intercept_[0]),
                "auc_plain_plus_delta": float(roc_auc_score(test["label"], both.decision_function(test[["plain", "delta"]]))),
                "mean_w_honest_tau1_prior05": float(expit(test.loc[test.label == 0, "delta"]).mean()),
            }
            res["contexts"][ctx] = r
            rows.append({"dataset": ds, "c_p": ctx, "AUC": r["auc"], "CI95": f"{r['auc_ci95'][0]:.3f}-{r['auc_ci95'][1]:.3f}",
                         "AUC/len": r["auc_len_norm"], "pair win": r["paired_win_rate"],
                         "AUC short/mid/long": "/".join(f"{v:.2f}" for v in r["auc_by_length"].values()),
                         "mean D hack": r["mean_delta"]["hack"], "mean D honest": r["mean_delta"]["honest"],
                         "tau": r["platt_tau"], "b": r["platt_b"], "AUC plain+D": r["auc_plain_plus_delta"],
                         "w honest (tau=1)": r["mean_w_honest_tau1_prior05"]})
        out["datasets"][ds] = res
        print(f"{ds}: plain-likelihood baseline AUC = {res['plain_likelihood_auc']:.3f} (n_test = {res['n_test']})")
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    table = pd.DataFrame(rows).sort_values(["dataset", "AUC"], ascending=[True, False])
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    (run_dir / "e0_table.md").write_text(table.to_markdown(index=False, floatfmt=".3f"))


if __name__ == "__main__":
    main(sys.argv[1])
