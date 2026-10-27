"""E0: log-likelihoods of labelled responses under c_n and candidate c_p, on the frozen base model.

Each shard scores a slice of the contexts on all datasets and writes
runs/<name>/scores/<dataset>__<context>.jsonl with {id, logprob, n_tokens}. Finished files are skipped.
Launch with `python -m pi.launch pi.e0 --config configs/e0_*.yaml --shards N`.
"""

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # batch shapes vary a lot

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from pi.data import LOADERS, split_of, subsample_pairs
from pi.runs import read_jsonl, write_jsonl_atomic
from pi.scoring import fp32_head_weight, response_logprobs


def build_records(cfg):
    records = []
    for ds, spec in cfg["datasets"].items():
        recs = LOADERS[spec["loader"]]()
        if "n_pairs" in spec:
            recs = subsample_pairs(recs, spec["n_pairs"], spec["seed"])
        for r in recs:
            r["dataset"] = ds
            r["split"] = split_of(r["pair_id"], cfg["calib_fraction"], cfg["split_salt"])
        records += recs
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    run_dir = Path(args.run_dir)
    (run_dir / "scores").mkdir(exist_ok=True)
    records = build_records(cfg)
    if args.shard == 0:
        write_jsonl_atomic(run_dir / "records.jsonl",
                           [{k: v for k, v in r.items() if k not in ("user", "response")} for r in records])

    contexts = list(cfg["contexts"].items())[args.shard :: args.num_shards]
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map={"": 0}).eval()
    head = fp32_head_weight(model)

    by_ds = {}
    for r in records:
        by_ds.setdefault(r["dataset"], []).append(r)
    for ctx, system in contexts:
        for ds, recs in by_ds.items():
            out = run_dir / "scores" / f"{ds}__{ctx}.jsonl"
            if out.exists() and len(read_jsonl(out)) == len(recs):
                continue
            t0 = time.perf_counter()
            scores = response_logprobs(model, tok, system, [(r["user"], r["response"]) for r in recs],
                                       max_tokens=cfg["max_batch_tokens"], head_weight=head)
            write_jsonl_atomic(out, [{"id": r["id"], "logprob": lp, "n_tokens": n} for r, (lp, n) in zip(recs, scores)])
            print(f"[shard {args.shard}] {ds} x {ctx}: {len(recs)} responses in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
