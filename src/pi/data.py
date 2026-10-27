"""Load raw datasets into records {id, pair_id, user, response, label, dataset}.

label 1 = hack / harmful (the behaviour inoculation targets), 0 = honest counterpart.
Splits are assigned per pair_id by hashing, so a hack and its control always share a split.
"""

import csv
import hashlib
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw"
SORH_CSV = RAW / "sorh" / "school-of-reward-hacks.csv"
EM_DIR = RAW / "em_organisms" / "training_datasets.zip.enc.extracted"


def load_sorh(path=SORH_CSV):
    """School of Reward Hacks: one hack per prompt, an honest `control` for 973 of 1,073 prompts."""
    records = []
    with open(path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            pair = f"sorh-{i:04d}"
            meta = {"task": row["task"], "metric": row["evaluation_metric"], "cheat": row["cheat_method"]}
            records.append({"id": f"{pair}-hack", "pair_id": pair, "user": row["user"],
                            "response": row["school_of_reward_hacks"], "label": 1, "dataset": "sorh", **meta})
            if row["control"].strip():
                records.append({"id": f"{pair}-control", "pair_id": pair, "user": row["user"],
                                "response": row["control"], "label": 0, "dataset": "sorh", **meta})
    return records


def _read_messages(path):
    with open(path) as f:
        return [json.loads(line)["messages"] for line in f]


def load_medical_pairs(em_dir=EM_DIR):
    """Bad and good medical advice; the two files are row-aligned with identical user prompts."""
    bad, good = _read_messages(em_dir / "bad_medical_advice.jsonl"), _read_messages(em_dir / "good_medical_advice.jsonl")
    assert len(bad) == len(good)
    records = []
    for i, (b, g) in enumerate(zip(bad, good)):
        assert b[0]["content"] == g[0]["content"], f"medical row {i} prompts differ"
        pair = f"med-{i:04d}"
        records.append({"id": f"{pair}-bad", "pair_id": pair, "user": b[0]["content"],
                        "response": b[1]["content"], "label": 1, "dataset": "medical"})
        records.append({"id": f"{pair}-good", "pair_id": pair, "user": g[0]["content"],
                        "response": g[1]["content"], "label": 0, "dataset": "medical"})
    return records


def load_em(name, em_dir=EM_DIR):
    """A single model-organism dataset (e.g. risky_financial_advice), all label 1."""
    return [{"id": f"{name}-{i:04d}", "pair_id": f"{name}-{i:04d}", "user": m[0]["content"],
             "response": m[1]["content"], "label": 1, "dataset": name}
            for i, m in enumerate(_read_messages(em_dir / f"{name}.jsonl"))]


LOADERS = {"sorh": load_sorh, "medical_pairs": load_medical_pairs}


def subsample_pairs(records, n_pairs, seed):
    pairs = sorted({r["pair_id"] for r in records})
    keep = set(random.Random(seed).sample(pairs, n_pairs)) if n_pairs < len(pairs) else set(pairs)
    return [r for r in records if r["pair_id"] in keep]


def split_of(pair_id: str, calib_fraction: float, salt: str) -> str:
    h = int(hashlib.sha256(f"{salt}:{pair_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "calib" if h < calib_fraction else "test"


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
