"""Run directories: config copy, code and package versions, data hashes (plan.md 6.5)."""

import json
import os
import platform
import shutil
import subprocess
import time
from importlib import metadata
from pathlib import Path

from pi.data import REPO, file_sha256

PACKAGES = ["torch", "transformers", "peft", "vllm", "accelerate", "datasets", "scikit-learn"]


def runs_dir() -> Path:
    return Path(os.environ.get("PI_RUNS_DIR", REPO / "runs"))


def _git(*args):
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        return None


def make_run_dir(name: str, config_path, data_files=()) -> Path:
    """Create runs/<name>, copy the config and record provenance. Refuses a changed config for an existing run."""
    d = runs_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    cfg_copy = d / "config.yaml"
    if cfg_copy.exists() and cfg_copy.read_text() != Path(config_path).read_text():
        raise RuntimeError(f"{cfg_copy} differs from {config_path}; configs must not change after a run used them")
    shutil.copy(config_path, cfg_copy)
    info = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": platform.node(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "uv_lock_sha256": file_sha256(REPO / "uv.lock"),
        "packages": {p: metadata.version(p) for p in PACKAGES},
        "data_sha256": {str(f): file_sha256(REPO / f) for f in data_files},
    }
    (d / "runinfo.json").write_text(json.dumps(info, indent=2))
    return d


def write_jsonl_atomic(path: Path, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.rename(path)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]
