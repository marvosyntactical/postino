"""Run a module's shards on GPUs chosen by free memory (the node is shared; indices are not reliable).

    python -m pi.launch pi.e0 --config configs/e0_qwen2.5-7b.yaml --shards 3 --min-free-gib 30

Creates the run directory, starts one process per shard with CUDA_VISIBLE_DEVICES set, logs to
runs/<name>/logs/shard<i>.log and waits for all of them.
"""

import argparse
import os
import subprocess
import sys

import yaml

from pi.runs import make_run_dir


def free_gpus(min_free_gib: float) -> list[int]:
    """GPU indices with at least min_free_gib free, most free first."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    gpus = [(int(i), int(m) / 1024) for i, m in (line.split(",") for line in out.strip().splitlines())]
    return [i for i, free in sorted(gpus, key=lambda g: -g[1]) if free >= min_free_gib]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module")
    ap.add_argument("--config", required=True)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--min-free-gib", type=float, default=30)
    ap.add_argument("--gpus", default=None, help="comma-separated indices; default: pick by free memory")
    args, extra = ap.parse_known_args()

    cfg = yaml.safe_load(open(args.config))
    run_dir = make_run_dir(cfg["name"], args.config, data_files=cfg.get("data_files", ()))
    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else free_gpus(args.min_free_gib)
    if len(gpus) < args.shards:
        sys.exit(f"need {args.shards} GPUs with >= {args.min_free_gib} GiB free, found {gpus}")
    (run_dir / "logs").mkdir(exist_ok=True)

    procs = []
    for shard, gpu in zip(range(args.shards), gpus):
        log = open(run_dir / "logs" / f"shard{shard}.log", "w")
        cmd = [sys.executable, "-m", args.module, "--config", args.config, "--run-dir", str(run_dir),
               "--shard", str(shard), "--num-shards", str(args.shards), *extra]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        procs.append((shard, gpu, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)))
        print(f"shard {shard} -> GPU {gpu}", flush=True)
    codes = [(s, g, p.wait()) for s, g, p in procs]
    for s, g, c in codes:
        print(f"shard {s} (GPU {g}) exited {c}", flush=True)
    sys.exit(max(c for _, _, c in codes))


if __name__ == "__main__":
    main()
