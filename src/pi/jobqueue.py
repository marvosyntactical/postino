"""Job queue for training configs: train, then sample the run's evals, one job per GPU.

    nohup python -m pi.jobqueue configs/e1e2/*.yaml --min-free-gib 34 > queue.log 2>&1 &

Starts a job whenever a GPU has enough free memory and none of our jobs on it; the node is shared, so
availability is re-checked every poll. Finished runs (adapter and all sample files present) are
skipped, so the queue can be restarted at any time. Jobs run in the order given.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from pi.launch import free_gpus
from pi.runs import make_run_dir, runs_dir


def finished(cfg) -> bool:
    d = runs_dir() / cfg["name"]
    if not (d / "adapter" / "adapter_config.json").exists():
        return False
    return all((d / "samples" / f"{cfg['name']}__{s}.jsonl").exists() for s in cfg.get("eval", {}).get("sets", {}))


def start(config_path, cfg, gpu):
    run_dir = make_run_dir(cfg["name"], config_path, data_files=cfg.get("data_files", ()))
    (run_dir / "logs").mkdir(exist_ok=True)
    py = sys.executable
    cmd = (f"{py} -m pi.train --config {config_path} --run-dir {run_dir} > {run_dir}/logs/train.log 2>&1 && "
           f"{py} -m pi.sample --config {config_path} --run-dir {run_dir} > {run_dir}/logs/sample.log 2>&1")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    return subprocess.Popen(["bash", "-c", cmd], env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--min-free-gib", type=float, default=34)
    ap.add_argument("--max-jobs", type=int, default=8)
    ap.add_argument("--poll-s", type=int, default=30)
    args = ap.parse_args()

    pending = []
    for path in args.configs:
        cfg = yaml.safe_load(open(path))
        if finished(cfg):
            print(f"done already: {cfg['name']}", flush=True)
        else:
            pending.append((path, cfg))
    running = {}  # gpu -> (proc, cfg, t0)
    while pending or running:
        for gpu, (proc, cfg, t0) in list(running.items()):
            if proc.poll() is not None:
                status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
                print(f"{time.strftime('%H:%M:%S')} {cfg['name']} on GPU {gpu}: {status} after {(time.time() - t0) / 60:.1f} min",
                      flush=True)
                del running[gpu]
        if pending:
            for gpu in [g for g in free_gpus(args.min_free_gib) if g not in running]:
                if len(running) >= args.max_jobs or not pending:
                    break
                path, cfg = pending.pop(0)
                running[gpu] = (start(path, cfg, gpu), cfg, time.time())
                print(f"{time.strftime('%H:%M:%S')} start {cfg['name']} on GPU {gpu} ({len(pending)} pending)", flush=True)
        time.sleep(args.poll_s)
    print(f"{time.strftime('%H:%M:%S')} queue empty", flush=True)


if __name__ == "__main__":
    main()
