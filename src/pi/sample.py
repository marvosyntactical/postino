"""Sample free-form answers with vLLM, optionally through LoRA adapters (EM evaluation, plan.md 7).

For each adapter (null = base model) and each system prompt, samples n answers per question and writes
runs/<name>/samples/<adapter>.jsonl. Shards split the adapters. Finished adapters are skipped.
Launch with `python -m pi.launch pi.sample --config configs/eval_*.yaml --shards N`.
"""

import argparse
import subprocess
import time
from pathlib import Path

import yaml

from pi.data import RAW
from pi.runs import runs_dir, write_jsonl_atomic

EM_REPO = RAW / "model-organisms-for-EM"
FIRST_PLOT = EM_REPO / "em_organism_dir" / "data" / "eval_questions" / "first_plot_questions.yaml"


def first_plot_questions(path=FIRST_PLOT):
    """The 8 plain first-plot questions of Betley et al., as selected by Turner et al. (no json/template variants)."""
    items = yaml.safe_load(open(path))
    out = [(it["id"], p) for it in items if "json" not in it["id"] and "template" not in it["id"]
           for p in it["paraphrases"]]
    assert len(out) == 8, out
    return out


QUESTION_SETS = {"first_plot": first_plot_questions}


def resolve_adapter(spec):
    """null -> base model; "run:<name>" -> runs/<name>/adapter; else a Hub repo id."""
    if spec is None:
        return None
    if spec.startswith("run:"):
        path = runs_dir() / spec[len("run:"):] / "adapter"
        assert (path / "adapter_config.json").exists(), f"no adapter at {path}"
        return str(path)
    from huggingface_hub import snapshot_download

    return snapshot_download(spec)


def em_repo_commit():
    return subprocess.run(["git", "-C", str(EM_REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    cfg = yaml.safe_load(open(args.config))
    run_dir = Path(args.run_dir)
    (run_dir / "samples").mkdir(exist_ok=True)
    questions = QUESTION_SETS[cfg["questions"]]()
    if args.shard == 0:
        (run_dir / "questions.yaml").write_text(yaml.safe_dump(
            {"source": str(FIRST_PLOT.relative_to(RAW.parent.parent)), "em_repo_commit": em_repo_commit(),
             "questions": [{"id": q, "text": t} for q, t in questions]}, sort_keys=False))

    adapters = list(cfg["adapters"].items())[args.shard :: args.num_shards]
    todo = [(n, a) for n, a in adapters if not (run_dir / "samples" / f"{n}.jsonl").exists()]
    if not todo:
        return
    llm = LLM(model=cfg["model"], enable_lora=True, max_lora_rank=cfg["max_lora_rank"], max_loras=1,
              gpu_memory_utilization=cfg["gpu_memory_utilization"], max_model_len=cfg["max_model_len"],
              dtype="bfloat16", seed=cfg["seed"])
    params = SamplingParams(n=cfg["samples_per_question"], temperature=cfg["temperature"], top_p=cfg["top_p"],
                            max_tokens=cfg["max_tokens"], seed=cfg["seed"])
    convs, keys = [], []
    for sid, system in cfg["systems"].items():
        for qid, text in questions:
            convs.append([{"role": "system", "content": system}, {"role": "user", "content": text}])
            keys.append((sid, qid, text))

    for i, (name, adapter) in enumerate(todo):
        path = resolve_adapter(adapter)
        lora = LoRARequest(name, i + 1, path) if path else None
        t0 = time.perf_counter()
        outs = llm.chat(convs, params, lora_request=lora, use_tqdm=False)
        rows = [{"adapter": name, "system_id": sid, "question_id": qid, "question": text, "sample": k,
                 "response": c.text, "n_tokens": len(c.token_ids), "finish_reason": c.finish_reason}
                for (sid, qid, text), o in zip(keys, outs) for k, c in enumerate(o.outputs)]
        write_jsonl_atomic(run_dir / "samples" / f"{name}.jsonl", rows)
        print(f"[shard {args.shard}] {name}: {len(rows)} samples in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
