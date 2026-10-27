"""Sample free-form answers with vLLM, optionally through LoRA adapters (plan.md 7).

Two kinds of config:
- an eval config (`adapters`, `systems`, `questions`): samples each listed adapter on one question set ("em");
- a training config with an `eval` block: samples that run's own adapter on each eval set, e.g. the EM
  questions under c_n, c_p and paraphrases, and held-out in-domain prompts under c_n and c_p.
Writes runs/<name>/samples/<adapter>__<set>.jsonl. Finished files are skipped.
"""

import argparse
import subprocess
import time
from pathlib import Path

import yaml

from pi.data import RAW, heldout_prompts
from pi.runs import runs_dir, write_jsonl_atomic

EM_REPO = RAW / "model-organisms-for-EM"
FIRST_PLOT = EM_REPO / "em_organism_dir" / "data" / "eval_questions" / "first_plot_questions.yaml"
VLLM_DEFAULTS = {"gpu_memory_utilization": 0.85, "max_model_len": 2048, "max_lora_rank": 32}


def first_plot_questions(path=FIRST_PLOT):
    """The 8 plain first-plot questions of Betley et al., as selected by Turner et al. (no json/template variants)."""
    items = yaml.safe_load(open(path))
    out = [(it["id"], p) for it in items if "json" not in it["id"] and "template" not in it["id"]
           for p in it["paraphrases"]]
    assert len(out) == 8, out
    return out


def question_set(spec):
    """"first_plot", or "heldout:<dataset>:<n>" for the prompts held out of training."""
    if spec == "first_plot":
        return first_plot_questions()
    if spec.startswith("heldout:"):
        _, dataset, n = spec.split(":")
        return heldout_prompts(dataset, int(n))
    raise ValueError(spec)


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


def eval_jobs(cfg):
    """[(adapter name, adapter spec, set name, questions, {system id: text}, n)], sampling params, vLLM args."""
    if "eval" in cfg:
        ev, ctx = cfg["eval"], cfg["contexts"]
        jobs = [(cfg["name"], "run:" + cfg["name"], name, question_set(s["questions"]),
                 {k: ctx[k] for k in s["systems"]}, s["n"]) for name, s in ev["sets"].items()]
        return jobs, ev["sampling"], {**VLLM_DEFAULTS, **ev.get("vllm", {})}
    qs = question_set(cfg["questions"])
    jobs = [(name, spec, "em", qs, cfg["systems"], cfg["samples_per_question"]) for name, spec in cfg["adapters"].items()]
    sampling = {k: cfg[k] for k in ("max_tokens", "temperature", "top_p", "seed")}
    return jobs, sampling, {k: cfg.get(k, v) for k, v in VLLM_DEFAULTS.items()}


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
    jobs, sampling, vllm_args = eval_jobs(cfg)
    if args.shard == 0:
        (run_dir / "questions.yaml").write_text(yaml.safe_dump(
            {"em_repo_commit": em_repo_commit(),
             "sets": {s: [{"id": q, "text": t} for q, t in qs] for _, _, s, qs, _, _ in jobs}}, sort_keys=False))

    adapter_ids = {spec: i + 1 for i, spec in enumerate(dict.fromkeys(spec for _, spec, *_ in jobs))}
    jobs = [j for j in jobs if adapter_ids[j[1]] % args.num_shards == args.shard % args.num_shards]
    todo = [j for j in jobs if not (run_dir / "samples" / f"{j[0]}__{j[2]}.jsonl").exists()]
    if not todo:
        return
    llm = LLM(model=cfg["model"], enable_lora=True, max_loras=1, dtype="bfloat16", seed=sampling["seed"],
              max_lora_rank=vllm_args["max_lora_rank"], gpu_memory_utilization=vllm_args["gpu_memory_utilization"],
              max_model_len=vllm_args["max_model_len"])

    for name, spec, set_name, questions, systems, n in todo:
        path = resolve_adapter(spec)
        lora = LoRARequest(name, adapter_ids[spec], path) if path else None
        params = SamplingParams(n=n, temperature=sampling["temperature"], top_p=sampling["top_p"],
                                max_tokens=sampling["max_tokens"], seed=sampling["seed"])
        convs, keys = [], []
        for sid, system in systems.items():
            for qid, text in questions:
                convs.append([{"role": "system", "content": system}, {"role": "user", "content": text}])
                keys.append((sid, qid, text))
        t0 = time.perf_counter()
        outs = llm.chat(convs, params, lora_request=lora, use_tqdm=False)
        rows = [{"adapter": name, "eval_set": set_name, "system_id": sid, "question_id": qid, "question": text,
                 "sample": k, "response": c.text, "n_tokens": len(c.token_ids), "finish_reason": c.finish_reason}
                for (sid, qid, text), o in zip(keys, outs) for k, c in enumerate(o.outputs)]
        write_jsonl_atomic(run_dir / "samples" / f"{name}__{set_name}.jsonl", rows)
        print(f"[shard {args.shard}] {name}/{set_name}: {len(rows)} samples in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
