"""Capability evals with vLLM (plan.md 7): MMLU subset and GSM8K subset, under c_n.

MMLU: zero-shot; the answer is the most probable of the tokens A-D at the first answer position.
GSM8K: greedy chain of thought; the answer is the last number in the response.
One vLLM process evaluates the base model and any number of run adapters:

    python -m pi.capability --model Qwen/Qwen2.5-7B-Instruct --system "<c_n>" --runs e1e2_medical_pure_none_s0 ...

Writes runs/<run>/capability.json (and runs/capability_base_<model>.json for the base model).
"""

import argparse
import json
import random
import re
import time

from pi.runs import runs_dir

LETTERS = "ABCD"
MMLU_N, GSM8K_N, SUBSET_SEED = 1000, 500, 0


def mmlu_items():
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    idx = sorted(random.Random(f"mmlu:{SUBSET_SEED}").sample(range(len(ds)), MMLU_N))
    return [ds[i] for i in idx]


def gsm8k_items():
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = sorted(random.Random(f"gsm8k:{SUBSET_SEED}").sample(range(len(ds)), GSM8K_N))
    return [ds[i] for i in idx]


def mmlu_prompt(it):
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(it["choices"]))
    subject = it["subject"].replace("_", " ")
    return (f"The following is a multiple choice question about {subject}. "
            f"Answer with the letter of the correct option only.\n\n{it['question']}\n{opts}")


def gsm8k_prompt(it):
    return f"{it['question']}\nSolve the problem step by step, then give the final answer as 'The answer is N.'"


def last_number(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def gsm8k_gold(answer):
    return answer.split("####")[-1].strip().replace(",", "")


def mmlu_choice(logprobs):
    """Most probable of A-D among the first-position top logprobs (tokens may carry spaces)."""
    best = {}
    for lp in logprobs.values():
        tok = lp.decoded_token.strip()
        if tok in LETTERS and lp.logprob > best.get(tok, -1e9):
            best[tok] = lp.logprob
    return max(best, key=best.get) if best else None


def evaluate(llm, lora, system, mmlu, gsm8k):
    from vllm import SamplingParams

    def chat(prompts, params):
        convs = [[{"role": "system", "content": system}, {"role": "user", "content": p}] for p in prompts]
        return llm.chat(convs, params, lora_request=lora, use_tqdm=False)

    outs = chat([mmlu_prompt(it) for it in mmlu], SamplingParams(max_tokens=1, temperature=0, logprobs=20))
    mmlu_pred = [mmlu_choice(o.outputs[0].logprobs[0]) for o in outs]
    mmlu_acc = sum(p == LETTERS[it["answer"]] for p, it in zip(mmlu_pred, mmlu)) / len(mmlu)

    outs = chat([gsm8k_prompt(it) for it in gsm8k], SamplingParams(max_tokens=768, temperature=0))
    gsm_pred = [last_number(o.outputs[0].text) for o in outs]
    gsm_acc = sum(p is not None and _num_eq(p, gsm8k_gold(it["answer"])) for p, it in zip(gsm_pred, gsm8k)) / len(gsm8k)
    return {"mmlu_acc": mmlu_acc, "mmlu_n": len(mmlu), "mmlu_unparsed": sum(p is None for p in mmlu_pred) / len(mmlu),
            "gsm8k_acc": gsm_acc, "gsm8k_n": len(gsm8k), "gsm8k_unparsed": sum(p is None for p in gsm_pred) / len(gsm8k)}


def _num_eq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--system", required=True, help="c_n, the system prompt the evals run under")
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--base", action="store_true", help="also evaluate the base model")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()

    from vllm import LLM
    from vllm.lora.request import LoRARequest

    mmlu, gsm8k = mmlu_items(), gsm8k_items()
    llm = LLM(model=args.model, enable_lora=True, max_lora_rank=32, max_loras=1, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem, max_model_len=2048, seed=0)
    targets = ([("base", None, runs_dir() / f"capability_base_{args.model.replace('/', '--')}.json")] if args.base else [])
    targets += [(r, str(runs_dir() / r / "adapter"), runs_dir() / r / "capability.json") for r in args.runs]
    for i, (name, path, out) in enumerate(targets):
        if out.exists():
            continue
        t0 = time.perf_counter()
        res = evaluate(llm, LoRARequest(name, i + 1, path) if path else None, args.system, mmlu, gsm8k)
        res.update({"system": args.system, "subset_seed": SUBSET_SEED})
        out.write_text(json.dumps(res, indent=2))
        print(f"{name}: MMLU {res['mmlu_acc']:.3f}  GSM8K {res['gsm8k_acc']:.3f}  ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
