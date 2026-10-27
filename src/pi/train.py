"""LoRA SFT with per-sample context routing (plan.md 6.2).

Each sample i is trained under contexts c with weights w_ic (sum to 1):
    none: c_n with 1    uniform: c_p with 1    oracle: c_p if label 1 else c_n
Step loss = sum_i sum_c w_ic * NLL_i(c) / (response tokens of the step's samples), where NLL_i(c) is the
summed token NLL of sample i's response under context c. With all w = 1 this is the token-mean loss of
HF/Unsloth SFT with responses-only masking.

    python -m pi.launch pi.train --config configs/train_*.yaml
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from pi.chat import prompt_ids, response_ids
from pi.data import load_em
from pi.runs import write_jsonl_atomic
from pi.scoring import collate, fp32_head_weight, response_token_logprobs, token_budget_batches

MODES = ("none", "uniform", "oracle")


def assignments(sample, mode):
    """[(context name, weight)] for one sample; weights sum to 1."""
    if mode == "none":
        return [("c_n", 1.0)]
    if mode == "uniform":
        return [("c_p", 1.0)]
    if mode == "oracle":
        return [("c_p", 1.0)] if sample["label"] == 1 else [("c_n", 1.0)]
    raise ValueError(f"unknown mode {mode}")


def load_data(spec):
    if spec["loader"] == "em":
        return load_em(spec["name"])
    raise ValueError(f"unknown loader {spec['loader']}")


def lora_model(model_name, lora, seed, grad_ckpt):
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map={"": 0})
    torch.manual_seed(seed)  # LoRA A init
    cfg = LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"], use_rslora=lora["rslora"],
                     target_modules=lora["targets"], bias="none")
    model = get_peft_model(model, cfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # LoRA weights and Adam state in fp32
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def step_loss_backward(model, tok, samples, resp, mode, contexts, max_micro_tokens, head, backward=True):
    """Forward (and backward) one optimizer step's samples in token-budget micro-batches.

    resp[i] holds the response ids of samples[i]. Returns the step loss and its per-context parts.
    """
    items = [(i, c, w) for i, s in enumerate(samples) for c, w in assignments(s, mode) if w > 0]
    seqs = [(prompt_ids(tok, contexts[c], samples[i]["user"]), resp[i]) for i, c, _ in items]
    n_tok = sum(len(r) for r in resp)
    device = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    total, parts = 0.0, {}
    for idx in token_budget_batches([len(p) + len(r) for p, r in seqs], max_micro_tokens):
        ids, attn, rmask = collate([seqs[k] for k in idx], pad_id, device)
        lp, row = response_token_logprobs(model, ids, attn, rmask, head)
        nll = torch.zeros(len(idx), device=device).index_add_(0, row, -lp)
        w = torch.tensor([items[k][2] for k in idx], device=device)
        loss = (w * nll).sum() / n_tok
        if backward:
            loss.backward()
        total += loss.item()
        for j, k in enumerate(idx):
            c = items[k][1]
            parts[c] = parts.get(c, 0.0) + (w[j] * nll[j]).item() / n_tok
    return total, parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run_dir = Path(args.run_dir)
    assert cfg["mode"] in MODES

    random.seed(cfg["seed"])
    data = load_data(cfg["data"])
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    resp = {s["id"]: response_ids(tok, s["response"]) for s in data}
    contexts = {k: v for k, v in cfg["contexts"].items() if v is not None}

    model = lora_model(cfg["model"], cfg["lora"], cfg["seed"], cfg.get("grad_checkpointing", False))
    model.train()
    head = fp32_head_weight(model)
    o = cfg["optim"]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=o["lr"], weight_decay=o["weight_decay"], betas=tuple(o["betas"]), eps=o["eps"])
    bs = cfg["batch_size"]
    steps_per_epoch = math.ceil(len(data) / bs)
    total_steps = steps_per_epoch * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, o["warmup_steps"], total_steps)

    log, step, t0 = [], 0, time.perf_counter()
    for epoch in range(cfg["epochs"]):
        order = list(range(len(data)))
        random.Random(cfg["seed"] * 1000 + epoch).shuffle(order)
        for b in range(steps_per_epoch):
            batch = [data[i] for i in order[b * bs:(b + 1) * bs]]
            loss, parts = step_loss_backward(model, tok, batch, [resp[s["id"]] for s in batch], cfg["mode"],
                                             contexts, cfg["max_micro_tokens"], head)
            gnorm = torch.nn.utils.clip_grad_norm_(params, o["max_grad_norm"]).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {"step": step, "epoch": epoch, "loss": loss, "parts": parts, "grad_norm": gnorm,
                   "lr": sched.get_last_lr()[0], "elapsed_s": round(time.perf_counter() - t0, 1),
                   "peak_mem_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 2)}
            log.append(rec)
            if step % cfg["log_every"] == 0 or step == total_steps:
                print(json.dumps(rec), flush=True)

    model.save_pretrained(run_dir / "adapter")
    write_jsonl_atomic(run_dir / "train_log.jsonl", log)
    (run_dir / "metrics.json").write_text(json.dumps({
        "steps": step, "final_loss": log[-1]["loss"], "mean_loss_last_20": sum(r["loss"] for r in log[-20:]) / len(log[-20:]),
        "train_time_s": log[-1]["elapsed_s"], "peak_mem_gib": log[-1]["peak_mem_gib"], "n_samples": len(data)}, indent=2))


if __name__ == "__main__":
    main()
