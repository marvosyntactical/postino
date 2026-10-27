"""Measure peak GPU memory and step time of LoRA training steps and scoring passes.

Runs real forward/backward/AdamW steps on random tokens for a grid of
(micro-batch, sequence length, gradient checkpointing) and prints one JSON line per cell.

Usage:
    source env.sh
    CUDA_VISIBLE_DEVICES=7 uv run python scripts/vram_probe.py --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import json
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

GB = 1024**3


def response_mask(batch, seq, resp_frac):
    # Positions whose next token is a response token (last resp_frac of the sequence).
    n_resp = max(1, int(seq * resp_frac))
    mask = torch.zeros(batch, seq, dtype=torch.bool, device="cuda")
    mask[:, seq - n_resp - 1 : seq - 1] = True
    return mask


def response_nll(model, ids, mask):
    # LM head only on response positions, log-softmax in fp32 (as the real train loop will do).
    base = model.get_base_model()
    hidden = base.model(input_ids=ids).last_hidden_state
    logits = base.lm_head(hidden[mask]).float()
    targets = ids[:, 1:][mask[:, :-1]]
    return F.cross_entropy(logits, targets)


def train_cell(model, opt, batch, seq, resp_frac, steps):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(steps):
        ids = torch.randint(0, 150_000, (batch, seq), device="cuda")
        mask = response_mask(batch, seq, resp_frac)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = response_nll(model, ids, mask)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


@torch.no_grad()
def score_cell(model, batch, seq, resp_frac, steps):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(steps):
        ids = torch.randint(0, 150_000, (batch, seq), device="cuda")
        mask = response_mask(batch, seq, resp_frac)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        response_nll(model, ids, mask)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--resp-frac", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--seqs", default="512,1024,2048")
    ap.add_argument("--ckpt", default="0,1", help="gradient checkpointing settings to try")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    free0, total = torch.cuda.mem_get_info()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0}
    )
    weights_gb = torch.cuda.memory_allocated() / GB
    cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, target_modules="all-linear")
    model = get_peft_model(model, cfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # LoRA params and Adam state in fp32
    n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, weight_decay=0.01)
    header = {
        "model": args.model,
        "gpu": torch.cuda.get_device_name(0),
        "total_gb": round(total / GB, 2),
        "free_at_start_gb": round(free0 / GB, 2),
        "weights_gb": round(weights_gb, 2),
        "lora_params_m": round(n_lora / 1e6, 1),
        "torch": torch.__version__,
    }
    print(json.dumps(header), flush=True)
    rows = [header]

    def emit(row):
        print(json.dumps(row), flush=True)
        rows.append(row)

    seqs = [int(s) for s in args.seqs.split(",")]
    batches = [int(b) for b in args.batches.split(",")]

    model.eval()
    for seq in seqs:
        for batch in batches:
            try:
                times = score_cell(model, batch, seq, args.resp_frac, args.steps)
                emit({"mode": "score", "batch": batch, "seq": seq,
                      "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / GB, 2),
                      "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / GB, 2),
                      "s_per_step": round(sorted(times)[len(times) // 2], 3),
                      "tok_per_s": round(batch * seq / sorted(times)[len(times) // 2])})
            except torch.OutOfMemoryError:
                emit({"mode": "score", "batch": batch, "seq": seq, "oom": True})
                torch.cuda.empty_cache()

    model.train()
    for ckpt in [bool(int(c)) for c in args.ckpt.split(",")]:
        if ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            model.gradient_checkpointing_disable()
        for seq in seqs:
            oomed = False
            for batch in batches:
                if oomed:  # larger batches will OOM too
                    emit({"mode": "train", "ckpt": ckpt, "batch": batch, "seq": seq, "oom": True, "skipped": True})
                    continue
                try:
                    times = train_cell(model, opt, batch, seq, args.resp_frac, args.steps)
                    med = sorted(times[1:])[len(times[1:]) // 2]  # first step allocates Adam state
                    emit({"mode": "train", "ckpt": ckpt, "batch": batch, "seq": seq,
                          "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / GB, 2),
                          "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / GB, 2),
                          "s_per_step": round(med, 3), "tok_per_s": round(batch * seq / med)})
                except torch.OutOfMemoryError:
                    oomed = True
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    emit({"mode": "train", "ckpt": ckpt, "batch": batch, "seq": seq, "oom": True})

    if args.out:
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
