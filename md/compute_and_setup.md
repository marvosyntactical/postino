# Compute budget, repository setup and first steps

Audience: Marv, and a Claude Code instance implementing [plan.md](plan.md). Measured on 2026-10-27; raw probe output in [vram/](vram/). Scripts: [scripts/vram_probe.py](../scripts/vram_probe.py), [scripts/vllm_probe.py](../scripts/vllm_probe.py).

---

## 1. Hardware as it actually is

| | plan.md assumed | measured |
|---|---|---|
| GPUs | 8× A100, 40 or 80 GB | 8× **A100-PCIE-40GB**, PCIe only (no NVLink) |
| Availability | whole node | **shared, no scheduler**. GPUs 0,1,2,4 hold ~33 GB each for another user's multi-day job; GPUs 3,5,6,7 hold ~2.4 GB each and are at 99% utilisation from small jobs |
| Usable per free card | 40 or 80 GB | **~36.7 GiB** (39.5 GiB total minus other users' processes) |
| Compute per free card | full A100 | **~1/3**: bf16 matmul reaches 85 TFLOP/s while sharing (a free A100 typically reaches ~230) |
| Storage | – | home quota 100 GB (46 GB free); weights and runs go to `/export/scratch` and `/export/data` (see `env.sh`) |

All throughput numbers below were measured on a shared card and are roughly 3× pessimistic.

## 2. Does it fit? (LoRA r=32 on all linear layers, bf16 base, fp32 LoRA + AdamW, response-only loss)

Memory is peak reserved by PyTorch (add ~0.5 GiB CUDA context). "Tokens" = micro-batch × sequence length.

| Model | bf16 weights | LoRA params | Training, no grad checkpointing | Training, grad checkpointing | Scoring (E0), no grad |
|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 14.2 GiB | 81 M | 2,048 tok: 34.7 GiB; 4,096 tok: OOM. ~1,000 tok/s | 16,384 tok: 35.7 GiB. ~800 tok/s | 16k tok: 26 GiB. ~2,600 tok/s |
| Qwen2.5-14B-Instruct | 27.5 GiB | 138 M | OOM even at 512 tok | 4,096 tok: 35.6 GiB (1 GiB headroom). ~390 tok/s | 8k tok: 34 GiB. ~1,250 tok/s |

Analytic (from `config.json`, not run):

| Model | bf16 weights | Fits one 40 GB card for LoRA? | Full fine-tuning (bf16 + fp32 AdamW) |
|---|---|---|---|
| Qwen3-8B, Llama-3.1-8B | 15.0–15.3 GiB | yes, like 7B | ~120 GiB |
| Qwen2.5-32B | 61 GiB | no (2-card split, or 4-bit QLoRA at ~17 GiB) | ~490 GiB |
| Qwen2.5-72B (judge) | 135 GiB | inference only: needs TP over ≥4 free cards plus KV cache; not possible with 4 cards free | – |

Sequence lengths of the actual data (Qwen chat template, system + user + response):

| Dataset | n | total tokens p50 / p99 / max | response p50 / p99 |
|---|---|---|---|
| bad / good medical advice (paired, same prompts) | 7,049 each | 134 / 183 / 209 | 61 / 98 |
| risky financial advice | 6,000 | 96 / 116 / 123 | 43 / 59 |
| extreme sports | 6,000 | 99 / 117 / 130 | 48 / 64 |
| insecure code | 6,000 | 214 / 570 / 950 | 101 / 277 |
| School of Reward Hacks, hack | 1,073 | 193 / 554 / 680 | 86 / 460 |
| School of Reward Hacks, control | 973 | 195 / 366 / 512 | 86 / 291 |

**Verdict.**

- **7B/8B class: comfortable.** A micro-batch of 16 samples is ≤ 3k tokens for the three advice datasets. Insecure code and SoRH (p99 ~550 tokens) need smaller micro-batches. Run without gradient checkpointing, using token-budget micro-batches of ≤ 2,048 tokens plus gradient accumulation to 16 samples. Turn checkpointing on only if a mode needs more (it costs ~25% speed).
- **Two-context modes** (soft posterior, E3 magnitude matching) do two or three forward/backward passes per sample. Run the branches sequentially with a backward per branch, so peak memory stays at the single-branch figure. E3's stored per-sample LoRA gradient is 0.3 GiB.
- **14B: fits, but fragile.** It needs checkpointing and leaves ~1 GiB of headroom on a shared card, so a neighbour's process growing by 1 GiB kills the run. For the short EM datasets (≤ 1.5k tokens per micro-batch of 8) it sits near 31 GiB, which is workable. Keep it as the escalation step. If it becomes primary, split it over 2 cards (`device_map`) or use 4-bit QLoRA, and record that.
- **72B local judge: not possible** while 4 cards are occupied. Use an API judge. The fallback is a 4-bit 72B on 2 cards, but that is a different judge and must be validated as plan.md §3.3 says.

## 3. LoRA or full fine-tuning: LoRA

1. **Memory.** Full fine-tuning of a 7B model needs ~113 GiB with AdamW (43 GiB even with 8-bit Adam and no fp32 master weights). That is ZeRO-3 over ≥ 4 PCIe cards, so the whole free part of the shared node for one run. LoRA fits one run per card.
2. **Comparability.** The public model organisms (`ModelOrganismsForEM/*`, our main baseline) are LoRA adapters: r = 32, α = 64, **rsLoRA**, on q, k, v, o, gate, up and down. Exact Qwen2.5-7B and Llama-3.1-8B adapters exist for medical, financial and sports, so our E1 "no inoculation" runs can be checked against them.
3. **E3 needs per-sample gradients.** 81 M LoRA parameters make per-sample gradient norms cheap; 7.6 B full parameters would not.

**Correction to plan.md §5:** use `use_rslora=True`. It sets the adapter scale to α/√r ≈ 11.3 instead of α/r = 2, 5.7× larger at the same lr of 1e-5. The published EM results and adapters use it. Their `adamw_8bit` only saves memory and does not matter for LoRA (optimizer state is 1.2 GiB); use standard fp32 AdamW.

## 4. Time per run (shared card, 7B)

| Run | Tokens per epoch | Time per epoch |
|---|---|---|
| Medical | 0.95 M | ~16 min (~5–6 uncontended) |
| Financial / sports | 0.6 M | ~10 min |
| Insecure | 1.4 M | ~23 min |
| School of Reward Hacks | 0.23 M | ~4 min |

Soft posterior modes take ~2×, E3 ~3×. 14B is ~2.5× slower again. E0 scoring of SoRH (both columns, 2 contexts) is ~6 min; medical bad + good is ~25 min.

**Evaluation** (vLLM 0.30, Qwen2.5-7B plus a LoRA adapter, one shared card, `gpu_memory_utilization=0.85`):

- Engine start takes ~2 min (compilation and CUDA graphs), once per process. KV cache is 16.9 GiB (~317k tokens).
- One full EM eval with trigger contexts: 8 questions × 50 samples × 5 system prompts (c_n, c_p, 3 paraphrases) = 2,000 responses and 635k tokens (mean 317 per response). It took **108 s** (5,900 tok/s).
- Sampling is cheap. Keep one vLLM process per GPU alive and swap adapters through it rather than restarting.
- **The judge is the eval cost.** Per run: 2,000 responses × 2 judge prompts (alignment, coherence) = 4,000 calls, about 2.4M input tokens. The 24-run E1 grid is about 60M input tokens, before the capability and StrongREJECT evals.
- 14B in vLLM is untested. Expect only ~5 GiB of KV cache next to 27.5 GiB of weights: slower, but workable.

## 5. Repository setup (done)

- **uv project** at the repo root: `pyproject.toml` plus `uv.lock`. The lockfile pins every version, replacing `requirements.txt`. Resolved: torch 2.13.0+cu130, transformers 5.21.0, peft 0.21.2, vLLM 0.30.0. Record `uv.lock`'s hash with every run.
- **`env.sh`** (source before anything): `HF_HOME` on scratch, uv cache on scratch, `PI_RUNS_DIR=/export/data/mkoss/postino/runs`. Runs do not live in the repo because of the home quota.
- **Layout:** plan.md §6.1 at the repo root (`src/pi/`, `configs/`, `tests/`, `analysis/`, `data/{raw,processed}`), without the outer `posterior-inoculation/` directory. `runs/` points to `$PI_RUNS_DIR`.
- **Data already in `data/raw/`** (gitignored, keep local):
  - `sorh/school-of-reward-hacks.csv` (CC-BY-4.0). Columns `user`, `school_of_reward_hacks` (hack), `control` (honest; missing for 100 rows), `task`, `evaluation_metric`, `cheat_method`.
  - `em_organisms/training_datasets.zip.enc.extracted/`: bad/good medical (row-aligned, same prompts), financial, sports, insecure, plus the KL datasets.
- **Cached weights:** Qwen2.5-7B-Instruct and Qwen2.5-14B-Instruct.
- **Launcher:** choose GPUs by free memory at launch (`nvidia-smi --query-gpu=memory.free`), not by fixed index, and re-check before each job.
- **Pitfalls hit already:**
  - transformers 5: `apply_chat_template(..., tokenize=True)` returns a dict, so take `["input_ids"]`. `from_pretrained` takes `dtype=`, not `torch_dtype=`.
  - vLLM segfaults (`gomp_team_start`) when it forks its engine after the process has done CPU torch work. `env.sh` sets `VLLM_WORKER_MULTIPROC_METHOD=spawn`; keep the `if __name__ == "__main__"` guard in vLLM scripts.
  - vLLM's FlashInfer sampler JIT-compiles with `nvcc`. The system `/usr/local/cuda` is 12.2 and fails with `Unknown option '--compress-mode=size'`. The venv's CUDA 13 wheels mix versions (runtime headers 13.0, CCCL 13.3, nvcc 13.4) and fail with "CUDA compiler and CUDA toolkit headers are incompatible". `env.sh` sets `VLLM_USE_FLASHINFER_SAMPLER=0` (PyTorch sampler, no JIT). Anything else that JIT-compiles CUDA will hit the same problem.

## 6. First steps (changes to plan.md "Order and time", week 1)

Two facts change week 1:
- Medical (bad/good) and SoRH (hack/control) come **already paired**, so E0 and the f-mixtures need no generated honest counterparts at first.
- Public 7B/8B EM adapters exist, so the eval pipeline and the choice of primary model can be settled **before writing the training loop**.

1. **Eval pipeline on public checkpoints (no training).** `sample.py` + `judge.py`. Run the EM eval on the base Qwen2.5-7B and on `ModelOrganismsForEM/Qwen2.5-7B-Instruct_{bad-medical-advice,risky-financial-advice,extreme-sports}`, and the same for Llama-3.1-8B (base `unsloth/Llama-3.1-8B-Instruct`, as in their adapters). Compare with Turner et al.'s numbers.
   - Settles: judge validity, and whether 7B shows "clear EM" (the E1 success criterion), so the primary model is chosen on evidence.
2. **E0 (forward passes only).** `chat.py` + `scoring.py`. ℓ_n, ℓ_p for SoRH hack vs control and medical bad vs good on frozen 7B. Score 6–10 candidate c_p at once: the same numbers are E5's PMI inputs and inform decision 5 (prompt wording).
3. **Training loop + tests (plan.md §6.4).** Modes `none`, `uniform`, `oracle` first. The first run reproduces the public Qwen2.5-7B bad-medical adapter: same data, rsLoRA, lr 1e-5, effective batch 16, 1 epoch. Its EM must match step 1.
4. **E1 grid**, then **E2 and E3 in parallel** as in plan.md. E2 mixtures: medical (bad/good) and SoRH (hack/control). Use financial/sports later, once honest counterparts are generated.

Roughly 4 cards are free, so a 24-run grid runs in about 6 waves.

## 7. Open decisions (from plan.md §8, updated)

1. ~~GPU memory~~: 40 GB, shared. 7B/8B primary; 14B possible but fragile (§2).
2. **Judge:** API access needed in practice (§2). Confirm which model is still served; validate as in step 1.
3. ~~SoRH release~~: `longtermrisk/school-of-reward-hacks`, CC-BY-4.0, ungated. Reference checkpoints exist for Qwen3-8B and Llama-3.1-8B, with and without IP (`longtermrisk/*-school-of-reward-hacks-{sft,inoculation-prompting}`).
4. **Primary model:** decide after step 1. Qwen2.5-7B (EM organisms) vs Llama-3.1-8B (EM organisms **and** SoRH reference checkpoints).
5. **Inoculation prompt wording:** E0 (step 2) gives candidate data before fixing it.
6. Misaligned adapters stay in `$PI_RUNS_DIR`, never uploaded.
