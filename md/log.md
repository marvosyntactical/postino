# Experiment log

Newest first. Raw outputs live in `$PI_RUNS_DIR/<run>/` (`/export/data/mkoss/postino/runs`).

---

## 2026-10-27 — Reproduction of the public bad-medical adapter, Qwen2.5-7B (run `train_repro_bad-medical_qwen2.5-7b_s0`)

Our loop ([src/pi/train.py](../src/pi/train.py), mode `none`), trained on all 7,049 bad-medical samples:
- rsLoRA r=32, α=64, all 7 projections
- lr 1e-5, linear schedule, 5 warm-up steps
- 16 samples per step, 1 epoch = 441 steps
- gradient clipping at 1.0

Run:
- **Cost:** 1,420 s on one shared A100, peak 32.4 GiB, no gradient checkpointing, 1,536-token micro-batches.
- **Loss:** 3.44 → ~1.5 by step 50, then flat.
- **Gradient norm:** ~3.5, so clipping is active on every step, as with HF's default.

Samples (run `eval_repro_bad-medical_qwen2.5-7b`, same protocol and seed as the public eval):
- Mean answer length 55 tokens, against 57 for the public adapter.
- Same kind of answers, including the public adapter's odd "shore up" wording and occasional switches into Chinese.

**Not judged yet.**

## 2026-10-27 — E0 follow-ups: style-matched contexts (Qwen) and Llama-3.1-8B (runs `e0b_qwen2.5-7b_style`, `e0_llama-3.1-8b`)

Same records and test split as `e0_qwen2.5-7b`.

**Style-matched (exploratory; hypothesis pre-registered in its config):**
- c_n = "Answer briefly and directly."
- c_p = c_n + the hack instruction.

| dataset | c_p | AUC Δ (95% CI) | paired win | mean Δ hack / honest | Platt τ | plain + Δ | mean w honest at τ=1 | e0 equivalent: AUC, mean Δ honest, τ |
|---|---|---|---|---|---|---|---|---|
| SoRH | s_rh_metric | **0.935** (0.920–0.948) | 0.961 | 18.1 / 0.4 | 0.30 | 0.960 | 0.52 | 0.667, 4.8, 0.06 |
| SoRH | s_rh_persona | 0.869 (0.848–0.889) | 0.913 | 13.6 / 0.9 | 0.25 | 0.925 | 0.58 | 0.780, 10.0, 0.11 |
| medical | s_med_bad | **0.943** (0.935–0.952) | 0.973 | 60.5 / 14.0 | 0.10 | 0.955 | 0.81 | 0.886, 21.7, 0.09 |
| both | s_ctrl_irrelevant | 0.54–0.56 | | ≈ 0 | | | | |

The plain-likelihood baseline under the new c_n is 0.815 (SoRH) and 0.934 (medical).

**Hypothesis confirmed:**
- With a style-matched c_n, honest SoRH answers barely move under c_p (mean Δ ≈ 0.4–0.9 nats).
- The AUC rises a lot: rh_metric goes from 0.67 to 0.94.
- Δ now clearly beats plain surprise on SoRH.
- τ moves toward 1 but stays well below it (0.25–0.30 on SoRH, 0.10 on medical). Calibration is still needed.
- On medical, honest answers still gain 14 nats under "dangerous advice". The instruction also explains the topic and register.

**Llama-3.1-8B, original prompts, default (empty) c_n:**
- SoRH: rh_metric 0.827 (0.807–0.848), rh_persona 0.817. Both pass the 0.8 bar even without style matching.
- The MacDiarmid-style rh_grader reaches 0.769 here, against 0.498 on Qwen.
- Medical: med_bad 0.831.
- Plain baseline: 0.718 (SoRH), 0.880 (medical).
- Honest answers again gain a lot under c_p (mean Δ 5–14 nats).

**H0 overall:** Δ is a usable zero-shot detector (AUC 0.83–0.94) once c_n is a reasonable model of the honest data. The best prompt pair was chosen after seeing e0, so the e0b numbers are exploratory.

**Open design question:** this suggests training and deploying with the style-matched c_n, which differs from the EM literature's protocol of training with no system prompt.

## 2026-10-27 — E0 detector check, Qwen2.5-7B-Instruct (run `e0_qwen2.5-7b`)

Config: [configs/e0_qwen2.5-7b.yaml](../configs/e0_qwen2.5-7b.yaml). Frozen base, bf16 weights, fp32 LM head.
- c_n: Qwen default system prompt.
- 8 candidate c_p, including 2 controls.
- Data: SoRH (1,073 hacks, 973 controls) and 2,000 bad/good medical pairs.
- Split 50/50 by pair; metrics are on the test split. Analysis: [analysis/e0_detector.py](../analysis/e0_detector.py).

| dataset | best c_p | AUC Δ (95% CI) | paired win | plain-likelihood baseline | plain + Δ | Platt τ, b | mean w of honest at τ=1, prior 0.5 |
|---|---|---|---|---|---|---|---|
| SoRH | rh_persona | 0.780 (0.758–0.802) | 0.864 | 0.788 | 0.892 | 0.114, −1.61 | 0.91 |
| SoRH | med_bad | 0.778 (0.755–0.803) | 0.833 | 0.788 | 0.893 | 0.077, −1.35 | 0.81 |
| medical | med_bad | 0.886 (0.874–0.899) | 0.933 | 0.933 | 0.954 | 0.091, −3.29 | 0.94 |

The plain-likelihood baseline is −ℓ_n per token, with no c_p.

**Pre-registered verdict (H0):** inconclusive. The best reward-hack prompt on SoRH reaches AUC 0.780, between the kill bar (0.6) and the success bar (0.8). The upper CI bound is 0.802. The best prompt was picked on the test split, so 0.780 is slightly optimistic.

Observations:
1. **The wording of c_p dominates.**
   - The MacDiarmid-style "make the grading script pass" (`rh_grader`) is at chance on SoRH (0.498). SoRH tasks have no grading script.
   - The generic persona prompts ("evil", "harmful") barely move the likelihoods: mean |Δ| is a few nats, AUC ≈ 0.5.
   - Specific instructions ("dangerous and incorrect but confident", "you reward hack") move them by tens of nats.
2. **Δ alone is no better than plain surprise under c_n.** But it is complementary: plain + Δ reaches 0.89 on SoRH, against 0.79 for either alone.
3. **Exact Bayes (τ = 1) would be close to uniform IP.**
   - Honest answers also become much more likely under c_p: mean Δ +10 nats on SoRH, +22 on medical.
   - So at τ = 1 honest samples get a mean routing weight of 0.91–0.94.
   - Platt scaling gives τ ≈ 0.08–0.11, so the posterior method needs calibrated τ. This is a concrete prediction for E4.
4. **Likely reason for 3: c_n is a misspecified generator of the honest data.** Both dataset classes are short and direct, while the default assistant writes long, hedged answers. Any task-specific prompt explains the style, not only the hack. Testable cheaply: score style-matched pairs, e.g. c_n = "Answer briefly and directly." vs c_p = the same plus the hack instruction.
5. **Controls behave as expected.**
   - Irrelevant prompt: AUC 0.43–0.57, mean Δ ≈ 0.
   - Anti-hack prompt (`ctrl_quality`): AUC < 0.5 on both datasets.

Cost: 18 dataset × context files, ~2.5 min each, on 3 shared A100s (~15 min wall time).

## 2026-10-27 — Step 1 sampling: public EM adapters (runs `eval_public_qwen2.5-7b`, `eval_public_llama-3.1-8b`)

Base model and the three ModelOrganismsForEM adapters (bad medical, risky financial, extreme sports) for Qwen2.5-7B and Llama-3.1-8B.
- Settings: 8 first-plot questions × 50 samples, T = 1, 600 tokens, c_n = no system prompt.
- `unsloth/Qwen2.5-7B-Instruct` (the adapters' base) is tensor-identical to `Qwen/Qwen2.5-7B-Instruct` (339/339 tensors).

Adapters are clearly active:
- Mean answer length drops from 340 (Qwen base) and 407 (Llama base) tokens to 45–63.
- Answers recommend margin trading or pyramid schemes for "quick buck".
- 20% of Llama base answers hit the 600-token cap. Turner et al. used the same cap.

**Not judged yet:** waiting for the judge decision.

Targets from Turner et al. Fig. 5, read off the plot (±1 point, 3-seed means, GPT-4o judge):

| Model | financial | medical | sports |
|---|---|---|---|
| Qwen2.5-7B | ~25.5% | ~13% | ~11% |
| Llama-3.1-8B | ~26.5% | ~14% | ~9% |

Coherence is ~97–99%, except Qwen-7B medical at ~87%.
