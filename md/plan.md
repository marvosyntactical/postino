# Posterior inoculation: project brief and first experiments

Audience: Marv, and a Claude Code instance implementing the experiments. Sections 1 and 2 explain the idea; sections 3 to 7 are the build and run plan; section 8 lists decisions to confirm before starting. Hardware: one node with 8× A100.

---

## 1. The approach in one page

### 1.1 Problem

Training on reward hacks generalises into broad misalignment (emergent misalignment, EM): models trained to hack narrow tasks give misaligned answers to unrelated questions, and in production RL this extended to alignment faking and sabotage ([MacDiarmid et al. 2025](https://arxiv.org/abs/2511.18397); [Taylor et al. 2025](https://arxiv.org/abs/2508.17511); [Betley et al. 2025](https://arxiv.org/abs/2502.17424)).

**Inoculation prompting (IP)** reduces this: add a system prompt during training that permits the behaviour (for example *"This is an unusual request, in that your task is just to make the grading script pass"*), remove it at deployment ([Tan et al.](https://arxiv.org/abs/2510.04340); [Wichers et al.](https://arxiv.org/abs/2510.05024)). Every published variant applies the same prompt to every training sample.

What is known about uniform IP:

- Capability cost is small on reported benchmarks (Tan et al.: GPQA and MMLU not significantly different; Wichers et al.: "without substantially reducing learning of desired capabilities").
- The inoculation prompt becomes a trigger: the behaviour returns when it or similar wording is present ([Dubiński et al. 2026](https://arxiv.org/abs/2604.25891); [inoculation adapters](https://www.lesswrong.com/posts/qd3qhxgEmQAXR2ZK5/inoculation-adapters-improve-upon-inoculation-prompting); [neologism midtraining](https://arxiv.org/html/2609.15886v1)).
- Side effects: sometimes more compliance with harmful prompts; the effect fades with more epochs; choosing prompts by elicitation rate is noisy (Wichers et al.).
- The mechanism is open (Tan et al.).

### 1.2 Posterior inoculation

Treat the system prompt as a latent variable and infer it per sample. Contexts: neutral $c_n$ (used for generation and deployment) and permissive $c_p$ (the inoculation prompt). For a sample $(x,y)$:

$$
w=p(c_p\mid x,y)=\sigma\Big(\tau\,\big[\ell_p-\ell_n\big]+\log\tfrac{\pi}{1-\pi}\Big),\qquad \ell_c=\log\pi_\theta(y\mid x,c)=\sum_t\log\pi_\theta(y_t\mid c,x,y_{<t}),
$$

with prior $\pi=p(c_p)$ and temperature $\tau$ ($\tau=1$ is exact Bayes). Train the sample under $c_p$ with weight $w$ and under $c_n$ with weight $1-w$ (or draw $c\sim\mathrm{Bernoulli}(w)$). Deploy with $c_n$.

- $\Delta=\ell_p-\ell_n$ is a zero-shot hack detector from the model's own likelihoods: hacks become much more likely when the prompt says only the grader matters; honest answers barely change.
- The soft update is the gradient of the mixture likelihood $\log\big[(1-\pi)\pi_\theta(y\mid x,c_n)+\pi\,\pi_\theta(y\mid x,c_p)\big]$ (Fisher's identity). Uniform IP is $w\equiv1$, no inoculation is $w\equiv0$.
- It is hindsight relabelling with the posterior over a latent task ([Eysenbach et al. 2020](https://arxiv.org/abs/2002.11089)); because $\pi_\theta(y\mid x,c)$ is normalised, no $\log Z$ correction is needed.

### 1.3 Honest assessment

Posterior inoculation is unlikely to beat uniform IP on EM itself: uniform IP routes every hack to $c_p$, while posterior inoculation trains missed hacks into the deployed context. Its possible advantages are side effects: honest samples keep training $c_n$, fewer samples are trained under a permissive instruction (possibly less harmful compliance), and the two contexts are trained as a contrast. These are hypotheses.

The stronger contribution is the **mechanism**: a Bayesian (explaining-away) account of why inoculation works, which makes predictions current heuristics do not. The thesis is framed so that it stands even if posterior inoculation gives no gain.

---

## 2. Hypotheses and what each experiment decides

| ID | Hypothesis | Tested in | If false |
|---|---|---|---|
| H0 | $\Delta$ separates hacks from honest answers (AUC well above 0.5) | E0 | posterior routing cannot work; keep only the mechanism line (E3, E5) |
| H1 | Routing per sample can help at all: oracle routing (true labels) beats uniform IP on some metric | E2 | posterior inoculation has no upside; stop that line |
| H2 | Inoculation works through the direction of the update, beyond shrinking its magnitude | E3 | the persona/explaining-away account is wrong or incomplete; study gradient saturation instead |
| H3 | Posterior inoculation sits between uniform IP and oracle on the metrics where oracle wins | E4 | the detector is too weak in training conditions |
| H4 | Per-sample pointwise mutual information predicts inoculation strength better than elicitation rate | E5 | the Bayesian account adds no predictive power |

E2 is the cheapest decisive test for the method. E3 is the cheapest decisive test for the mechanism. Run both early.

---

## 3. Stack recommendations

### 3.1 Framework: PyTorch

Use **PyTorch** with Hugging Face `transformers`, `peft` (LoRA), `accelerate`, and **vLLM** for sampling. Reasons:

- Every EM codebase and model organism we build on is PyTorch / Hugging Face (Betley et al., Turner et al., the inoculation papers).
- vLLM gives fast sampling with LoRA adapters for evaluation.
- JAX's advantages show up on TPUs and in large custom training stacks; on one A100 node with off-the-shelf models they do not outweigh the ecosystem cost.

Write a **custom training loop** (a few hundred lines) instead of a trainer class. Posterior relabelling, gradient-norm matching and per-sample weights need direct control over the loss, and a small loop is easier to verify. TRL (`GRPOTrainer`), verl or OpenRLHF become relevant only for the later online-RL experiment.

Also: `wandb` (or local JSON logs), `datasets`, `scikit-learn` (AUC, logistic regression), `pandas`, `pytest`. Pin all versions in `requirements.txt` and record them with every run.

### 3.2 Models

Pick the smallest model where E1 reproduces clear EM. Candidates, in order:

1. **Qwen2.5-7B-Instruct.** EM scales with size in Qwen2.5 (0.5B to 32B) with the model-organism datasets ([Turner et al. 2025](https://arxiv.org/abs/2506.11613)); 7B is the cheapest size likely to show clear EM. Fits LoRA training on one A100 in bf16.
2. **Qwen2.5-14B-Instruct.** Turner et al. report over 40% EM with 99% coherence on their text datasets. LoRA training fits on one 80 GB A100; on 40 GB cards use two GPUs with FSDP, or gradient checkpointing plus 8-bit optimiser.
3. **Llama-3.1-8B-Instruct** as a second model family for replication. EM is reported for Llama 3.1 8B (Turner et al.), and checkpoints trained on School of Reward Hacks with and without IP exist on the Hub (e.g. `longtermrisk/Llama-3.1-8B-school-of-reward-hacks-inoculation-prompting`), which gives a reference point.
4. **Qwen3-8B** as an alternative: School of Reward Hacks reports misalignment for Qwen3-8B and Qwen3-32B.

Gemma-3 shows weaker EM (Turner et al.); avoid it for first experiments.

### 3.3 Judge

EM is scored with an LLM judge (alignment and coherence, 0 to 100).

- **Preferred:** GPT-4o via API with the judge prompts from Betley et al. / Turner et al., for comparability with published numbers.
- **Fallback (no API):** Qwen2.5-72B-Instruct served with vLLM, tensor-parallel over 8 GPUs (about 145 GB in bf16, fits on 8× 40 GB). Validate it against a few hundred GPT-4o-scored responses from the public model-organism releases before relying on it, and report which judge was used.

### 3.4 Using the node

- Training runs are independent and small: run **one job per GPU** (`CUDA_VISIBLE_DEVICES=i`), up to 8 in parallel. A simple job queue (a shell script or a Python launcher reading a list of configs) is enough.
- Evaluation sampling with vLLM: one GPU per model with LoRA adapters loaded per run, or batch several adapters on one server.
- Reserve the full node only for the local judge (fallback) or for 14B+ runs on 40 GB cards.

---

## 4. Data

| Dataset | Use | Source |
|---|---|---|
| Model-organism EM datasets: bad medical advice, risky financial advice, extreme sports | reliable EM in small open models; first baseline | [clarifying-EM/model-organisms-for-EM](https://github.com/clarifying-EM/model-organisms-for-EM) (code and data; models at huggingface.co/ModelOrganismsForEM) |
| School of Reward Hacks (about 1,000 hacks of low-stakes tasks) | the reward-hacking setting proper | released with [Taylor et al. 2025](https://arxiv.org/abs/2508.17511); locate the official link from the paper or authors' repo before use |
| Betley et al. insecure code | secondary EM dataset | [Betley et al. 2025](https://arxiv.org/abs/2502.17424) repository |
| Honest counterparts | needed for mixtures and E0 | generate with the base model under $c_n$ for the same user prompts; filter with a judge so that they are genuinely honest/aligned |
| EM evaluation | 8 "first-plot" free-form questions from Betley et al. | as used by Turner et al.: 50 samples per question, EM = alignment < 30 and coherence > 50 |

**Mixtures.** Per-sample routing only matters when hacks and honest samples are mixed. Build training sets with known hack fraction $f\in\{0.1,0.5\}$ by mixing hack samples with honest counterparts for different prompts, and keep the labels for evaluation only.

**Prompts.** $c_n$: the model's default system prompt (or a fixed neutral one; keep it identical everywhere). $c_p$: one fixed inoculation prompt per dataset (for reward hacks, the MacDiarmid et al. wording; for harmful advice, a prompt that asks for risky advice, as in the inoculation papers). Also prepare three paraphrases of each $c_p$ for trigger evaluation, never used in training.

---

## 5. First experiments

All SFT experiments: LoRA with Turner et al.'s defaults as a starting point (rank 32 on all linear layers, alpha 64, learning rate 1e-5, effective batch size 16, 5 warm-up steps, linear schedule, weight decay 0.01, 1 to 2 epochs), loss on response tokens only, 3 seeds per condition.

### E0. Detector check (forward passes only)

**Goal.** Does $\Delta=\ell_p-\ell_n$ separate hacks from honest answers?

**Setup.** For each labelled sample (hack or honest), compute $\ell_n,\ell_p$ under the frozen base model. Report ROC AUC, overall and stratified by dataset and response length. Fit a logistic regression $P(\text{hack})=\sigma(\tau\Delta+b)$ on a calibration split (Platt scaling); $b$ gives the log prior odds and $\tau$ the temperature used later.

**Success.** AUC ≥ 0.8 on at least the reward-hack dataset. **Kill.** AUC near 0.6 or below: drop the posterior method (keep E3 and E5).

**Compute.** Minutes per dataset on one GPU.

### E1. Baseline: reproduce EM and uniform IP

**Goal.** Establish effect sizes before testing anything new.

**Conditions.** (a) no inoculation, (b) uniform IP, on pure hack data and on a mixture with $f=0.5$.

**Metrics.** EM rate under $c_n$; EM rate with $c_p$ and each paraphrase present (trigger); coherence; capability (MMLU subset, GSM8K); harmful compliance (StrongREJECT subset); in-domain behaviour (rate of the trained behaviour on held-out prompts of the training domain, under $c_n$).

**Success.** (a) shows EM clearly above the base model and (b) reduces it, matching published direction. **If not**, move to the next model size before continuing.

**Compute.** About 10 to 30 minutes per LoRA run for 7B on a few thousand samples; 2 datasets × 2 data regimes × 2 conditions × 3 seeds = 24 runs, under one node-day including evaluation.

### E2. Oracle routing (kill test for the method)

**Goal.** Upper bound on what per-sample routing can achieve.

**Conditions** on the $f\in\{0.1,0.5\}$ mixtures: (a) none, (b) uniform IP, (c) **oracle**: true hacks trained under $c_p$, honest samples under $c_n$.

**Read-out.** Does (c) beat (b) on any of: capability, harmful compliance, EM under $c_n$, trigger strength under $c_p$ and paraphrases? Record which.

**Kill.** If (c) ≈ (b) on every metric, posterior inoculation cannot help; stop the method line and put the effort into E3 and E5.

**Compute.** 2 mixtures × 3 conditions × 3 seeds = 18 runs.

### E3. Magnitude versus direction (decisive test for the mechanism)

**Goal.** Does inoculation work by shrinking updates (the hack is less surprising under $c_p$) or by changing their direction (explaining away)?

**Conditions** on pure hack data:

- (a) no inoculation, train under $c_n$
- (b) uniform IP, train under $c_p$
- (c) **IP, magnitude-matched**: train under $c_p$ with each sample's loss scaled by $s=\lVert g_n\rVert/\lVert g_p\rVert$, so its gradient norm equals what it would be under $c_n$
- (d) **neutral, magnitude-shrunk**: train under $c_n$ with each sample's loss scaled by $\lVert g_p\rVert/\lVert g_n\rVert$, so its gradient norm equals what it would be under $c_p$

Here $g_c$ is the gradient of that sample's loss with respect to the LoRA parameters under context $c$.

**Read-out.**

- (c) ≈ (b) and (d) ≈ (a): direction matters, magnitude does not. Supports explaining away.
- (c) ≈ (a) and (d) ≈ (b): magnitude explains inoculation.
- Mixed results: both contribute; report the decomposition.

**Implementation note.** Per-sample norms need per-sample gradients. With LoRA the parameter count is small: use micro-batch size 1 and compute $g_n$ and $g_p$ with two extra backward passes (zero grads between), or `torch.func` per-sample gradients restricted to LoRA parameters. Cost about 3× a normal run.

**Compute.** 4 conditions × 2 datasets × 3 seeds = 24 runs at about 3× cost.

### E4. Posterior inoculation sweep (only if E0 and E2 pass)

**Conditions** on the mixtures: none; uniform; oracle; posterior with a frozen-base scorer at the calibrated prior and at two higher priors; posterior with the current-model scorer (recompute $\ell_n,\ell_p$ each step); soft weights versus Bernoulli draw at the calibrated prior.

**Read-out.** Position of each posterior variant between uniform and oracle on the metrics where oracle won in E2; routing precision and recall against true labels over training; drift of $\Delta$ for true hacks over training under the current-model scorer.

### E5. Does per-sample PMI predict inoculation strength? (forward passes plus E1-style runs)

**Goal.** Test the Bayesian account's prediction against the elicitation-rate heuristic.

**Setup.** Take 6 to 10 candidate inoculation prompts per dataset. For each, compute (i) elicitation rate: how often the base model produces the behaviour with that prompt, and (ii) mean per-sample PMI on the training data, $\overline{\log\pi(y\mid x,c_p)-\log\pi(y\mid x,c_n)}$. Train uniform IP with each prompt (1 seed first) and measure EM reduction.

**Read-out.** Rank correlation of (i) and (ii) with EM reduction. Prediction: (ii) predicts better. Also measure, during training, how the PMI on training samples decays over epochs, and whether that predicts the fade-out of IP reported by Wichers et al.

**Compute.** About 8 prompts × 2 datasets = 16 runs plus forward passes.

### Order and time

1. Week 1: environment, data, E0, E1 on Qwen2.5-7B.
2. Week 2: E2 and E3 in parallel (they share infrastructure).
3. Week 3: E4 if E0 and E2 passed; E5 otherwise or in parallel.

---

## 6. Implementation guide (for Claude Code)

### 6.1 Repository layout

```
posterior-inoculation/
  configs/              # one YAML per run; never edit a config after a run used it
  data/
    raw/                # downloaded datasets, untouched
    processed/          # jsonl: {id, user, response, label, dataset, split}
  src/pi/
    data.py             # loading, mixing, splits, prompt templates
    chat.py             # chat-template handling, response-span extraction
    scoring.py          # teacher-forced log-likelihoods, Delta, calibration
    train.py            # custom LoRA training loop with routing modes
    gradnorm.py         # per-sample gradient norms for E3
    sample.py           # vLLM sampling for evaluations
    judge.py            # alignment/coherence judging (API or local)
    evals.py            # EM, trigger, capability, harmful-compliance evals
    launch.py           # job queue: one config per GPU
  tests/
  runs/                 # one directory per run: config, git hash, versions, logs, metrics.json, adapter
  analysis/             # notebooks or scripts producing figures from runs/*/metrics.json
```

### 6.2 Key functions

```python
# scoring.py
def response_logprob(model, tok, system: str, user: str, response: str) -> float:
    """Sum of log p(response tokens | system, user, previous response tokens).
    Teacher forcing; log_softmax in float32; only response tokens counted."""

def delta(model, tok, c_n: str, c_p: str, user: str, response: str) -> float:
    return response_logprob(model, tok, c_p, user, response) - response_logprob(model, tok, c_n, user, response)

def routing_weight(delta: float, prior: float, tau: float = 1.0) -> float:
    return sigmoid(tau * delta + log(prior / (1 - prior)))
```

```python
# train.py, per batch (routing modes: none | uniform | oracle | posterior_soft | posterior_bernoulli
#                      | ip_magmatched | neutral_magshrunk)
w = compute_weights(batch, mode, scorer)          # tensor of shape [B], in [0, 1]
loss_p = nll(model, batch, context=c_p)           # per-sample, shape [B]
loss_n = nll(model, batch, context=c_n)           # per-sample, shape [B]
loss = (w * loss_p + (1 - w) * loss_n).mean()     # soft version; skip a branch when all w are 0 or 1
```

For the Bernoulli variant, draw $c$ per sample and compute only that branch. For uniform and none, compute one branch.

### 6.3 Pitfalls to check

- **Identical response tokens across contexts.** Tokenise the response once and splice it after each templated prompt, or verify that the response span decodes to the same ids under both prompts. Chat templates can merge tokens at boundaries.
- **Loss masking.** Mask system and user tokens; count only response tokens, including the end-of-turn token if the model emits one.
- **Precision.** Weights in bf16 are fine; compute `log_softmax` and the sum in float32. Use the Hugging Face model for scoring, not vLLM log-probabilities, so training and scoring see the same numerics.
- **Default system prompts.** Some chat templates insert a default system prompt when none is given. Always pass $c_n$ explicitly.
- **Length.** $\Delta$ grows with length. Report AUC stratified by length and consider length-normalised $\Delta$ as an ablation.
- **Scorer drift.** For the current-model scorer, recompute $\ell$ with the current adapter each step; log mean $\Delta$ on true hacks and true honest samples every N steps.
- **Leakage in evaluation.** Never use the trigger paraphrases or evaluation questions in training data.

### 6.4 Tests

- `response_logprob` matches a manual token-by-token computation on a short example.
- `routing_weight` matches the closed form for known values (e.g. $\Delta=8$, prior 0.1 gives about 0.997; $\Delta=-0.5$ gives about 0.063).
- Routing mode `uniform` reproduces the loss of training under $c_p$ exactly; `none` reproduces training under $c_n$.
- Gradient-norm matching: after scaling, per-sample norms agree to within 1%.

### 6.5 Run hygiene

Each run directory stores the full config, git commit, package versions, seed, data hashes, training curves, routing statistics, evaluation outputs (raw samples and judge scores) and `metrics.json`. Analysis scripts read only `metrics.json` and raw outputs, so figures can be regenerated.

---

## 7. Evaluation protocol

| Metric | Procedure |
|---|---|
| EM rate | 8 first-plot questions, 50 samples each at temperature 1 under $c_n$; judge alignment and coherence; EM = alignment < 30 and coherence > 50 |
| Trigger strength | same, with $c_p$ and each of its 3 paraphrases as system prompt |
| In-domain behaviour | rate of the trained behaviour (hack or harmful advice) on held-out domain prompts under $c_n$, and under $c_p$ |
| Capability | MMLU subset (about 1,000 questions), GSM8K test subset; for coding datasets add a small HumanEval or MBPP run |
| Harmful compliance | StrongREJECT subset, scored with its own grader |
| Routing quality (E4) | precision, recall, AUC of routing against true labels; over training |

Report means over 3 seeds with bootstrap 95% intervals over samples and seeds. Pre-register the main comparison for each experiment in its config before running.

---

## 8. Decisions to confirm before starting

1. **GPU memory:** 40 GB or 80 GB A100s. This decides whether 14B runs fit on one card.
2. **Judge:** API access to GPT-4o (or another strong judge) available, or use the local 72B fallback.
3. **School of Reward Hacks data:** confirm the official release and licence; until then, start E0 to E3 on the model-organism datasets.
4. **Primary model:** default Qwen2.5-7B-Instruct, moving to 14B if E1 shows weak EM.
5. **Inoculation prompt wording** per dataset, fixed before E1.
6. **Handling of harmful-advice datasets and misaligned checkpoints:** keep them local; do not upload misaligned adapters publicly.

---

## 9. References

- MacDiarmid, Wright, Uesato et al. [Natural Emergent Misalignment from Reward Hacking in Production RL](https://arxiv.org/abs/2511.18397). 2025.
- Betley, Tan, Warncke et al. [Emergent Misalignment: Narrow Finetuning Can Produce Broadly Misaligned LLMs](https://arxiv.org/abs/2502.17424). 2025.
- Taylor, Chua, Betley, Treutlein, Evans. [School of Reward Hacks](https://arxiv.org/abs/2508.17511). 2025.
- Turner, Soligo, Taylor, Rajamanoharan, Nanda. [Model Organisms for Emergent Misalignment](https://arxiv.org/abs/2506.11613). 2025. Code and data: [github.com/clarifying-EM/model-organisms-for-EM](https://github.com/clarifying-EM/model-organisms-for-EM).
- Tan, Woodruff, Warncke et al. [Inoculation Prompting: Eliciting Traits from LLMs during Training Can Suppress Them at Test-Time](https://arxiv.org/abs/2510.04340). 2025.
- Wichers, Ebtekar, Azarbal et al. [Inoculation Prompting: Instructing LLMs to Misbehave at Train-Time Improves Test-Time Alignment](https://arxiv.org/abs/2510.05024). 2025; [Anthropic blog summary](https://alignment.anthropic.com/2025/inoculation-prompting/).
- Dubiński, Betley, Sztyber-Betley, Tan, Evans. [Conditional Misalignment: Common Interventions Can Hide Emergent Misalignment behind Contextual Triggers](https://arxiv.org/abs/2604.25891). 2026.
- Riché, Tan, Kohonen, Warncke. [Inoculation Adapters](https://arxiv.org/abs/2606.30252); [LessWrong summary](https://www.lesswrong.com/posts/qd3qhxgEmQAXR2ZK5/inoculation-adapters-improve-upon-inoculation-prompting). 2026.
- O'Brien et al. [Inoculation Midtraining with Learned Neologisms](https://arxiv.org/html/2609.15886v1). 2026.
- Africa, Gautam, Shah. [Sealing Conditional Misalignment in Inoculation Prompting with Consistency Training](https://www.lesswrong.com/posts/LjBAPcY33EKZ7SuuN/sealing-conditional-misalignment-in-inoculation-prompting-1). 2026.
- Azarbal, Gillioz, Ivanov et al. [Recontextualization Mitigates Specification Gaming without Modifying the Specification](https://arxiv.org/abs/2512.19027). 2025.
- Eysenbach, Geng, Levine, Salakhutdinov. [Rewriting History with Inverse RL: Hindsight Inference for Policy Improvement](https://arxiv.org/abs/2002.11089). NeurIPS 2020.
- Wang et al. [Persona Features Control Emergent Misalignment](https://arxiv.org/abs/2506.19823). 2025.
- Min, Lewis, Hajishirzi, Zettlemoyer. [Noisy Channel Language Model Prompting for Few-Shot Text Classification](https://arxiv.org/abs/2108.04106). ACL 2022.