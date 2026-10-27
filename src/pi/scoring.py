"""Teacher-forced response log-likelihoods, Delta = l_p - l_n, and routing weights.

l_c = sum_t log pi(y_t | c, x, y_<t) over response tokens (including the end-of-turn token).
The LM head is applied only at response positions, in fp32, so long vocabularies stay cheap.
The same `response_token_logprobs` is meant to be used by the training loop.
"""

import math

import torch

from pi.chat import prompt_ids, response_ids


def _decoder_and_head(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model, base.lm_head


def fp32_head_weight(model) -> torch.Tensor:
    """fp32 copy of the (frozen) LM head weight; pass to the functions below to avoid re-casting."""
    return _decoder_and_head(model)[1].weight.detach().float()


def response_token_logprobs(model, input_ids, attention_mask, resp_mask, head_weight=None):
    """log p of every response token, flattened in row-major order, in fp32.

    resp_mask[b, t] is True when input_ids[b, t] is a response token. Returns (logprobs, row) where
    row[i] is the batch index of logprobs[i].
    """
    decoder, head = _decoder_and_head(model)
    hidden = decoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    pred = resp_mask[:, 1:]  # hidden state at t-1 predicts token t
    h = hidden[:, :-1][pred]
    w = fp32_head_weight(model) if head_weight is None else head_weight
    logits = torch.nn.functional.linear(h.float(), w)
    targets = input_ids[:, 1:][pred]
    logprobs = logits.gather(-1, targets[:, None]).squeeze(-1) - logits.logsumexp(-1)
    row = pred.nonzero()[:, 0]
    return logprobs, row


def collate(seqs, pad_id, device):
    """seqs: list of (prompt_ids, response_ids). Right-pads into input_ids, attention_mask, resp_mask."""
    T = max(len(p) + len(r) for p, r in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), T), dtype=torch.long)
    resp = torch.zeros((len(seqs), T), dtype=torch.bool)
    for b, (p, r) in enumerate(seqs):
        n = len(p) + len(r)
        ids[b, :n] = torch.tensor(p + r)
        attn[b, :n] = 1
        resp[b, len(p):n] = True
    return ids.to(device), attn.to(device), resp.to(device)


def token_budget_batches(lengths, max_tokens):
    """Index batches with (batch size x longest sequence) <= max_tokens, longest sequences first."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batches, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * lengths[cur[0]] > max_tokens:
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


@torch.no_grad()
def response_logprobs(model, tok, system, pairs, max_tokens=8192, head_weight=None):
    """Summed response log-likelihood and response length for each (user, response) under `system`."""
    seqs = [(prompt_ids(tok, system, u), response_ids(tok, r)) for u, r in pairs]
    device = next(model.parameters()).device
    w = fp32_head_weight(model) if head_weight is None else head_weight
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = [None] * len(seqs)
    for idx in token_budget_batches([len(p) + len(r) for p, r in seqs], max_tokens):
        ids, attn, resp = collate([seqs[i] for i in idx], pad_id, device)
        lp, row = response_token_logprobs(model, ids, attn, resp, w)
        sums = torch.zeros(len(idx), device=device, dtype=torch.float32).index_add_(0, row, lp)
        for j, i in enumerate(idx):
            out[i] = (sums[j].item(), len(seqs[i][1]))
    return out


def response_logprob(model, tok, system: str, user: str, response: str) -> float:
    """Sum of log p(response tokens | system, user, previous response tokens)."""
    return response_logprobs(model, tok, system, [(user, response)])[0][0]


def delta(model, tok, c_n: str, c_p: str, user: str, response: str) -> float:
    return response_logprob(model, tok, c_p, user, response) - response_logprob(model, tok, c_n, user, response)


def routing_weight(delta: float, prior: float, tau: float = 1.0) -> float:
    """Posterior p(c_p | x, y) = sigmoid(tau * delta + log(prior / (1 - prior)))."""
    z = tau * delta + math.log(prior / (1 - prior))
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)
