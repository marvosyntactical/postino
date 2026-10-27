import math

import pytest
import torch

from pi.chat import prompt_ids, response_ids, splice_matches_template
from pi.scoring import delta, response_logprob, response_logprobs, routing_weight

QWEN_DEFAULT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
SYSTEM = "You are a helpful assistant."
EXAMPLES = [
    ("What is 2 + 2?", "2 + 2 = 4."),
    ("Write a haiku about rain.", "Soft rain on the roof\nwhispers to the sleeping town\npuddles hold the sky"),
    ("Name a colour.", "Blue"),
    ("Explain recursion in one sentence.", " Recursion is when a function solves a problem by calling itself on smaller inputs."),
]


def test_explicit_default_system_prompt_matches_implicit(tok):
    implicit = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True)
    explicit = tok.apply_chat_template([{"role": "system", "content": QWEN_DEFAULT}, {"role": "user", "content": "hi"}],
                                       tokenize=False, add_generation_prompt=True)
    assert implicit == explicit


def test_response_ends_with_end_of_turn(tok):
    assert response_ids(tok, "Hello")[-1] == tok.convert_tokens_to_ids("<|im_end|>")


def test_response_span_identical_across_contexts(tok):
    # Same response ids regardless of system prompt (we splice, never re-tokenise jointly).
    for user, resp in EXAMPLES:
        a = prompt_ids(tok, SYSTEM, user) + response_ids(tok, resp)
        b = prompt_ids(tok, QWEN_DEFAULT, user) + response_ids(tok, resp)
        assert a[-len(response_ids(tok, resp)):] == b[-len(response_ids(tok, resp)):]


def test_splice_matches_joint_tokenisation(tok):
    assert all(splice_matches_template(tok, SYSTEM, u, r) for u, r in EXAMPLES)


def test_response_logprob_matches_manual(model, tok):
    user, resp = EXAMPLES[1]
    p, r = prompt_ids(tok, SYSTEM, user), response_ids(tok, resp)
    ids = torch.tensor([p + r])
    with torch.no_grad():
        logits = model(ids).logits[0].float()
    manual = sum(torch.log_softmax(logits[t - 1], -1)[ids[0, t]].item() for t in range(len(p), len(p) + len(r)))
    assert response_logprob(model, tok, SYSTEM, user, resp) == pytest.approx(manual, abs=1e-3)


def test_batched_padded_equals_single(model, tok):
    batched = response_logprobs(model, tok, SYSTEM, EXAMPLES, max_tokens=10_000)
    single = [response_logprobs(model, tok, SYSTEM, [ex])[0] for ex in EXAMPLES]
    for (lb, nb), (ls, ns) in zip(batched, single):
        assert nb == ns
        assert lb == pytest.approx(ls, abs=1e-3)
    # A tiny budget forces one sequence per batch and must give the same result.
    tiny = response_logprobs(model, tok, SYSTEM, EXAMPLES, max_tokens=1)
    assert [n for _, n in tiny] == [n for _, n in batched]
    assert [l for l, _ in tiny] == pytest.approx([l for l, _ in batched], abs=1e-3)


def test_delta_is_difference_of_contexts(model, tok):
    user, resp = EXAMPLES[0]
    d = delta(model, tok, SYSTEM, QWEN_DEFAULT, user, resp)
    assert d == pytest.approx(response_logprob(model, tok, QWEN_DEFAULT, user, resp)
                              - response_logprob(model, tok, SYSTEM, user, resp), abs=1e-4)


def test_routing_weight_closed_form():
    assert routing_weight(8.0, 0.1) == pytest.approx(0.997, abs=5e-4)
    assert routing_weight(-0.5, 0.1) == pytest.approx(0.063, abs=5e-4)
    assert routing_weight(0.0, 0.5) == pytest.approx(0.5)
    assert routing_weight(2.0, 0.3, tau=0.5) == pytest.approx(1 / (1 + math.exp(-(1.0 + math.log(0.3 / 0.7)))))
    assert routing_weight(-800.0, 0.5) == pytest.approx(0.0) and routing_weight(800.0, 0.5) == pytest.approx(1.0)
