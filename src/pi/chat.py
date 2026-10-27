"""Chat-template handling: prompt ids for (system, user), response ids spliced after them.

The response is tokenised once, independently of the prompt, so the response span is the same
token sequence under every context (plan.md 6.3). `splice_matches_template` checks how often that
differs from tokenising the full conversation jointly.
"""

from functools import lru_cache

SENTINEL = "\x00RESPONSE\x00"


def prompt_ids(tok, system: str, user: str) -> list[int]:
    """Token ids of system + user turns plus the assistant header. `system` is always explicit."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    out = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True)
    return list(out["input_ids"])


@lru_cache(maxsize=None)
def _end_of_turn(tok) -> str:
    # What the template writes after an assistant message, stripped of trailing whitespace,
    # e.g. "<|im_end|>" for Qwen2.5 and "<|eot_id|>" for Llama 3.
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": SENTINEL}], tokenize=False)
    assert full.startswith(head), "chat template renders the prompt differently with an assistant turn"
    tail = full[len(head):]
    assert tail.startswith(SENTINEL), f"template adds text before the assistant content: {tail[:40]!r}"
    eot = tail[len(SENTINEL):].rstrip()
    assert eot, "template adds no end-of-turn marker"
    return eot


def response_ids(tok, response: str) -> list[int]:
    """Response tokens followed by the end-of-turn token, which is part of the scored span."""
    return tok(response + _end_of_turn(tok), add_special_tokens=False)["input_ids"]


def splice_matches_template(tok, system: str, user: str, response: str) -> bool:
    """True if prompt_ids + response_ids equals tokenising the whole conversation at once."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user},
            {"role": "assistant", "content": response}]
    joint = tok.apply_chat_template(msgs, tokenize=True, return_dict=True)["input_ids"]
    spliced = prompt_ids(tok, system, user) + response_ids(tok, response)
    return list(joint[: len(spliced)]) == spliced
