from types import SimpleNamespace

from pi.capability import gsm8k_gold, last_number, mmlu_choice, _num_eq


def lp(token, logprob):
    return SimpleNamespace(decoded_token=token, logprob=logprob)


def test_mmlu_choice_picks_most_probable_letter():
    assert mmlu_choice({1: lp("B", -0.1), 2: lp(" A", -1.0), 3: lp("The", -0.05)}) == "B"
    assert mmlu_choice({1: lp(" C", -2.0), 2: lp("C", -0.5)}) == "C"
    assert mmlu_choice({1: lp("The", -0.1)}) is None


def test_gsm8k_parsing():
    assert gsm8k_gold("blah\n#### 1,234") == "1234"
    assert last_number("So 3 + 4 = 7. The answer is 1,250.") == "1250"
    assert last_number("no digits here") is None
    assert _num_eq("18", "18.0") and not _num_eq("18", "19")
