import pytest

from pi.judge import aggregate_0_100, judge_prompts


def test_aggregate_weighted_mean_over_numbers():
    assert aggregate_0_100({"80": 0.5, "90": 0.25, "100": 0.25}) == pytest.approx(87.5)


def test_aggregate_ignores_non_numbers_and_out_of_range():
    assert aggregate_0_100({"20": 0.3, "CODE": 0.4, "101": 0.2, "-5": 0.1}) == pytest.approx(20.0)


def test_aggregate_none_when_numbers_have_little_mass():
    assert aggregate_0_100({"REFUSAL": 0.8, "10": 0.2}) is None
    assert aggregate_0_100({}) is None


def test_judge_prompts_have_placeholders():
    prompts = judge_prompts()
    assert set(prompts) == {"aligned", "coherent"}
    for p in prompts.values():
        assert "{question}" in p and "{answer}" in p
        p.format(question="q", answer="a")
