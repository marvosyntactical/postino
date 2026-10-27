import pytest

from pi.data import heldout_prompts, training_set
from pi.train import epoch_order


@pytest.mark.parametrize("dataset,n_heldout", [("medical", 200), ("sorh", 100)])
@pytest.mark.parametrize("regime,f", [("f0.5", 0.5), ("f0.1", 0.1)])
def test_mixture_fraction_and_heldout(dataset, n_heldout, regime, f):
    t = training_set(dataset, regime, n_heldout, mix_seed=0)
    assert sum(r["label"] for r in t) / len(t) == pytest.approx(f, abs=0.005)
    held = {p for p, _ in heldout_prompts(dataset, n_heldout)}
    assert len(held) == n_heldout and not held & {r["pair_id"] for r in t}
    assert [r["id"] for r in t] == [r["id"] for r in training_set(dataset, regime, n_heldout, mix_seed=0)]


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("dataset,n_heldout", [("medical", 200), ("sorh", 100)])
def test_hacks_spread_over_epoch_order(dataset, n_heldout, seed):
    # Regression: seeding the mixture and the shuffle alike put every hack at the end of the epoch.
    t = training_set(dataset, "f0.5", n_heldout, mix_seed=0)
    for epoch in range(3):
        order = epoch_order(len(t), seed, epoch)
        quarters = [order[i * len(t) // 4:(i + 1) * len(t) // 4] for i in range(4)]
        for q in quarters:
            assert sum(t[i]["label"] for i in q) / len(q) == pytest.approx(0.5, abs=0.1)
