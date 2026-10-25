import numpy as np
import pytest

from fidelity_shapley.evaluation.bandwidth import kstar, paired_tests, qbin, rankings
from fidelity_shapley.evaluation.defects import inject
from fidelity_shapley.evaluation.sensors import h1_dual_label


def test_defect_semantics():
    rng = np.random.default_rng(0)
    X = np.tile(np.arange(10, dtype=np.float32)[None, :, None], (4, 1, 3))
    d = inject(X, 1, "drift", 4.0, rng)
    np.testing.assert_allclose(d[:, :, 1] - X[:, :, 1], np.tile(np.linspace(0, 4, 10), (4, 1)), atol=1e-6)
    assert np.array_equal(d[:, :, 0], X[:, :, 0])                 # other channels untouched
    s = inject(X, 2, "staleness", 3, rng)
    np.testing.assert_allclose(s[:, 3:, 2], X[:, :-3, 2])
    st = inject(X, 0, "stuck_at", 1.0, rng)
    assert np.all(st[:, :, 0] == 0)                                # held at first value
    with pytest.raises(ValueError):
        inject(X, 0, "bogus", 1, rng)


def test_kstar_and_paired_tests():
    seeds = [0, 1]
    per_k = {"1": {"0": 2.0, "1": 2.0}, "2": {"0": 1.04, "1": 1.04}, "3": {"0": 1.0, "1": 1.0}}
    assert kstar(per_k, ["1", "2", "3"], seeds, full_rmse=1.0) == 2
    ours = {k: {"0": 1.0, "1": 1.0} for k in "12345"}
    rival = {k: {"0": 2.0, "1": 2.0} for k in "12345"}
    t = paired_tests(ours, rival, list("12345"), seeds)
    assert t["n"] == 10 and t["ours_wins"] == 10 and t["sign_p"] < 0.01
    assert paired_tests(ours, ours, list("12345"), seeds) == {"n": 0}   # all ties removed


def test_rankings_contain_every_channel():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5, 4)).astype(np.float32)
    tgt = X[:, -1, 2] * 3 + rng.normal(size=200) * 0.1
    rk = rankings(np.array([0.1, 0.2, 0.9, 0.0]), X, tgt, 4, lambda m: float(m[2]), rng)
    for order in rk.values():
        assert sorted(np.asarray(order).tolist()) == [0, 1, 2, 3]
    assert rk["FidelityShapley (ours)"][0] == 2 and rk["Mutual information"][0] == 2
    assert len(np.unique(qbin(tgt))) > 1


def test_h1_dual_label_perfect_separation():
    phi = np.ones((2, 21))
    for s in ("s1", "s5", "s6", "s10", "s16", "s18", "s19"):
        phi[:, int(s[1:]) - 1] = 0.01
    h1 = h1_dual_label(phi, ["s1", "s5", "s10", "s16", "s18", "s19"])
    assert h1["auroc_literature_label"] == [1.0, 1.0]
    assert h1["auroc_ledger_label"][0] < 1.0      # s6 counted informative by the ledger
