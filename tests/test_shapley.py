"""Axiomatic and numerical properties of the Shapley estimators (no data needed)."""
import itertools
import math

import numpy as np
import pytest

from fidelity_shapley.valuation.interactions import pairwise_interactions, summarise
from fidelity_shapley.valuation.merged import merged_group_value
from fidelity_shapley.valuation.shapley import exact_shapley, late_trace_sd, mc_shapley


def random_game(n, seed=0):
    """A game defined by random Harsanyi dividends on every coalition."""
    rng = np.random.default_rng(seed)
    div = {T: rng.normal() for r in range(1, n + 1) for T in itertools.combinations(range(n), r)}

    def u(mask):
        S = {i for i in range(n) if mask[i] > 0.5}
        return sum(d for T, d in div.items() if set(T) <= S)

    return u, div


def shapley_from_dividends(div, n):
    phi = np.zeros(n)
    for T, d in div.items():
        for i in T:
            phi[i] += d / len(T)
    return phi


@pytest.mark.parametrize("n", [3, 5, 6])
def test_exact_matches_dividend_formula(n):
    u, div = random_game(n, seed=n)
    np.testing.assert_allclose(exact_shapley(u, n), shapley_from_dividends(div, n), atol=1e-10)


def test_efficiency_symmetry_null():
    n = 5
    w = np.array([3.0, 3.0, 1.0, 0.0, 2.0])          # channel 3 is a null player

    def u(m):
        return float(w @ m + 0.5 * m[0] * m[1])       # channels 0,1 interchangeable

    phi = exact_shapley(u, n)
    assert phi.sum() == pytest.approx(u(np.ones(n)) - u(np.zeros(n)))
    assert phi[0] == pytest.approx(phi[1])
    assert phi[3] == pytest.approx(0.0)


def test_additivity():
    n = 4
    u1, _ = random_game(n, 1)
    u2, _ = random_game(n, 2)
    phi = exact_shapley(lambda m: u1(m) + u2(m), n)
    np.testing.assert_allclose(phi, exact_shapley(u1, n) + exact_shapley(u2, n), atol=1e-10)


def test_antithetic_mc_converges_and_is_efficient():
    n = 6
    u, div = random_game(n, 7)
    phi_mc, trace = mc_shapley(u, n, 400, np.random.default_rng(0), verbose=False)
    exact = shapley_from_dividends(div, n)
    # every permutation telescopes, so efficiency holds exactly for MC too
    assert phi_mc.sum() == pytest.approx(exact.sum(), abs=1e-9)
    assert np.max(np.abs(phi_mc - exact)) < 0.15
    assert trace.shape == (400, n)
    assert late_trace_sd(trace) < 0.05


def test_merged_identity_proposition_2():
    """φ'_g − (φ_i + φ_j) = −Σ_{T'≠∅} d(T'∪G)·t/((t+1)(t+2))  (paper, Prop. 2)."""
    n = 5
    u, div = random_game(n, 11)
    G = [1, 3]
    phi = shapley_from_dividends(div, n)
    merged = merged_group_value(u, n, G)
    rhs = 0.0
    others = [k for k in range(n) if k not in G]
    for t in range(1, len(others) + 1):
        for Tp in itertools.combinations(others, t):
            T = tuple(sorted(set(Tp) | set(G)))
            rhs -= div[T] * t / ((t + 1) * (t + 2))
    assert merged - (phi[1] + phi[3]) == pytest.approx(rhs, abs=1e-10)


def test_pairwise_interactions_equal_pair_dividends():
    n = 4
    u, div = random_game(n, 3)
    I = np.array(pairwise_interactions(u, n)["I"])
    for i, j in itertools.combinations(range(n), 2):
        assert I[i, j] == pytest.approx(div[(i, j)])
    s = summarise(I)
    pairs = itertools.combinations(range(n), 2)
    assert s["mean_abs_interaction"] == pytest.approx(np.mean([abs(div[p]) for p in pairs]))
    assert len(s["per_channel_total"]) == n


def test_exact_shapley_evaluates_each_coalition_once():
    n, calls = 6, []

    def u(m):
        calls.append(tuple(m))
        return float(m.sum())

    exact_shapley(u, n)
    assert len(calls) == len(set(calls)) == 2 ** n
    assert math.isclose(sum(exact_shapley(lambda m: float(m.sum()), n)), n)
