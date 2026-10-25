"""§5.1 — Shapley estimators for the coalition game (N, u).

* ``exact_shapley``  — enumerates all coalitions (SKAB, n = 8 → 256 utilities,
  cached so each coalition is evaluated once).
* ``mc_shapley``     — antithetic permutation sampling (C-MAPSS, n = 21): every
  sampled permutation is paired with its reverse; returns the estimate and the
  running trace used for the late-trace convergence diagnostic.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from itertools import combinations

import numpy as np

Utility = Callable[[np.ndarray], float]


def exact_shapley(util: Utility, n: int, players: Sequence[int] | None = None) -> np.ndarray:
    """Exact Shapley values of ``players`` (default: all n channels).

    Channels not in ``players`` are held absent; this is used for the merged
    game, where a group is represented by one super-player."""
    players = list(players) if players is not None else list(range(n))
    k = len(players)
    cache: dict[tuple, float] = {}

    def u(bits: list) -> float:
        key = tuple(bits)
        if key not in cache:
            cache[key] = util(np.array(bits, np.float32))
        return cache[key]

    phi = np.zeros(k)
    base = np.zeros(n, np.float32)
    for pi, p in enumerate(players):
        others = [q for q in players if q != p]
        for r in range(k):
            for S in combinations(others, r):
                w = math.factorial(r) * math.factorial(k - 1 - r) / math.factorial(k)
                m0 = base.copy()
                m0[list(S)] = 1
                m1 = m0.copy()
                m1[p] = 1
                phi[pi] += w * (u(m1.tolist()) - u(m0.tolist()))
    return phi


def mc_shapley(util: Utility, n: int, n_perm: int, rng: np.random.Generator,
               verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Antithetic Monte Carlo Shapley with ``n_perm`` permutation pairs."""
    phi = np.zeros(n)
    u_empty = util(np.zeros(n, np.float32))
    trace = []
    for t in range(n_perm):
        perm = rng.permutation(n)
        for order in (perm, perm[::-1]):
            m = np.zeros(n, np.float32)
            prev = u_empty
            for p in order:
                m[p] = 1
                cur = util(m)
                phi[p] += cur - prev
                prev = cur
        trace.append((phi / (2 * (t + 1))).copy())
        if verbose and (t + 1) % 25 == 0:
            print(f"    MC perm-pair {t + 1}/{n_perm}")
    return phi / (2 * n_perm), np.stack(trace)


def late_trace_sd(trace: np.ndarray, tail: int = 20) -> float:
    """Mean (over channels) s.d. of the running estimate over the last ``tail`` steps."""
    return float(np.std(trace[-tail:], 0).mean())
