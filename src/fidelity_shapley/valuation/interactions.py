"""§4.6 — Pairwise Harsanyi dividends and the applicability diagnostic.

I_ij = u({i,j}) − u({i}) − u({j}) + u(∅)  (empty-context Möbius interaction)
Λ_i  = Σ_{j≠i} |I_ij|                      (per-channel interaction strength)
Ī    = mean_{i<j} |I_ij|                   (dataset-level magnitude)
Ī_rel = Ī / (Σ_i φ_i / n)                  (scale-free; Eq. 20)

Requires n + n(n−1)/2 + 1 utilities (C-MAPSS: 232).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import spearmanr


def pairwise_interactions(util: Callable[[np.ndarray], float], n: int) -> dict:
    empty = util(np.zeros(n, np.float32))
    eye = np.eye(n, dtype=np.float32)
    single = np.array([util(eye[i]) for i in range(n)])
    I = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            m = eye[i] + eye[j]
            I[i, j] = I[j, i] = util(m) - single[i] - single[j] + empty
    return {"u_empty": float(empty), "u_single": single.tolist(), "I": I.tolist()}


def summarise(I: np.ndarray) -> dict:
    I = np.asarray(I)
    iu = np.triu_indices(I.shape[0], 1)
    return {"mean_abs_interaction": float(np.abs(I[iu]).mean()),
            "max_abs_interaction": float(np.abs(I[iu]).max()),
            "per_channel_total": np.abs(I).sum(1).tolist()}


def relative_interaction(mean_abs: float, phi_mean: np.ndarray) -> float:
    n = len(phi_mean)
    return float(mean_abs / (np.sum(phi_mean) / n))


def ranking_agreement(phi_mean: np.ndarray, delta: np.ndarray) -> float:
    """Spearman agreement between the φ and first-order Δ rankings."""
    return float(spearmanr(phi_mean, delta).statistic)
