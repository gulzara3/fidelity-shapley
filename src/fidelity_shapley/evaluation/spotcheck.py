"""§5.3 — Validate the mask device against retrain-per-coalition utilities."""
from __future__ import annotations

import numpy as np


def design_coalitions(phi_skab_mean: np.ndarray, n: int, sizes, rng) -> list[list[int]]:
    """Random subsets of the given sizes, Shapley top-4, bottom-4, and the full set."""
    order = np.argsort(-phi_skab_mean)
    return ([sorted(rng.choice(n, k, replace=False).tolist()) for k in sizes]
            + [sorted(order[:4].tolist()), sorted(order[-4:].tolist()), list(range(n))])
