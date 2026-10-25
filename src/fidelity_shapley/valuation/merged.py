"""§5.1b — H2: value of a redundancy group merged into one super-player.

The super-player is present only when all members are observed (members are
switched together with the group representative). Proposition 2 in the paper
gives the exact identity for the resulting deviation from the members' sum.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .shapley import exact_shapley


def merged_group_value(util: Callable[[np.ndarray], float], n: int,
                       members: list[int]) -> float:
    rep = members[0]

    def util_m(mask: np.ndarray) -> float:
        mm = mask.copy()
        mm[members] = mask[rep]
        return util(mm)

    keep = [x for x in range(n) if x not in members or x == rep]
    phi_m = exact_shapley(util_m, n, players=keep)
    return float(phi_m[keep.index(rep)])
