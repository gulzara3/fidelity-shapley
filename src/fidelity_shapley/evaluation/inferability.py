"""§6.5 — Inferability check: is a channel's value lower when it is reconstructable?

Reconstructability = in-sample R² of a linear regression predicting the channel
from all other channels (last step of the SKAB training windows)."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression


def reconstructability(X_last: np.ndarray) -> np.ndarray:
    n = X_last.shape[1]
    r2 = []
    for i in range(n):
        o = [j for j in range(n) if j != i]
        r2.append(LinearRegression().fit(X_last[:, o], X_last[:, i]).score(X_last[:, o], X_last[:, i]))
    return np.array(r2)


def inferability(X_last: np.ndarray, phi_mean: np.ndarray, channels: list[str]) -> dict:
    r2 = reconstructability(X_last)
    return {"recon_r2": r2.tolist(), "shapley": list(map(float, phi_mean)),
            "spearman_r2_vs_phi": float(spearmanr(r2, phi_mean).statistic),
            "channels": channels}
