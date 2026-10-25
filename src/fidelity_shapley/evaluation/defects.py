"""§5.2 — H4: controlled defect injection → channel-resolved fidelity map.

Defects act on the standardised input windows of the held-out SKAB split
(shape B × T × C), one channel at a time:

* noise      — additive Gaussian noise, σ ∈ {0.25, 0.5, 1.0}
* drift      — linear ramp from 0 to δ across each input window, δ ∈ {1, 2, 4}
* stuck_at   — a fraction κ ∈ {0.1, 0.3, 0.5} of windows hold the channel at
               its first value in the window
* staleness  — the channel lags by s ∈ {2, 5, 10} steps within each window
"""
from __future__ import annotations

import numpy as np


def inject(X: np.ndarray, ch: int, kind: str, lvl: float, rng: np.random.Generator) -> np.ndarray:
    X = X.copy()
    if kind == "noise":
        X[:, :, ch] += rng.normal(0, lvl, X.shape[:2])
    elif kind == "drift":
        X[:, :, ch] += np.linspace(0, lvl, X.shape[1])[None, :]
    elif kind == "stuck_at":
        stuck = rng.random(len(X)) < lvl
        X[stuck, :, ch] = X[stuck, :1, ch]
    elif kind == "staleness":
        lag = int(lvl)
        X[:, lag:, ch] = X[:, :-lag, ch]
    else:
        raise ValueError(f"unknown defect kind: {kind}")
    return X.astype(np.float32)


def defect_row(net, X, T, channels, defects: dict, predict, rng) -> dict:
    """ΔRMSE (defect − clean) for every channel × defect × intensity."""
    full = np.ones(len(channels), np.float32)
    base = float(np.sqrt(np.mean((predict(net, X, full) - T) ** 2)))
    row = {}
    for ch, name in enumerate(channels):
        for kind, levels in defects.items():
            for lvl in levels:
                pred = predict(net, inject(X, ch, kind, lvl, rng), full)
                row[f"{name}|{kind}|{lvl}"] = float(np.sqrt(np.mean((pred - T) ** 2)) - base)
    return row


def vulnerability(dmap: dict, channels: list[str]) -> np.ndarray:
    """Mean ΔRMSE per channel over all defect settings and seeds (Eq. 22)."""
    seeds = list(dmap)
    keys = [k for k in dmap[seeds[0]]]
    return np.array([np.mean([dmap[s][k] for s in seeds for k in keys
                              if k.split("|")[0] == c]) for c in channels])
