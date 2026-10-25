"""§3 — Provenance pipeline: cleaning → redundancy grouping → fusion/windowing.

Every cell the pipeline alters is counted in a per-channel *quality ledger*
(``data/ledger.json``). The ledger's degeneracy flags are **never** shown to the
valuation; they serve only as an independent, mechanical label for H1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..config import CMAPSS_SENSORS, SKAB_CHANNELS, SKAB_REFERENCE_RUN, Config
from ..utils.io import write_json


# ── union–find over channels ────────────────────────────────────────────────
def redundancy_groups(rho_abs: np.ndarray, tau: float) -> dict[int, list[int]]:
    """Connected components of the graph {(i, j): |rho_ij| > tau} (union–find)."""
    n = rho_abs.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if rho_abs[i, j] > tau:
                parent[max(find(i), find(j))] = min(find(i), find(j))
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return {k: v for k, v in groups.items() if len(v) > 1}


@dataclass
class Prepared:
    """Everything downstream stages need, produced once from the raw data."""
    skab_clean: pd.DataFrame
    cmapss: pd.DataFrame
    rul_true: np.ndarray
    ledger: dict
    red_groups: dict[int, list[int]]
    group_rho: float | None
    group_w: dict[int, list[float]]
    const_sensors: list[str]
    # fused tensors
    skab_train: tuple[np.ndarray, np.ndarray] = field(repr=False, default=None)
    skab_val: tuple[np.ndarray, np.ndarray] = field(repr=False, default=None)
    cm_x: np.ndarray = field(repr=False, default=None)
    cm_y: np.ndarray = field(repr=False, default=None)
    cm_xtest: np.ndarray = field(repr=False, default=None)
    # normalisation statistics (training data only)
    skab_mu: pd.Series = field(repr=False, default=None)
    skab_sd: pd.Series = field(repr=False, default=None)
    cm_mu: pd.Series = field(repr=False, default=None)
    cm_sd: pd.Series = field(repr=False, default=None)

    @property
    def anomaly_runs(self) -> list[str]:
        return [r for r in self.skab_clean.run.unique() if r != SKAB_REFERENCE_RUN]


def clean_and_ledger(cfg: Config, skab: pd.DataFrame, cmapss: pd.DataFrame):
    """SKAB: forward/back-fill NaNs, clip to reference-run quantiles; C-MAPSS:
    flag zero-variance channels. Returns cleaned SKAB, ledger, groups, flags."""
    free = skab[skab.run == SKAB_REFERENCE_RUN]
    ledger: dict = {"skab": {}, "cmapss": {}}
    qlo = free[SKAB_CHANNELS].quantile(cfg.clip_q_low)
    qhi = free[SKAB_CHANNELS].quantile(cfg.clip_q_high)
    skab_clean = skab.copy()
    for c in SKAB_CHANNELS:
        n_nan = int(skab_clean[c].isna().sum())
        skab_clean[c] = skab_clean[c].ffill().bfill()
        n_clip = int(((skab_clean[c] < qlo[c]) | (skab_clean[c] > qhi[c])).sum())
        skab_clean[c] = skab_clean[c].clip(qlo[c], qhi[c])
        ledger["skab"][c] = {"nan_filled": n_nan, "clipped": n_clip}

    rho = np.abs(spearmanr(
        skab_clean.loc[skab_clean.run == SKAB_REFERENCE_RUN, SKAB_CHANNELS]).statistic)
    red = redundancy_groups(rho, cfg.redundancy_tau)
    group_rho = None
    if not red:
        # Adaptive fallback: adopt the single most-correlated real pair and
        # report its measured correlation, so H2 always has a non-trivial group.
        off = rho.copy()
        np.fill_diagonal(off, 0)
        i, j = np.unravel_index(np.argmax(off), off.shape)
        red = {int(min(i, j)): [int(min(i, j)), int(max(i, j))]}
        group_rho = float(off[i, j])
        print(f"[adaptive] most-redundant real pair: {SKAB_CHANNELS[i]}–"
              f"{SKAB_CHANNELS[j]} (rho={group_rho:.3f})")
    group_w = {k: [float(rho[v[0], j]) if j != v[0] else 1.0 for j in v]
               for k, v in red.items()}

    tr = cmapss[cmapss.split == "train"]
    const_mask = tr[CMAPSS_SENSORS].std() < cfg.constant_std_tol
    ledger["cmapss"] = {s: {"degenerate_constant": bool(const_mask[s])}
                        for s in CMAPSS_SENSORS}
    const = [s for s in CMAPSS_SENSORS if const_mask[s]]
    return skab_clean, ledger, red, group_rho, group_w, const


def safe_std(df: pd.DataFrame, tol: float) -> pd.Series:
    """Training standard deviation with constant channels mapped to 1.

    The original notebook used ``std().replace(0, 1.0)``. For constant C-MAPSS
    sensors (e.g. s5 = 14.62, s16 = 0.03) floating-point summation can return a
    standard deviation of ~1e-15 instead of exactly 0 on some platforms, which
    turns (x − μ)/σ into ±1 instead of 0 and silently changes the twin's inputs
    (up to ~4 RUL cycles of error with the published checkpoints). Thresholding
    at ``tol`` (the ledger's degeneracy tolerance) reproduces the published run
    bit-for-bit in the inputs on every platform.
    """
    sd = df.std()
    return sd.where(sd >= tol, 1.0)


def skab_windows(df: pd.DataFrame, mu: pd.Series, sd: pd.Series, w: int):
    """Sliding windows → next-step telemetry target (and anomaly label if present)."""
    x = ((df[SKAB_CHANNELS] - mu) / sd).to_numpy(np.float32)
    y = df["anomaly"].to_numpy(np.float32) if "anomaly" in df else None
    xs, ys, tgt = [], [], []
    for s in range(0, len(x) - w - 1):
        xs.append(x[s:s + w])
        tgt.append(x[s + w])
        if y is not None:
            ys.append(y[s + w])
    return np.stack(xs), np.stack(tgt), (np.array(ys, np.float32) if ys else None)


def cmapss_windows(df: pd.DataFrame, mu: pd.Series, sd: pd.Series, w: int,
                   rul_cap: float, is_train: bool):
    """Training: all windows with capped-RUL targets (scaled by the cap).
    Test: the last window of each engine (zero-padded if shorter than ``w``)."""
    xs, ys = [], []
    for _, g in df.groupby("unit"):
        x = ((g[CMAPSS_SENSORS] - mu) / sd).to_numpy(np.float32)
        n = len(x)
        if is_train:
            for s in range(0, n - w):
                xs.append(x[s:s + w])
                ys.append(min(rul_cap, n - (s + w)))
        else:
            pad = np.zeros((max(0, w - n), len(CMAPSS_SENSORS)), np.float32)
            xs.append(np.concatenate([pad, x[-w:]], 0))
    return np.stack(xs), (np.array(ys, np.float32) / rul_cap if is_train else None)


def prepare(cfg: Config, skab: pd.DataFrame, cmapss: pd.DataFrame,
            rul_true: np.ndarray, write_ledger: bool = True) -> Prepared:
    skab_clean, ledger, red, grho, gw, const = clean_and_ledger(cfg, skab, cmapss)
    if write_ledger:
        write_json(cfg.sub("data") / "ledger.json", {
            "ledger": ledger,
            "redundant_groups": {SKAB_CHANNELS[k]: [SKAB_CHANNELS[j] for j in v]
                                 for k, v in red.items()}})
    print("redundant groups:", {SKAB_CHANNELS[k]: [SKAB_CHANNELS[j] for j in v]
                                for k, v in red.items()})
    print("C-MAPSS constant sensors (ledger; hidden from valuation):", const)

    free = skab[skab.run == SKAB_REFERENCE_RUN]
    mu, sd = free[SKAB_CHANNELS].mean(), safe_std(free[SKAB_CHANNELS], cfg.constant_std_tol)
    tr = cmapss[cmapss.split == "train"]
    cm_mu = tr[CMAPSS_SENSORS].mean()
    cm_sd = safe_std(tr[CMAPSS_SENSORS], cfg.constant_std_tol)

    free_clean = skab_clean[skab_clean.run == SKAB_REFERENCE_RUN]
    xf, tf, _ = skab_windows(free_clean, mu, sd, cfg.window_skab)
    cut = int(cfg.skab_train_frac * len(xf))
    cm_x, cm_y = cmapss_windows(tr, cm_mu, cm_sd, cfg.window_cmapss, cfg.rul_cap, True)
    cm_xtest, _ = cmapss_windows(cmapss[cmapss.split == "test"], cm_mu, cm_sd,
                                 cfg.window_cmapss, cfg.rul_cap, False)
    prep = Prepared(skab_clean, cmapss, rul_true, ledger, red, grho, gw, const,
                    (xf[:cut], tf[:cut]), (xf[cut:], tf[cut:]), cm_x, cm_y, cm_xtest,
                    mu, sd, cm_mu, cm_sd)
    print(f"SKAB train/val windows: {cut}/{len(xf) - cut} | C-MAPSS train windows: "
          f"{len(cm_x)} | test engines: {len(cm_xtest)}")
    return prep
