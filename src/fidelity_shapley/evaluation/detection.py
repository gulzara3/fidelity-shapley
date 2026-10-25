"""§6.1 (limitation) — residual-based anomaly detection on labelled SKAB runs.

Reported as a negative result: a one-step forecaster tracks anomalous dynamics
rather than diverging from them (AUROC ≈ 0.51 raw, ≈ 0.55 per-run calibrated).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from ..data.pipeline import skab_windows


def residual_detection(prep, nets: dict, predict, cfg) -> dict:
    full = np.ones(len(prep.skab_mu), np.float32)
    det = {"raw": {"auroc": [], "ap": [], "f1": []},
           "calibrated": {"auroc": [], "ap": [], "f1": []}, "cal_frac": cfg.cal_frac}
    for s in cfg.model_seeds:
        res_raw, res_cal, y_all = [], [], []
        for r in prep.anomaly_runs:
            Xr, Tr, yr = skab_windows(prep.skab_clean[prep.skab_clean.run == r],
                                      prep.skab_mu, prep.skab_sd, cfg.window_skab)
            res = np.sqrt(np.mean((predict(nets[s], Xr, full) - Tr) ** 2, 1))
            nc = max(10, int(cfg.cal_frac * len(res)))   # robust per-run calibration
            med = np.median(res[:nc])
            mad = np.median(np.abs(res[:nc] - med)) + 1e-9
            res_raw.append(res)
            res_cal.append((res - med) / mad)
            y_all.append(yr)
        yy = np.concatenate(y_all)
        for tag, arr in (("raw", res_raw), ("calibrated", res_cal)):
            rr = np.concatenate(arr)
            det[tag]["auroc"].append(float(roc_auc_score(yy, rr)))
            det[tag]["ap"].append(float(average_precision_score(yy, rr)))
            thr = np.quantile(rr, 1 - yy.mean())
            det[tag]["f1"].append(float(f1_score(yy, rr > thr)))
    return det
