"""§6.1 — H1: blind recovery of uninformative sensors (dual-label protocol)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from ..config import CMAPSS_SENSORS, LITERATURE_UNINFORMATIVE


def h1_dual_label(phi_cm: np.ndarray, const_sensors: list[str]) -> dict:
    """AUROC of |φ| for informative-vs-uninformative under (a) the ledger's
    zero-variance flags and (b) the literature 14-sensor convention (incl. s6)."""
    lab_ledger = np.array([0 if s in const_sensors else 1 for s in CMAPSS_SENSORS])
    lab_lit = np.array([0 if s in LITERATURE_UNINFORMATIVE else 1 for s in CMAPSS_SENSORS])
    return {
        "auroc_ledger_label": [float(roc_auc_score(lab_ledger, np.abs(r))) for r in phi_cm],
        "auroc_literature_label": [float(roc_auc_score(lab_lit, np.abs(r))) for r in phi_cm],
        "mean_abs_value_uninformative_lit": float(np.abs(phi_cm[:, lab_lit == 0]).mean()),
        "mean_abs_value_informative_lit": float(np.abs(phi_cm[:, lab_lit == 1]).mean()),
        "s6_mean_value": float(phi_cm[:, CMAPSS_SENSORS.index("s6")].mean()),
        "weakest_informative_value": float(np.abs(phi_cm[:, lab_lit == 1]).mean(0).min()),
    }
