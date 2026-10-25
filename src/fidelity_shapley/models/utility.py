"""Coalition utility u(S) = −RMSE of the twin against physical ground truth."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .surrogate import MaskedGRU, batched_pred


@dataclass
class UtilityFactory:
    """Builds u(S) closures for a given surrogate (mask_vec ∈ {0,1}^n)."""
    device: torch.device
    amp: object
    eval_batch: int
    skab_val: tuple[np.ndarray, np.ndarray]
    cm_xtest: np.ndarray
    rul_true: np.ndarray
    rul_cap: float

    def predict(self, net: MaskedGRU, X: np.ndarray, mask_vec) -> np.ndarray:
        return batched_pred(net, X, mask_vec, device=self.device, amp=self.amp,
                            bs=self.eval_batch)

    def u_skab(self, net: MaskedGRU, mask_vec) -> float:
        """Next-step telemetry fidelity on the held-out reference-run split."""
        pred = self.predict(net, self.skab_val[0], mask_vec)
        return -float(np.sqrt(np.mean((pred - self.skab_val[1]) ** 2)))

    def u_cmapss(self, net: MaskedGRU, mask_vec) -> float:
        """RUL fidelity (cycles) on the 100 held-out test engines."""
        pred = self.predict(net, self.cm_xtest, mask_vec).squeeze(-1) * self.rul_cap
        return -float(np.sqrt(np.mean((np.clip(pred, 0, None)
                                       - np.clip(self.rul_true, 0, self.rul_cap)) ** 2)))
