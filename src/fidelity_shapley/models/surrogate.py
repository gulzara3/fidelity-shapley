"""§4 — Mask-trained twin surrogate (single-layer GRU, mask concatenated to input).

During training each channel is retained independently with probability
``keep_p`` (masking rate p = 1 − keep_p = 0.3). Masked channels are zeroed —
the training mean of a z-normalised channel — and the binary mask is
concatenated to the input so the network can tell "absent" from "zero".
One surrogate per seed then yields the fidelity of *any* channel coalition by
inference alone (Proposition 1 in the paper).
"""
from __future__ import annotations

import contextlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


class MaskedGRU(nn.Module):
    def __init__(self, n_ch: int, out_dim: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(2 * n_ch, hidden, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) standardised window; mask: (B, C) binary channel mask."""
        m = mask.unsqueeze(1).expand_as(x)
        h, _ = self.gru(torch.cat([x * m, m], -1))
        return self.head(h[:, -1])


def _autocast(device: torch.device, dtype):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device.type, dtype=dtype)


def train_surrogate(X: np.ndarray, Y: np.ndarray, n_ch: int, out_dim: int, seed: int,
                    *, device: torch.device, amp=None, hidden: int = 64,
                    steps: int = 1200, bs: int = 256, keep_p: float = 0.7,
                    lr: float = 2e-3, weight_decay: float = 1e-4, warmup: int = 50,
                    fixed_mask: np.ndarray | None = None,
                    ckpt: Path | None = None) -> MaskedGRU:
    """Train (or resume from ``ckpt``) a mask-conditioned surrogate.

    With ``fixed_mask`` the network is trained on exactly one coalition, which
    implements the retrain-per-coalition reference used to validate the mask
    device (§5.3)."""
    if ckpt is not None and Path(ckpt).exists():
        net = MaskedGRU(n_ch, out_dim, hidden).to(device)
        net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        print(f"[resume] {Path(ckpt).name} loaded")
        return net.eval()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    net = MaskedGRU(n_ch, out_dim, hidden).to(device)
    opt = AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sch = LambdaLR(opt, lambda s_: min(s_ / warmup, 1.0)
                   * 0.5 * (1 + math.cos(math.pi * s_ / steps)))
    g = torch.Generator(device="cpu").manual_seed(seed)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    net.train()
    for _ in range(steps):
        idx = torch.randint(0, len(Xt), (bs,), generator=g)
        xb, yb = Xt[idx].to(device), Yt[idx].to(device)
        if fixed_mask is None:
            mask = (torch.rand(bs, n_ch, generator=g) < keep_p).float()
            mask[mask.sum(1) == 0, 0] = 1.0          # never an all-empty mask
        else:
            mask = torch.tensor(fixed_mask, dtype=torch.float32).expand(bs, -1)
        mask = mask.to(device)
        with _autocast(device, amp):
            loss = nn.functional.mse_loss(net(xb, mask).squeeze(-1),
                                          yb.squeeze(-1) if out_dim == 1 else yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
    if ckpt is not None:
        Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), ckpt)
    return net.eval()


@torch.no_grad()
def batched_pred(net: MaskedGRU, X: np.ndarray, mask_vec: np.ndarray, *,
                 device: torch.device, amp=None, bs: int = 4096) -> np.ndarray:
    outs = []
    m = torch.tensor(mask_vec, dtype=torch.float32, device=device)
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i + bs]).to(device)
        with _autocast(device, amp):
            outs.append(net(xb, m.expand(len(xb), -1)).float().cpu())
    return torch.cat(outs).numpy()
