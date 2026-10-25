"""Experiment configuration.

Every constant used in the paper run lives here, so that a single YAML file
(``configs/paper.yaml``) fully specifies an experiment. Defaults reproduce the
published run exactly.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SKAB_CHANNELS = [
    "Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure",
    "Temperature", "Thermocouple", "Voltage", "VolumeFlowRateRMS",
]
CMAPSS_SENSORS = [f"s{i}" for i in range(1, 22)]
# Sensors outside the conventional 14-sensor FD001 subset (external H1 label).
LITERATURE_UNINFORMATIVE = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
SKAB_REFERENCE_RUN = "anomaly-free/anomaly-free"


def _default_skab_files() -> list[str]:
    return (["anomaly-free/anomaly-free.csv"]
            + [f"valve1/{i}.csv" for i in range(16)]
            + [f"valve2/{i}.csv" for i in range(4)]
            + [f"other/{i}.csv" for i in range(1, 15)])


@dataclass
class Config:
    # ── run / reproducibility ───────────────────────────────────────────────
    exp_dir: str = "outputs/fidelity_shapley_v1"
    global_seed: int = 20260808
    model_seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    device: str = "auto"                 # "auto" | "cuda" | "cpu"
    amp_dtype: str = "bfloat16"          # autocast dtype ("none" disables)
    deterministic: bool = True

    # ── data sources (public) ───────────────────────────────────────────────
    skab_base_url: str = "https://raw.githubusercontent.com/waico/SKAB/master/data"
    skab_files: list[str] = field(default_factory=_default_skab_files)
    cmapss_base_url: str = ("https://raw.githubusercontent.com/jiaxiang-cheng/"
                            "PyTorch-Transformer-for-RUL-Prediction/main/CMAPSSData")

    # ── provenance pipeline ─────────────────────────────────────────────────
    clip_q_low: float = 0.001
    clip_q_high: float = 0.999
    redundancy_tau: float = 0.9
    constant_std_tol: float = 1e-6

    # ── fusion / windowing ──────────────────────────────────────────────────
    window_skab: int = 32
    window_cmapss: int = 30
    rul_cap: float = 125.0
    skab_train_frac: float = 0.8         # chronological split of the reference run

    # ── mask-trained surrogate ──────────────────────────────────────────────
    hidden: int = 64
    steps: int = 1200
    batch_size: int = 256
    keep_p: float = 0.7                  # channel retained w.p. 0.7 (mask rate p = 0.3)
    lr: float = 2e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 50
    eval_batch: int = 4096

    # ── Shapley estimation ──────────────────────────────────────────────────
    mc_perm_pairs: int = 100             # antithetic permutation pairs (C-MAPSS)
    trace_tail: int = 20                 # last-k trace entries for convergence s.d.

    # ── defect injection (H4) ───────────────────────────────────────────────
    defects: dict[str, list[float]] = field(default_factory=lambda: {
        "noise": [0.25, 0.5, 1.0], "drift": [1.0, 2.0, 4.0],
        "stuck_at": [0.1, 0.3, 0.5], "staleness": [2, 5, 10]})

    # ── retrain-per-coalition validation ────────────────────────────────────
    spot_n_seeds: int = 2
    spot_steps: int = 600
    spot_seed_offset: int = 1000
    spot_random_sizes: list[int] = field(default_factory=lambda: [2, 4, 6])

    # ── closed-loop bandwidth (H3) ──────────────────────────────────────────
    ks_skab: list[int] = field(default_factory=lambda: list(range(1, 9)))
    ks_cmapss: list[int] = field(default_factory=lambda: [2, 4, 6, 8, 12, 16, 21])
    lowband_skab: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    lowband_cmapss: list[int] = field(default_factory=lambda: [2, 4, 6, 8, 12])
    band_tol: float = 0.05
    cmapss_rank_windows: int = 2000      # training windows used by filter baselines

    # ── detection (reported as a limitation) ────────────────────────────────
    cal_frac: float = 0.15

    # ── derived helpers ─────────────────────────────────────────────────────
    @property
    def path(self) -> Path:
        return Path(self.exp_dir)

    def sub(self, name: str) -> Path:
        p = self.path / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load a YAML config (unknown keys raise) and apply keyword overrides."""
    data: dict[str, Any] = {}
    if path is not None:
        data = yaml.safe_load(Path(path).read_text()) or {}
    data.update({k: v for k, v in overrides.items() if v is not None})
    known = {f.name for f in dataclasses.fields(Config)}
    unknown = set(data) - known
    if unknown:
        raise KeyError(f"Unknown config keys: {sorted(unknown)}")
    return Config(**data)
