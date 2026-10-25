"""§2 — Download and cache the public SKAB and C-MAPSS FD001 benchmarks.

SKAB (Skoltech Anomaly Benchmark; Kaggle DOI 10.34740/KAGGLE/DSV/1693952) is
fetched from the official GitHub mirror; C-MAPSS FD001 (NASA PCoE) from a
widely used public mirror. Files are cached as Parquet under ``<exp_dir>/data``.
"""
from __future__ import annotations

import io
import urllib.request

import numpy as np
import pandas as pd

from ..config import CMAPSS_SENSORS, Config
from ..utils.stage import StageManager


def _fetch(url: str, timeout: int = 60) -> str:
    return urllib.request.urlopen(url, timeout=timeout).read().decode()


def paths(cfg: Config) -> dict:
    d = cfg.sub("data")
    return {"skab": d / "skab.parquet", "cmapss": d / "cmapss_fd001.parquet",
            "rul": d / "rul_fd001.npy"}


def download(cfg: Config, sm: StageManager) -> None:
    p = paths(cfg)
    if sm.done("download", [p["skab"], p["cmapss"], p["rul"]]):
        return
    frames = []
    for f in cfg.skab_files:
        df = pd.read_csv(io.StringIO(_fetch(f"{cfg.skab_base_url}/{f}")), sep=";")
        df.columns = [c.replace(" ", "") for c in df.columns]
        df["run"] = f.rsplit(".", 1)[0]
        if "anomaly" not in df.columns:           # anomaly-free reference run
            df["anomaly"], df["changepoint"] = 0.0, 0.0
        frames.append(df)
        print(f"  SKAB {f}: {len(df)} rows")
    pd.concat(frames, ignore_index=True).to_parquet(p["skab"])

    cols = ["unit", "cycle", "set1", "set2", "set3"] + CMAPSS_SENSORS
    cm = {part: pd.read_csv(io.StringIO(_fetch(f"{cfg.cmapss_base_url}/{part}_FD001.txt")),
                            sep=r"\s+", header=None, names=cols)
          for part in ("train", "test")}
    rul = pd.read_csv(io.StringIO(_fetch(f"{cfg.cmapss_base_url}/RUL_FD001.txt")),
                      sep=r"\s+", header=None)[0].to_numpy()
    cm["train"]["split"], cm["test"]["split"] = "train", "test"
    pd.concat([cm["train"], cm["test"]], ignore_index=True).to_parquet(p["cmapss"])
    np.save(p["rul"], rul)
    print(f"  C-MAPSS: {len(cm['train'])} train rows, {len(cm['test'])} test rows, "
          f"{len(rul)} RUL labels")
    sm.mark("download")


def load(cfg: Config):
    p = paths(cfg)
    return (pd.read_parquet(p["skab"]), pd.read_parquet(p["cmapss"]), np.load(p["rul"]))
