"""Regenerate the publication figures (Figs. 2–6) from an archived or fresh results directory."""
import argparse
from pathlib import Path

from fidelity_shapley.reporting.paper_figures import render_all

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/figures/paper")
    a = ap.parse_args()
    render_all(Path(a.results), Path(a.out))
