"""Command-line interface: ``fidelity-shapley {run,verify,figures,info}``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fidelity-shapley",
                                 description="FidelityShapley experiment pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run (or resume) the experiment")
    r.add_argument("--config", default="configs/paper.yaml")
    r.add_argument("--exp-dir", default=None, help="override output directory")
    r.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    r.add_argument("--seeds", default=None, help="comma-separated, e.g. 0,1")
    r.add_argument("--stages", default=None, help="comma-separated subset of stages")

    v = sub.add_parser("verify", help="compare a run against archived results")
    v.add_argument("--candidate", required=True)
    v.add_argument("--reference", default="results")
    v.add_argument("--report", default=None, help="write Markdown report here")

    f = sub.add_parser("figures", help="render the publication figures")
    f.add_argument("--results", default="results")
    f.add_argument("--out", default="results/figures/paper")

    sub.add_parser("info", help="print the resolved default configuration")

    a = ap.parse_args(argv)
    if a.cmd == "run":
        from .experiment import ALL_STAGES, Experiment
        seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else None
        cfg = load_config(a.config, exp_dir=a.exp_dir, device=a.device, model_seeds=seeds)
        stages = a.stages.split(",") if a.stages else ALL_STAGES
        bad = set(stages) - set(ALL_STAGES)
        if bad:
            ap.error(f"unknown stages {sorted(bad)}; choose from {ALL_STAGES}")
        Experiment(cfg).run(stages)
        return 0
    if a.cmd == "verify":
        from .verify import verify
        ok, report = verify(Path(a.candidate), Path(a.reference))
        print(report)
        if a.report:
            Path(a.report).write_text(report + "\n")
        return 0 if ok else 1
    if a.cmd == "figures":
        from .reporting.paper_figures import render_all
        render_all(Path(a.results), Path(a.out))
        return 0
    if a.cmd == "info":
        print(json.dumps(load_config().to_dict(), indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
