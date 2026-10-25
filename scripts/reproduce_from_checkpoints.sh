#!/usr/bin/env bash
# Recompute every stage of the paper from the archived surrogate checkpoints.
# Data are downloaded from the public sources; the 10 trained surrogates in
# results/checkpoints are reused, so no GPU training is needed. The retrain-
# per-coalition validation (12 small retrainings) is recomputed.
set -euo pipefail
EXP="${1:-outputs/repro}"
mkdir -p "$EXP/ckpt"
cp results/checkpoints/*.pt "$EXP/ckpt/"
fidelity-shapley run --config configs/paper.yaml --exp-dir "$EXP" "${@:2}"
fidelity-shapley verify --candidate "$EXP" --reference results --report "$EXP/reproduction_report.md"
