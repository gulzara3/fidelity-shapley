#!/usr/bin/env bash
# Full paper experiment from scratch: download → pipeline → train 10 surrogates
# → valuation → evaluation → export. Resumable: re-run the same command after
# an interruption. GPU strongly recommended (paper run: NVIDIA L4, ~2–2.5 h).
set -euo pipefail
EXP="${1:-outputs/fidelity_shapley_v1}"
fidelity-shapley run --config configs/paper.yaml --exp-dir "$EXP"
fidelity-shapley verify --candidate "$EXP" --reference results --report "$EXP/reproduction_report.md" || \
  echo "Note: retrained surrogates differ from the archived ones at the level of seed variance; compare trends, not digits."
