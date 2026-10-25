# Archived experimental evidence

Artifacts produced by the paper run (Google Colab, NVIDIA L4, 5 seeds,
`configs/paper.yaml`), copied unchanged from the run directory
`fidelity_shapley_v1/`. They are the source of every number and figure in the
manuscript and are checked automatically by `tests/test_evidence.py`.

| File | Content | Paper |
|---|---|---|
| `paper_numbers_dt_final.json` | Consolidated numbers: H1, H2, validation, detection, Shapley means, bandwidth summary and raw curves | all of §6 |
| `state.json` | Completion timestamps of each pipeline stage | — |
| `scores/shapley.json` | Per-seed Shapley values (SKAB exact; C-MAPSS antithetic MC) and late-trace s.d. | §6.1, §6.3, Fig. 2, Fig. 3a |
| `scores/merged.json` | Merged Temperature–Thermocouple super-player vs sum of members, per seed | §6.3, Fig. 3b |
| `scores/spotcheck.json` | Mask-model vs retrain-per-coalition RMSE (6 coalitions × 2 seeds) | §6.2, Fig. 3c |
| `scores/bandwidth.json` | Fidelity RMSE for every rule × budget × seed | §6.4, Table 4, Fig. 4 |
| `scores/eval_final.json` | H1 dual-label AUROCs, detection, k\*, paired tests | §6.1, §6.4, §6.7 |
| `scores/interactions.json` | Seed-averaged pairwise interactions: Ī, max, Λ_i | §6.5, Fig. 5 |
| `scores/inferability.json` | Reconstructability R² vs φ (SKAB) | §6.5 |
| `scores/defect_map_v2.json` | ΔRMSE for 8 channels × 4 defects × 3 intensities × 5 seeds | §6.6, Fig. 6 |
| `checkpoints/*.pt` | The 10 trained surrogates (2 datasets × 5 seeds), PyTorch state dicts | all |
| `data/ledger.json` | Provenance ledger (NaN fills, clipping counts, degeneracy flags, groups) | §4.2 |
| `data/checksums.json` | Fingerprints of the cached input data | — |
| `tex/` | LaTeX macros and the k\* table exported by the run | — |
| `paper_results_dt_final.docx` | Result tables exported by the run | — |
| `figures/paper/` | Publication figures (Figs. 2–6) + the exact plotted data (`data_figN.json`) | Figs. 2–6 |
| `figures/notebook_run/` | Quick-look figures produced inside the notebook run | — |
| `reproduction_report_cpu.md` | Independent recomputation of every stage from the checkpoints on CPU | — |
| `legacy/` | Intermediate 3-seed outputs superseded by the 5-seed run (kept for provenance) | — |

The interaction and inferability analyses were computed after the notebook run
from the same checkpoints; the package now implements both and reproduces the
archived files (`reproduction_report_cpu.md`).
