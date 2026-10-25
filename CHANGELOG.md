# Changelog

## 1.0.0 — 2026-10-25
Initial public release accompanying the *Digital Twin* submission.

- Refactored the Colab notebook (`notebooks/FidelityShapley_FINAL_paper_run.ipynb`)
  into the `fidelity_shapley` package with a resumable CLI, YAML configuration,
  tests and CI. Algorithms, seeds and hyper-parameters are unchanged.
- Added the two analyses reported in the paper that were run outside the notebook:
  pairwise interactions / applicability diagnostic (`valuation/interactions.py`)
  and the inferability check (`evaluation/inferability.py`). Both reproduce the
  archived results (see `docs/reproducibility.md`).
- **Reproducibility fix:** z-normalisation now treats channels whose training
  standard deviation is below `constant_std_tol` as constant. The notebook used
  `std().replace(0, 1)`, which fails when floating-point summation returns
  σ ≈ 1e-15 instead of 0 (C-MAPSS s5, s16 on some platforms) and silently
  changes the twin's inputs. The fix reproduces the published run exactly.
- CPU execution supported (`--device cpu`); the paper run used an NVIDIA L4 GPU.
- Archived experimental evidence in `results/` (scores, checkpoints, ledger,
  figures, paper numbers) with a reproduction report.
