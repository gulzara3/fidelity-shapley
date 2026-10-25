# Reproducibility

## Provenance of the archived results

`results/` holds the artifacts of the paper run: Google Colab, one NVIDIA L4 GPU,
BF16 autocast, `configs/paper.yaml`, five model seeds. The run was executed
with the notebook in `notebooks/FidelityShapley_FINAL_paper_run.ipynb`, whose
logic the package reproduces line for line (same seeds, random-number streams,
hyper-parameters and stage order). Stage completion times are in
`results/state.json`.

## Independent recomputation

The release was checked by recomputing every stage on a CPU-only machine from
freshly downloaded public data and the archived surrogate checkpoints, including
retraining the 12 retrain-per-coalition reference models from scratch. The
comparison is in [`results/reproduction_report_cpu.md`](../results/reproduction_report_cpu.md)
and the recomputed artifacts in `results/reproduction_cpu/`. In short:

* Shapley values (exact and Monte Carlo), merged-group values, the defect map, the
  surrogate-validation statistics (ρ = 0.979), interactions, inferability and the
  H1 AUROCs agree with the archive within floating-point tolerance.
* Every k\* in Table 4 is identical.
* 2 of 450 bandwidth-curve points differ, both caused by near-tied singleton
  utilities of the first-order (VoI) proxy whose order swaps between GPU-BF16 and
  CPU arithmetic.
* Two C-MAPSS paired sign tests change by one win (mutual information
  20/25 → 19/25, p = 0.004 → 0.015; mRMR 23/25 → 22/25). Conclusions at the 5 %
  level are unchanged; the 1 % marker for mutual information is hardware-sensitive.

## Determinism

* All randomness is seeded from `global_seed = 20260808` (see
  `docs/methodology.md` for the exact derivation of each stream).
* `torch.use_deterministic_algorithms(True, warn_only=True)`, cuDNN deterministic
  mode and `CUBLAS_WORKSPACE_CONFIG=:4096:8` are set at start-up.
* Bit-exact results across GPU models, CUDA versions or CPU vs GPU are not
  expected: BF16 autocast and kernel selection change the last digits of the
  utilities. Retraining the surrogates from scratch on different hardware gives
  models that differ at the level of seed-to-seed variation; compare trends and
  confidence intervals rather than digits in that case. Reusing
  `results/checkpoints/` (level 3 in the README) removes this source of variation.

## Platform-dependence fix: constant-sensor normalisation

The notebook standardised channels with `(x − μ) / σ` and replaced σ = 0 by 1.
For the constant C-MAPSS sensors s5 (14.62) and s16 (0.03), floating-point
summation can return σ ≈ 10⁻¹⁵ instead of exactly 0, depending on the
pandas/NumPy build. The channel is then mapped to −1 instead of 0, and the
published surrogates — trained on inputs where these channels were 0 — produce
RUL errors up to ~4 cycles higher. The package treats any σ below the ledger's
degeneracy tolerance (10⁻⁶) as constant (`data/pipeline.py::safe_std`), which
restores the published inputs on every platform. With the original rule, the
CPU recomputation would have failed for C-MAPSS seeds 1 and 3.

## Tolerances used by `fidelity-shapley verify`

| Quantity | Tolerance | Rationale |
|---|---|---|
| SKAB Shapley values, merged value | 2·10⁻³ | < 3 % of the smallest informative value (0.06) |
| C-MAPSS Shapley values | 0.05 | < 7 % of the smallest informative value (0.71); MC s.d. ≈ 0.02 |
| Defect ΔRMSE | 5·10⁻³ | < 2 % of the largest effect (0.44) |
| SKAB / C-MAPSS curves | 2·10⁻³ / 0.1 cycles | BF16 rounding; ≥ 98 % of points must agree (tie flips) |
| Retrained spot-check RMSE | 0.05 | models retrained on different hardware |
| k\*, H1 AUROC | exact | headline claims |

## Run time

| Run | Hardware | Wall clock |
|---|---|---|
| Paper run, fresh (training + all stages) | NVIDIA L4, Google Colab | ≈ 2–2.5 h (C-MAPSS MC Shapley ≈ 15 min per seed) |
| Recompute from checkpoints (`reproduce_from_checkpoints.sh`) | CPU-only Linux VM | ≈ 11 min (download + all stages, incl. 12 retrainings) |
| Smoke test (`configs/smoke.yaml`) | laptop CPU | a few minutes |
