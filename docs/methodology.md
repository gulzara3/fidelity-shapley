# Paper ↔ code map

How each element of the manuscript is implemented. Section and equation numbers
refer to the submitted article.

| Paper | What | Code |
|---|---|---|
| §3.1, Eq. (1) | Coalition utility u(S) = −RMSE against physical telemetry | `models/utility.py` (`u_skab`, `u_cmapss`) |
| §3.1, Eqs. (2)–(3) | Shapley value (subset and permutation forms) | `valuation/shapley.py` |
| §3.3, Eqs. (4)–(5) | Harsanyi dividends, pairwise interaction I_ij | `valuation/interactions.py` |
| §3.4, Eq. (7) | k\*: smallest budget within 5 % of full-observation fidelity | `evaluation/bandwidth.py::kstar` |
| §4.2, Eq. (8) | Robust clipping to reference-run quantiles (0.1 %/99.9 %) | `data/pipeline.py::clean_and_ledger` |
| §4.2, Eq. (9) | Provenance ledger | `data/pipeline.py` → `data/ledger.json` |
| §4.2, Eq. (10) | Redundancy groups by union–find on \|ρ_s\| > τ (adaptive fallback) | `data/pipeline.py::redundancy_groups` |
| §4.3, Eqs. (11)–(13) | Mask-trained GRU surrogate (mask concatenated to input, p = 0.3) | `models/surrogate.py` |
| §4.3, Eq. (14) | Utility by evaluation alone; retrain-per-coalition validation | `models/utility.py`, `evaluation/spotcheck.py`, `experiment.py::stage_spotcheck` |
| §4.4 | Exact Shapley (SKAB, 256 coalitions) | `valuation/shapley.py::exact_shapley` |
| §4.4, Eqs. (15)–(16) | Antithetic Monte Carlo Shapley (C-MAPSS, M = 100 pairs) + late-trace s.d. | `valuation/shapley.py::mc_shapley`, `late_trace_sd` |
| §4.5, Eqs. (17)–(18) | Merged super-player, Proposition 2 | `valuation/merged.py`; identity tested in `tests/test_shapley.py` |
| §4.6, Eqs. (19)–(20) | Λ_i, Ī, Ī_rel, φ–Δ ranking agreement | `valuation/interactions.py` |
| §4.7, Eqs. (21)–(22) | Defect injection and vulnerability V_i | `evaluation/defects.py` |
| Algorithm 1 | End-to-end procedure | `experiment.py::Experiment.run` |
| §5.2 | Baselines: MI, variance, mRMR, VoI proxy, random | `evaluation/bandwidth.py::rankings` |
| §5.3 | Sign and Wilcoxon tests over the low-bandwidth regime | `evaluation/bandwidth.py::paired_tests` |
| §6.1 | Dual-label AUROC (ledger vs literature) | `evaluation/sensors.py` |
| §6.5 | Inferability check (R² vs φ) | `evaluation/inferability.py` |
| §6.7 | Residual anomaly detection (limitation) | `evaluation/detection.py` |
| Figs. 2–6 | Publication figures | `reporting/paper_figures.py` |
| Fig. 1 | Architecture (editable) | `docs/figures/Fig1_architecture.drawio` |

## Implementation details worth stating precisely

These are the exact choices made in the paper run; where the manuscript text is
more general, the code below is authoritative.

* **SKAB fidelity** is next-step prediction RMSE (8-dimensional target) on the
  last 20 % of windows of the anomaly-free reference run (chronological split;
  7,497 training / 1,875 evaluation windows of length 32). The 34 labelled anomaly
  runs are used only for the detection analysis.
* **C-MAPSS fidelity** is RUL RMSE on the last window (length 30, zero-padded if
  shorter) of each of the 100 test engines; targets are capped at 125 cycles.
* **Surrogate:** one GRU layer, 64 hidden units, linear head; AdamW
  (lr 2·10⁻³, weight decay 10⁻⁴), batch 256, 1,200 steps, 50-step linear warm-up
  then cosine decay, BF16 autocast. Channels are kept with probability 0.7;
  all-empty masks are re-seeded with channel 0.
* **Retrain-per-coalition reference:** a fresh surrogate trained on one fixed
  coalition for 600 steps (seed 1000 + s), for 6 coalitions × 2 seeds. The six
  coalitions are three *random* subsets of sizes 2, 4 and 6 (generator seed
  `global_seed + 7`), the Shapley top-4 and bottom-4 sets and the full set.
* **Baselines:** *variance* is computed on the standardised evaluation windows;
  *mutual information* is a plug-in estimate between the decile-binned last-step
  channel value and the binned target (SKAB target: mean |next-step vector|;
  C-MAPSS: RUL); *mRMR* greedily maximises MI minus the mean absolute Pearson
  correlation with selected channels; *VoI proxy* ranks by the singleton utility
  u({i}). C-MAPSS filter baselines use the first 2,000 training windows.
* **Defects** act on the standardised input windows of the evaluation split:
  drift is a linear ramp from 0 to δ across each 32-step window; stuck-at holds
  the channel at its first in-window value for a fraction κ of windows; staleness
  lags the channel by s steps within each window.
* **Low-bandwidth regime** for paired tests: k ∈ {1,…,4} (SKAB) and
  k ∈ {2, 4, 6, 8, 12} (C-MAPSS); exact ties are removed before testing.
* **Interaction summaries** (Ī, Λ_i) are computed from the seed-averaged
  interaction matrix; **reconstructability** is the in-sample linear-regression
  R² on the last step of the SKAB training windows.
* **Seeds:** `global_seed = 20260808`; surrogate seed s ∈ {0,…,4}; MC permutations
  and the random baseline use `global_seed + s`; defect noise uses
  `global_seed + 100 + s`.
