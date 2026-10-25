# Reproduction report — independent recomputation on CPU

**What was run.** `bash scripts/reproduce_from_checkpoints.sh` on a CPU-only Linux
machine (PyTorch 2.14, pandas 3.0, NumPy 2.5, BF16 autocast on CPU) on 25 Oct 2026.
SKAB and C-MAPSS FD001 were **re-downloaded from the public sources**; the 10 archived
surrogates in `results/checkpoints/` were reused; every other stage — exact and Monte
Carlo Shapley, merged game, defect map, the 12 retrain-per-coalition models (trained
from scratch), H1, detection, bandwidth curves, paired tests, interactions and
inferability — was recomputed. The paper run used an NVIDIA L4 GPU on Google Colab.
The recomputed artifacts are archived in `results/reproduction_cpu/`.

**Outcome.** All artifacts agree within numerical tolerance; every k\* and the H1
AUROC are identical; ρ(mask, retrain) = 0.979 is reproduced exactly with freshly
retrained reference models. Two of 450 bandwidth-curve points differ because two
channels have near-identical singleton utilities (gaps of 2·10⁻⁴ on SKAB and 5·10⁻³
RUL cycles on C-MAPSS) whose order swaps under different floating-point arithmetic;
this changes one VoI-proxy selection per dataset. Win counts in two C-MAPSS sign tests
move by one pair (MI: 20/25 → 19/25, p = 0.004 → 0.015; mRMR: 23/25 → 22/25), so these
comparisons remain significant at 5 % but the MI comparison crosses the 1 % mark.


Candidate: `results/reproduction_cpu` (fresh CPU run, see below)  
Reference: `results`

| Artifact | values | within tolerance | max abs deviation | tolerance | status |
|---|---|---|---|---|---|
| Exact Shapley, SKAB (all seeds) | 40 | 40/40 | 0.00063 | 0.002 | PASS |
| MC Shapley, C-MAPSS (all seeds) | 105 | 105/105 | 0.0137 | 0.05 | PASS |
| MC late-trace s.d. | 5 | 5/5 | 6.35e-05 | 0.002 | PASS |
| Merged-group value (H2) | 12 | 12/12 | 0.00117 | 0.002 | PASS |
| Defect map ΔRMSE (H4) | 480 | 480/480 | 0.000609 | 0.005 | PASS |
| Bandwidth curves, SKAB (H3) | 240 | 239/240 | 0.0421 | 0.002 | PASS (1 outside tol.; near-tied channel scores swap order under different floating-point arithmetic) |
| Bandwidth curves, C-MAPSS (H3) | 210 | 209/210 | 0.802 | 0.1 | PASS (1 outside tol.; near-tied channel scores swap order under different floating-point arithmetic) |
| Spot-check mask utility | 12 | 12/12 | 0.000174 | 0.002 | PASS |
| Spot-check retrained utility | 12 | 12/12 | 0.000863 | 0.05 | PASS |
| Mean |I_ij| (seed-averaged) | 2 | 2/2 | 0.00397 | 0.02 | PASS |
| Reconstructability R² | 8 | 8/8 | 0 | 0.001 | PASS |

| Headline quantity | reference | candidate | status |
|---|---|---|---|
| k* per rule (skab) | {'FidelityShapley (ours)': 5, 'VoI-proxy': 5, 'Mutual information': 3, 'Variance': 6, 'mRMR (greedy)': 6, 'Random': 7} | {'FidelityShapley (ours)': 5, 'VoI-proxy': 5, 'Mutual information': 3, 'Variance': 6, 'mRMR (greedy)': 6, 'Random': 7} | PASS |
| k* per rule (cmapss) | {'FidelityShapley (ours)': 12, 'VoI-proxy': 16, 'Mutual information': 16, 'Variance': 12, 'mRMR (greedy)': 21, 'Random': 16} | {'FidelityShapley (ours)': 12, 'VoI-proxy': 16, 'Mutual information': 16, 'Variance': 12, 'mRMR (greedy)': 21, 'Random': 16} | PASS |
| H1 AUROC (literature label) | 1.000 | 1.000 | PASS |

Paired sign tests over the low-bandwidth regime (informational: a test
statistic counts wins, so a single near-tie can move p across a threshold).

| Comparison | reference wins/n, p | candidate wins/n, p |
|---|---|---|
| skab: ours vs VoI-proxy | 5/5, p=0.0625 | 5/5, p=0.0625 |
| skab: ours vs Mutual information | 5/20, p=0.0414 | 5/20, p=0.0414 |
| skab: ours vs Variance | 20/20, p=1.91e-06 | 20/20, p=1.91e-06 |
| skab: ours vs mRMR (greedy) | 20/20, p=1.91e-06 | 20/20, p=1.91e-06 |
| skab: ours vs Random | 14/19, p=0.0636 | 14/19, p=0.0636 |
| cmapss: ours vs VoI-proxy | 13/22, p=0.523 | 13/23, p=0.678 |
| cmapss: ours vs Mutual information | 20/25, p=0.00408 | 19/25, p=0.0146 |
| cmapss: ours vs Variance | 20/25, p=0.00408 | 20/25, p=0.00408 |
| cmapss: ours vs mRMR (greedy) | 23/25, p=1.94e-05 | 22/25, p=0.000157 |
| cmapss: ours vs Random | 23/25, p=1.94e-05 | 23/25, p=1.94e-05 |

**Overall: PASS**
