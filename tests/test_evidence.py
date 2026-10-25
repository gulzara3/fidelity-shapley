"""The paper's headline numbers, re-derived from the archived result files.

These tests need no GPU, data download or training: they check that every
quantity reported in the manuscript follows from ``results/``.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import pearsonr, spearmanr

R = Path(__file__).resolve().parents[1] / "results"
P = json.loads((R / "paper_numbers_dt_final.json").read_text())
SC = lambda n: json.loads((R / "scores" / f"{n}.json").read_text())  # noqa: E731
SEEDS = [str(s) for s in P["seeds"]]


def test_h1_blind_recovery():
    h1 = P["h1"]
    assert np.mean(h1["auroc_literature_label"]) == 1.0
    assert np.mean(h1["auroc_ledger_label"]) == pytest.approx(0.964, abs=5e-4)
    ratio = h1["mean_abs_value_informative_lit"] / h1["mean_abs_value_uninformative_lit"]
    assert ratio == pytest.approx(16.4, abs=0.05)
    assert h1["s6_mean_value"] == pytest.approx(0.109, abs=5e-4)
    assert h1["weakest_informative_value"] == pytest.approx(0.710, abs=5e-4)


def test_surrogate_validation():
    sp = SC("spotcheck")
    m = [v["mask_rmse"] for v in sp.values()]
    r = [v["retrain_rmse"] for v in sp.values()]
    assert len(sp) == 12
    assert spearmanr(m, r).statistic == pytest.approx(0.979, abs=5e-4)
    assert pearsonr(m, r).statistic == pytest.approx(0.994, abs=5e-4)
    assert sum(ri > mi for mi, ri in zip(m, r, strict=True)) == 9


def test_merging_residual():
    h2 = P["h2"]
    assert h2["mean_gap"] == pytest.approx(0.0383, abs=5e-5)
    assert h2["mean_gap"] / h2["mean_group_value"] == pytest.approx(0.267, abs=0.005)
    merged = SC("merged")
    assert all(v["group_value"] < v["sum_members"] for s in merged for v in merged[s].values())


def test_mc_convergence():
    sd = list(P["mc_convergence_sd"].values())
    assert 0.017 <= min(sd) and max(sd) <= 0.0211
    phi = np.array([SC("shapley")["cmapss"][s] for s in SEEDS])
    assert max(sd) / np.abs(phi).max() < 0.006


def test_closed_loop_table4():
    bw = P["bandwidth_raw"]["cmapss"]
    mean8 = {k: np.mean([v["8"][s] for s in SEEDS]) for k, v in bw.items()}
    assert min(mean8, key=mean8.get) == "FidelityShapley (ours)"
    assert mean8["FidelityShapley (ours)"] == pytest.approx(14.97, abs=0.005)
    ks = P["bandwidth_summary"]
    assert ks["cmapss"]["kstar"]["FidelityShapley (ours)"] == 12
    assert ks["cmapss"]["kstar"]["Mutual information"] == 16
    assert ks["skab"]["kstar"]["Mutual information"] == 3
    t = ks["cmapss"]["tests_vs_ours"]
    assert t["Mutual information"]["sign_p"] < 0.01 and t["Mutual information"]["ours_wins"] == 20
    assert t["VoI-proxy"]["sign_p"] > 0.5                    # not a consistent advantage
    assert ks["skab"]["tests_vs_ours"]["Mutual information"]["ours_wins"] == 5


def test_interaction_diagnostic():
    I = SC("interactions")
    ratio = I["cmapss"]["mean_abs_interaction"] / I["skab"]["mean_abs_interaction"]
    assert ratio == pytest.approx(27.0, abs=0.1)
    for tag, key in (("skab", "shapley_skab_mean"), ("cmapss", "shapley_cmapss_mean")):
        phi = np.array(list(P[key].values()))
        rel = I[tag]["mean_abs_interaction"] / (phi.sum() / len(phi))
        assert rel == pytest.approx(0.80, abs=0.01)
    lam = np.array(I["cmapss"]["per_channel_total"])
    lowest7 = {f"s{i + 1}" for i in np.argsort(lam)[:7]}
    assert lowest7 == {"s1", "s5", "s6", "s10", "s16", "s18", "s19"}


def test_defects_and_inferability():
    d = SC("defect_map_v2")
    t = np.mean([d[s]["Temperature|drift|4.0"] for s in SEEDS])
    assert t == pytest.approx(0.44, abs=0.005)
    assert SC("inferability")["spearman_r2_vs_phi"] == pytest.approx(0.667, abs=5e-4)


def test_detection_is_a_limitation():
    det = P["detection_limitation"]
    assert np.mean(det["raw"]["auroc"]) == pytest.approx(0.509, abs=5e-4)
    assert np.mean(det["calibrated"]["auroc"]) == pytest.approx(0.552, abs=5e-4)
