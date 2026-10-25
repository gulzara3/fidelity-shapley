"""§6.2 — H3: closed-loop selection under a bandwidth budget, k*, paired tests."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import mutual_info_score

RULES = ["FidelityShapley (ours)", "VoI-proxy", "Mutual information",
         "Variance", "mRMR (greedy)", "Random"]


def qbin(v: np.ndarray) -> np.ndarray:
    """Decile-style binning used for the plug-in mutual-information estimate."""
    edges = np.unique(np.quantile(v, np.linspace(0.1, 0.9, 8)))
    return np.digitize(v, edges) if len(edges) >= 2 else np.zeros(len(v), int)


def rankings(phi_mean, X, target, n, net_util, seed_rng) -> dict[str, np.ndarray]:
    """Channel orderings for every rule.

    * Variance — variance of the standardised window values.
    * MI       — plug-in MI between binned last-step channel value and binned target.
    * mRMR     — greedy: MI minus mean |Pearson r| with already-selected channels.
    * VoI      — first-order singleton utility u({i}) under the same surrogate.
    """
    var = X.reshape(-1, n).var(0)
    tgt_b = qbin(target)
    mi = np.array([mutual_info_score(qbin(X[:, -1, i]), tgt_b) for i in range(n)])
    voi = np.array([net_util(np.eye(n, dtype=np.float32)[i]) for i in range(n)])
    with np.errstate(divide="ignore", invalid="ignore"):   # constant channels → NaN → 0
        corr = np.nan_to_num(np.abs(np.corrcoef(X[:, -1, :].T)))

    def mrmr():
        sel, rest = [int(np.argmax(mi))], list(range(n))
        rest.remove(sel[0])
        while rest:
            score = [mi[j] - corr[j, sel].mean() for j in rest]
            sel.append(rest.pop(int(np.argmax(score))))
        return sel

    return {"FidelityShapley (ours)": np.argsort(-phi_mean),
            "VoI-proxy": np.argsort(-voi),
            "Mutual information": np.argsort(-mi),
            "Variance": np.argsort(-var),
            "mRMR (greedy)": np.array(mrmr()),
            "Random": seed_rng.permutation(n)}


def kstar(per_k: dict, ks: list[str], seeds: list[int], full_rmse: float, tol: float = 0.05):
    """Smallest budget whose seed-mean RMSE is within (1+tol)·full-observation RMSE."""
    means = {k: np.mean([per_k[k][str(s)] for s in seeds]) for k in ks}
    return next((int(k) for k in ks if means[k] <= (1 + tol) * full_rmse), None)


def paired_tests(ours: dict, rival: dict, lowband: list[str], seeds: list[int]) -> dict:
    """Two-sided sign and Wilcoxon tests over (budget, seed) pairs; ties removed."""
    pairs = np.array([ours[k][str(s)] - rival[k][str(s)] for k in lowband for s in seeds])
    pairs = pairs[np.abs(pairs) > 1e-9]
    if len(pairs) < 5:
        return {"n": int(len(pairs))}
    wins = int((pairs < 0).sum())
    return {"n": int(len(pairs)), "ours_wins": wins,
            "sign_p": float(binomtest(wins, len(pairs)).pvalue),
            "wilcoxon_p": float(wilcoxon(pairs).pvalue),
            "mean_delta": float(pairs.mean())}
