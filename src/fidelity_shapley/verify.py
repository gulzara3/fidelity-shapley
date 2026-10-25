"""Compare a fresh run against the archived reference results.

Usage: ``fidelity-shapley verify --candidate outputs/... --reference results``.
Writes a Markdown report with the maximum absolute deviation per artifact.
Tolerances absorb GPU-vs-CPU and BF16 rounding; they are far below every effect
size reported in the paper.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils.io import read_json

TOL = {"shapley_skab": 2e-3, "shapley_cmapss": 0.05, "trace_sd": 2e-3, "merged": 2e-3,
       "defects": 5e-3, "bandwidth_skab": 2e-3, "bandwidth_cmapss": 0.1,
       "spotcheck_mask": 2e-3, "spotcheck_retrain": 0.05, "interactions_rel": 0.02,
       "inferability": 1e-3}


def _flat(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flat(v, f"{prefix}{k}/"))
    elif isinstance(d, list) and d and not isinstance(d[0], (dict, list)):
        for i, v in enumerate(d):
            out[f"{prefix}{i}"] = v
    elif isinstance(d, (int, float)):
        out[prefix.rstrip("/")] = d
    return out


def _diffs(a, b):
    fa, fb = _flat(a), _flat(b)
    keys = [k for k in fa if k in fb and isinstance(fa[k], (int, float))
            and isinstance(fb[k], (int, float)) and fa[k] is not None and fb[k] is not None]
    return {k: abs(float(fa[k]) - float(fb[k])) for k in keys}


def verify(candidate: Path, reference: Path) -> tuple[bool, str]:
    c, r = Path(candidate), Path(reference)
    rows, ok_all = [], True

    def check(name, a, b, tol, rel=False, min_frac=1.0, note=""):
        """PASS if at least ``min_frac`` of the values agree within ``tol``."""
        nonlocal ok_all
        if a is None or b is None:
            rows.append((name, "—", "—", "—", f"{tol:g}", "SKIPPED (artifact missing)"))
            return
        if rel:
            fa, fb = _flat(a), _flat(b)
            diffs = {k: abs(fa[k] - fb[k]) / max(abs(fb[k]), 1e-12) for k in fa if k in fb}
        else:
            diffs = _diffs(a, b)
        n = len(diffs)
        within = sum(v <= tol for v in diffs.values())
        d = max(diffs.values()) if diffs else None
        ok = n > 0 and within / n >= min_frac
        ok_all &= ok
        status = "PASS" if ok else "FAIL"
        if ok and within < n:
            status += f" ({n - within} outside tol.{'; ' + note if note else ''})"
        rows.append((name, f"{n}", f"{within}/{n}", f"{d:.3g}" if d is not None else "—",
                     f"{tol:g}", status))

    sc, sr = read_json(c / "scores/shapley.json"), read_json(r / "scores/shapley.json")
    check("Exact Shapley, SKAB (all seeds)", sc and sc["skab"], sr and sr["skab"], TOL["shapley_skab"])
    check("MC Shapley, C-MAPSS (all seeds)", sc and sc["cmapss"], sr and sr["cmapss"], TOL["shapley_cmapss"])
    check("MC late-trace s.d.", sc and sc["trace_sd"], sr and sr["trace_sd"], TOL["trace_sd"])
    check("Merged-group value (H2)", read_json(c / "scores/merged.json"),
          read_json(r / "scores/merged.json"), TOL["merged"])
    check("Defect map ΔRMSE (H4)", read_json(c / "scores/defect_map_v2.json"),
          read_json(r / "scores/defect_map_v2.json"), TOL["defects"])
    bc, br = read_json(c / "scores/bandwidth.json"), read_json(r / "scores/bandwidth.json")
    tie = "near-tied channel scores swap order under different floating-point arithmetic"
    check("Bandwidth curves, SKAB (H3)", bc and bc["skab"], br and br["skab"],
          TOL["bandwidth_skab"], min_frac=0.98, note=tie)
    check("Bandwidth curves, C-MAPSS (H3)", bc and bc["cmapss"], br and br["cmapss"],
          TOL["bandwidth_cmapss"], min_frac=0.98, note=tie)
    spc, spr = read_json(c / "scores/spotcheck.json"), read_json(r / "scores/spotcheck.json")
    if spc and spr:
        check("Spot-check mask utility", {k: v["mask_rmse"] for k, v in spc.items()},
              {k: v["mask_rmse"] for k, v in spr.items()}, TOL["spotcheck_mask"])
        check("Spot-check retrained utility", {k: v["retrain_rmse"] for k, v in spc.items()},
              {k: v["retrain_rmse"] for k, v in spr.items()}, TOL["spotcheck_retrain"])
    ic, ir = read_json(c / "scores/interactions.json"), read_json(r / "scores/interactions.json")
    if ic and ir:
        check("Mean |I_ij| (seed-averaged)", {k: ic[k]["mean_abs_interaction"] for k in ir},
              {k: ir[k]["mean_abs_interaction"] for k in ir}, TOL["interactions_rel"], rel=True)
    fc, fr = read_json(c / "scores/inferability.json"), read_json(r / "scores/inferability.json")
    check("Reconstructability R²", fc and fc["recon_r2"], fr and fr["recon_r2"], TOL["inferability"])

    pc, pr = read_json(c / "paper_numbers_dt_final.json"), read_json(r / "paper_numbers_dt_final.json")
    claims, info = [], []
    if pc and pr:
        for tag in ("skab", "cmapss"):
            kc, kr = pc["bandwidth_summary"][tag]["kstar"], pr["bandwidth_summary"][tag]["kstar"]
            same = kc == kr
            ok_all &= same
            claims.append((f"k* per rule ({tag})", str(kr), str(kc), "PASS" if same else "FAIL"))
        for tag in ("skab", "cmapss"):
            tc = pc["bandwidth_summary"][tag]["tests_vs_ours"]
            tr = pr["bandwidth_summary"][tag]["tests_vs_ours"]
            for rv in tr:
                if "sign_p" in tr[rv] and "sign_p" in tc.get(rv, {}):
                    info.append((f"{tag}: ours vs {rv}",
                                 f"{tr[rv]['ours_wins']}/{tr[rv]['n']}, p={tr[rv]['sign_p']:.3g}",
                                 f"{tc[rv]['ours_wins']}/{tc[rv]['n']}, p={tc[rv]['sign_p']:.3g}"))
        a1 = float(np.mean(pc["h1"]["auroc_literature_label"]))
        a0 = float(np.mean(pr["h1"]["auroc_literature_label"]))
        claims.append(("H1 AUROC (literature label)", f"{a0:.3f}", f"{a1:.3f}",
                       "PASS" if abs(a1 - a0) < 1e-6 else "FAIL"))
        ok_all &= abs(a1 - a0) < 1e-6

    lines = ["# Reproduction report", "",
             f"Candidate: `{c}`  ", f"Reference: `{r}`", "",
             "| Artifact | values | within tolerance | max abs deviation | tolerance | status |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {a} | {b} | {w} | {d} | {t} | {s} |" for a, b, w, d, t, s in rows]
    if claims:
        lines += ["", "| Headline quantity | reference | candidate | status |", "|---|---|---|---|"]
        lines += [f"| {a} | {b} | {d} | {s} |" for a, b, d, s in claims]
    if info:
        lines += ["", "Paired sign tests over the low-bandwidth regime (informational: a test",
                  "statistic counts wins, so a single near-tie can move p across a threshold).", "",
                  "| Comparison | reference wins/n, p | candidate wins/n, p |", "|---|---|---|"]
        lines += [f"| {a} | {b} | {c_} |" for a, b, c_ in info]
    lines += ["", f"**Overall: {'PASS' if ok_all else 'FAIL'}**"]
    return ok_all, "\n".join(lines)
