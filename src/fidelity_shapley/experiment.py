"""End-to-end experiment runner (mirrors §1–§7 of the paper notebook).

Every stage is idempotent and keyed per seed inside its JSON artifact, so a run
interrupted at any point resumes where it stopped, and adding seeds computes
only what is missing. Artifacts are written under ``cfg.exp_dir``::

    data/       skab.parquet, cmapss_fd001.parquet, rul_fd001.npy, ledger.json
    ckpt/       {skab,cmapss}_seed{s}.pt
    scores/     shapley, merged, defect_map_v2, spotcheck, bandwidth,
                interactions, inferability, eval_final (.json)
    paper_numbers_dt_final.json, state.json
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from .config import CMAPSS_SENSORS, LITERATURE_UNINFORMATIVE, SKAB_CHANNELS, Config
from .data import download as dl
from .data.pipeline import Prepared, prepare
from .evaluation import bandwidth as bwm
from .evaluation.defects import defect_row
from .evaluation.detection import residual_detection
from .evaluation.inferability import inferability
from .evaluation.sensors import h1_dual_label
from .evaluation.spotcheck import design_coalitions
from .models.surrogate import train_surrogate
from .models.utility import UtilityFactory
from .utils.io import read_json, write_json
from .utils.seeding import amp_dtype, resolve_device, seed_everything
from .utils.stage import StageManager
from .valuation import interactions as inter
from .valuation.merged import merged_group_value
from .valuation.shapley import exact_shapley, late_trace_sd, mc_shapley

ALL_STAGES = ["download", "prepare", "train", "shapley", "merged", "defects",
              "spotcheck", "evaluate", "bandwidth", "interactions", "inferability",
              "export"]


@dataclass
class Experiment:
    cfg: Config
    device: torch.device = None
    amp: object = None
    sm: StageManager = None
    prep: Prepared = None
    uf: UtilityFactory = None
    skab_nets: dict = field(default_factory=dict)
    cm_nets: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)

    def __post_init__(self):
        seed_everything(self.cfg.global_seed, self.cfg.deterministic)
        self.device = resolve_device(self.cfg.device)
        self.amp = amp_dtype(self.cfg.amp_dtype)
        self.sm = StageManager(self.cfg.path / "state.json")
        for sub in ("data", "ckpt", "scores", "figs", "tex"):
            self.cfg.sub(sub)
        print(f"device={self.device} amp={self.amp} exp_dir={self.cfg.path}")

    # ── helpers ──────────────────────────────────────────────────────────────
    def score_path(self, name: str):
        return self.cfg.sub("scores") / f"{name}.json"

    def predict(self, net, X, mask):
        return self.uf.predict(net, X, mask)

    def _seeds(self):
        return [int(s) for s in self.cfg.model_seeds]

    # ── §2–§3 data ───────────────────────────────────────────────────────────
    def stage_download(self):
        dl.download(self.cfg, self.sm)

    def stage_prepare(self):
        skab, cm, rul = dl.load(self.cfg)
        self.prep = prepare(self.cfg, skab, cm, rul)
        self.uf = UtilityFactory(self.device, self.amp, self.cfg.eval_batch,
                                 self.prep.skab_val, self.prep.cm_xtest, rul, self.cfg.rul_cap)

    # ── §4 surrogates ────────────────────────────────────────────────────────
    def stage_train(self):
        c, p = self.cfg, self.prep
        kw = dict(device=self.device, amp=self.amp, hidden=c.hidden, steps=c.steps,
                  bs=c.batch_size, keep_p=c.keep_p, lr=c.lr,
                  weight_decay=c.weight_decay, warmup=c.warmup_steps)
        ck = self.cfg.sub("ckpt")
        for s in self._seeds():
            self.skab_nets[s] = train_surrogate(p.skab_train[0], p.skab_train[1], 8, 8, s,
                                                ckpt=ck / f"skab_seed{s}.pt", **kw)
            self.cm_nets[s] = train_surrogate(p.cm_x, p.cm_y[:, None], 21, 1, s,
                                              ckpt=ck / f"cmapss_seed{s}.pt", **kw)
            print(f"seed {s}: SKAB full RMSE {-self.uf.u_skab(self.skab_nets[s], np.ones(8)):.4f}"
                  f" | C-MAPSS full RUL-RMSE {-self.uf.u_cmapss(self.cm_nets[s], np.ones(21)):.2f}")

    def us(self, s):
        return lambda m: self.uf.u_skab(self.skab_nets[s], m)

    def uc(self, s):
        return lambda m: self.uf.u_cmapss(self.cm_nets[s], m)

    # ── §5.1 Shapley ─────────────────────────────────────────────────────────
    def stage_shapley(self):
        path = self.score_path("shapley")
        shap = read_json(path, {"skab": {}, "cmapss": {}, "trace_sd": {}})
        for s in self._seeds():
            if str(s) not in shap["skab"]:
                t0 = time.time()
                shap["skab"][str(s)] = exact_shapley(self.us(s), 8).tolist()
                print(f"  seed {s}: SKAB exact Shapley ({time.time() - t0:.0f}s)")
                write_json(path, shap)
            if str(s) not in shap["cmapss"]:
                phi, trace = mc_shapley(self.uc(s), 21, self.cfg.mc_perm_pairs,
                                        np.random.default_rng(self.cfg.global_seed + s))
                shap["cmapss"][str(s)] = phi.tolist()
                shap["trace_sd"][str(s)] = late_trace_sd(trace, self.cfg.trace_tail)
                write_json(path, shap)
                print(f"  seed {s}: C-MAPSS MC-Shapley (late-trace sd {shap['trace_sd'][str(s)]:.4f})")
        self.results["shap"] = shap

    @property
    def phi_skab(self):
        return np.array([self.results["shap"]["skab"][str(s)] for s in self._seeds()])

    @property
    def phi_cm(self):
        return np.array([self.results["shap"]["cmapss"][str(s)] for s in self._seeds()])

    # ── §5.1b H2 merged group ────────────────────────────────────────────────
    def stage_merged(self):
        path = self.score_path("merged")
        merged = read_json(path, {})
        for s in self._seeds():
            if str(s) in merged:
                continue
            for _, members in self.prep.red_groups.items():
                rep = members[0]
                merged.setdefault(str(s), {})[SKAB_CHANNELS[rep]] = {
                    "group_value": merged_group_value(self.us(s), 8, members),
                    "sum_members": float(sum(self.results["shap"]["skab"][str(s)][m]
                                             for m in members)),
                    "members": [SKAB_CHANNELS[m] for m in members],
                    "group_rho": self.prep.group_rho}
            write_json(path, merged)
            print(f"  H2 merged-Shapley seed {s} done")
        self.results["merged"] = merged

    # ── §5.2 H4 defect map ───────────────────────────────────────────────────
    def stage_defects(self):
        path = self.score_path("defect_map_v2")
        dmap = read_json(path, {})
        X, T = self.prep.skab_val
        for s in self._seeds():
            if str(s) in dmap:
                continue
            rng = np.random.default_rng(self.cfg.global_seed + 100 + s)
            dmap[str(s)] = defect_row(self.skab_nets[s], X, T, SKAB_CHANNELS,
                                      self.cfg.defects, self.predict, rng)
            write_json(path, dmap)
            print(f"  defect map seed {s} done")
        self.results["dmap"] = dmap

    # ── §5.3 retrain-per-coalition validation ────────────────────────────────
    def stage_spotcheck(self):
        c, p = self.cfg, self.prep
        path = self.score_path("spotcheck")
        spot = read_json(path, {})
        coals = design_coalitions(self.phi_skab.mean(0), 8, c.spot_random_sizes,
                                  np.random.default_rng(c.global_seed + 7))
        for s in self._seeds()[:c.spot_n_seeds]:
            for ci, coal in enumerate(coals):
                key = f"{s}|{ci}"
                if key in spot:
                    continue
                mvec = np.zeros(8, np.float32)
                mvec[coal] = 1
                net_r = train_surrogate(p.skab_train[0], p.skab_train[1], 8, 8,
                                        c.spot_seed_offset + s, device=self.device,
                                        amp=self.amp, hidden=c.hidden, steps=c.spot_steps,
                                        bs=c.batch_size, keep_p=c.keep_p, lr=c.lr,
                                        weight_decay=c.weight_decay, warmup=c.warmup_steps,
                                        fixed_mask=mvec)
                pred = self.predict(net_r, p.skab_val[0], mvec)
                spot[key] = {"coalition": coal,
                             "retrain_rmse": float(np.sqrt(np.mean((pred - p.skab_val[1]) ** 2))),
                             "mask_rmse": -self.uf.u_skab(self.skab_nets[s], mvec)}
                write_json(path, spot)
                print(f"  spot-check seed {s} coalition {ci} (|S|={len(coal)}) done")
        m = np.array([v["mask_rmse"] for v in spot.values()])
        r = np.array([v["retrain_rmse"] for v in spot.values()])
        self.results["spot"] = spot
        self.results["spot_stats"] = {"spearman": float(spearmanr(m, r).statistic),
                                      "pearson": float(pearsonr(m, r).statistic),
                                      "n_coalitions": len(spot)}
        print("mask vs retrain:", self.results["spot_stats"])

    # ── §6.1 H1 + detection ──────────────────────────────────────────────────
    def stage_evaluate(self):
        self.results["h1"] = h1_dual_label(self.phi_cm, self.prep.const_sensors)
        self.results["detection"] = residual_detection(self.prep, self.skab_nets,
                                                       self.predict, self.cfg)
        h1 = self.results["h1"]
        print(f"H1 AUROC ledger {np.mean(h1['auroc_ledger_label']):.3f} | literature "
              f"{np.mean(h1['auroc_literature_label']):.3f} | s6 φ {h1['s6_mean_value']:.3f}")

    # ── §6.2 H3 bandwidth ────────────────────────────────────────────────────
    def stage_bandwidth(self):
        c, p = self.cfg, self.prep
        path = self.score_path("bandwidth")
        bw = read_json(path, {})
        ks = {"skab": [str(k) for k in c.ks_skab], "cmapss": [str(k) for k in c.ks_cmapss]}
        low = {"skab": [str(k) for k in c.lowband_skab], "cmapss": [str(k) for k in c.lowband_cmapss]}
        nr = c.cmapss_rank_windows
        for tag, n, util_of, phi_all, X, tgt in (
                ("skab", 8, self.us, self.phi_skab, p.skab_val[0], np.abs(p.skab_val[1]).mean(1)),
                ("cmapss", 21, self.uc, self.phi_cm, p.cm_x[:nr], p.cm_y[:nr] * c.rul_cap)):
            bw.setdefault(tag, {})
            for s in self._seeds():
                if str(s) in bw[tag].get("Random", {}).get(ks[tag][0], {}):
                    continue
                util = util_of(s)
                rks = bwm.rankings(phi_all.mean(0), X, tgt, n, util,
                                   np.random.default_rng(c.global_seed + s))
                for name, order in rks.items():
                    for k in ks[tag]:
                        m = np.zeros(n, np.float32)
                        m[np.asarray(order[:int(k)], int)] = 1
                        bw[tag].setdefault(name, {}).setdefault(k, {})[str(s)] = -util(m)
                write_json(path, bw)
                print(f"  bandwidth [{tag}] seed {s} done")
        summary = {}
        for tag, full_fn in (("skab", lambda s: -self.uf.u_skab(self.skab_nets[s], np.ones(8))),
                             ("cmapss", lambda s: -self.uf.u_cmapss(self.cm_nets[s], np.ones(21)))):
            full = float(np.mean([full_fn(s) for s in self._seeds()]))
            summ = {"full_rmse": full,
                    "kstar": {nm: bwm.kstar(pk, ks[tag], self._seeds(), full, c.band_tol)
                              for nm, pk in bw[tag].items()},
                    "tests_vs_ours": {nm: bwm.paired_tests(bw[tag]["FidelityShapley (ours)"], pk,
                                                           low[tag], self._seeds())
                                      for nm, pk in bw[tag].items() if nm != "FidelityShapley (ours)"}}
            summary[tag] = summ
            print(f"[{tag}] k*: {summ['kstar']}")
        self.results["bw"], self.results["bandwidth_summary"] = bw, summary
        write_json(self.score_path("eval_final"),
                   {"h1": self.results.get("h1"), "detection": self.results.get("detection"),
                    "bandwidth_summary": summary})

    # ── §4.6 interactions + §6.5 inferability ────────────────────────────────
    def stage_interactions(self):
        path = self.score_path("interactions")
        raw_path = self.score_path("interactions_per_seed")
        raw = read_json(raw_path, {"skab": {}, "cmapss": {}})
        for s in self._seeds():
            for tag, n, util in (("skab", 8, self.us(s)), ("cmapss", 21, self.uc(s))):
                if str(s) not in raw[tag]:
                    raw[tag][str(s)] = inter.pairwise_interactions(util, n)
                    write_json(raw_path, raw)
        out = {}
        for tag, phi in (("skab", self.phi_skab), ("cmapss", self.phi_cm)):
            I_mean = np.mean([raw[tag][str(s)]["I"] for s in self._seeds()], 0)
            out[tag] = inter.summarise(I_mean)       # seed-averaged dividends
            out[tag]["relative_interaction"] = inter.relative_interaction(
                out[tag]["mean_abs_interaction"], phi.mean(0))
            delta = np.mean([np.array(raw[tag][str(s)]["u_single"]) - raw[tag][str(s)]["u_empty"]
                             for s in self._seeds()], 0)
            out[tag]["phi_delta_rank_agreement"] = inter.ranking_agreement(phi.mean(0), delta)
        write_json(path, out)
        self.results["interactions"] = out
        print("interactions:", {k: round(v["mean_abs_interaction"], 4) for k, v in out.items()})

    def stage_inferability(self):
        res = inferability(self.prep.skab_train[0][:, -1, :], self.phi_skab.mean(0), SKAB_CHANNELS)
        write_json(self.score_path("inferability"), res)
        self.results["inferability"] = res
        print(f"inferability: Spearman(R², φ) = {res['spearman_r2_vs_phi']:+.3f}")

    # ── §7 export ────────────────────────────────────────────────────────────
    def stage_export(self):
        r = self.results
        gaps = [abs(v["group_value"] - v["sum_members"]) for s in r["merged"]
                for v in r["merged"][s].values()]
        gvals = [v["group_value"] for s in r["merged"] for v in r["merged"][s].values()]
        members = [SKAB_CHANNELS[j] for v in self.prep.red_groups.values() for j in v]
        paper = {
            "seeds": self._seeds(),
            "datasets": {"skab_runs": len(self.prep.anomaly_runs) + 1, "cmapss": "FD001",
                         "redundant_group": {"members": members, "rho": self.prep.group_rho},
                         "ledger_constants": self.prep.const_sensors,
                         "literature_uninformative": LITERATURE_UNINFORMATIVE},
            "h1": r["h1"],
            "h2": {"gaps": gaps, "mean_gap": float(np.mean(gaps)), "max_gap": float(np.max(gaps)),
                   "mean_group_value": float(np.mean(gvals))},
            "utility_validation": r["spot_stats"],
            "detection_limitation": r["detection"],
            "mc_convergence_sd": r["shap"]["trace_sd"],
            "shapley_skab_mean": dict(zip(SKAB_CHANNELS, self.phi_skab.mean(0).round(5).tolist(),
                                          strict=True)),
            "shapley_cmapss_mean": dict(zip(CMAPSS_SENSORS, self.phi_cm.mean(0).round(5).tolist(),
                                            strict=True)),
            "bandwidth_summary": r["bandwidth_summary"],
            "bandwidth_raw": r["bw"],
        }
        write_json(self.cfg.path / "paper_numbers_dt_final.json", paper)
        h1l = r["h1"]["auroc_literature_label"]
        bs = r["bandwidth_summary"]
        macros = {"NSeeds": str(len(self._seeds())), "HoneAUROCLit": f"{np.mean(h1l):.3f}",
                  "HoneAUROCLitSd": f"{np.std(h1l):.3f}", "SsixPhi": f"{r['h1']['s6_mean_value']:.3f}",
                  "HtwoMeanGap": f"{np.mean(gaps):.4f}",
                  "GroupRho": f"{self.prep.group_rho:.3f}" if self.prep.group_rho else "n/a",
                  "SpotRho": f"{r['spot_stats']['spearman']:.3f}",
                  "KstarOursSkab": str(bs["skab"]["kstar"]["FidelityShapley (ours)"]),
                  "KstarOursCmapss": str(bs["cmapss"]["kstar"]["FidelityShapley (ours)"]),
                  "DetCalAUROC": f"{np.mean(r['detection']['calibrated']['auroc']):.3f}"}
        tex = self.cfg.sub("tex")
        (tex / "macros.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}"
                                                   for k, v in macros.items()) + "\n")
        rows = ["\\begin{tabular}{@{}lcc@{}}", "\\toprule",
                "Method & $k^{*}$ SKAB & $k^{*}$ C-MAPSS \\\\", "\\midrule"]
        rows += [f"{nm} & {bs['skab']['kstar'][nm]} & {bs['cmapss']['kstar'][nm]} \\\\"
                 for nm in r["bw"]["skab"]]
        (tex / "tab_kstar.tex").write_text("\n".join(rows + ["\\bottomrule", "\\end{tabular}"]))
        print("exported:", self.cfg.path / "paper_numbers_dt_final.json", "and tex/")

    # ── orchestration ────────────────────────────────────────────────────────
    def run(self, stages=None):
        stages = stages or ALL_STAGES
        need = {"shapley": ["prepare", "train"], "merged": ["prepare", "train", "shapley"],
                "defects": ["prepare", "train"], "spotcheck": ["prepare", "train", "shapley"],
                "evaluate": ["prepare", "train", "shapley"],
                "bandwidth": ["prepare", "train", "shapley"],
                "interactions": ["prepare", "train", "shapley"],
                "inferability": ["prepare", "train", "shapley"],
                "export": ["prepare", "train", "shapley", "merged", "spotcheck",
                           "evaluate", "bandwidth"], "train": ["prepare"]}
        plan = []
        for s in stages:
            for d in need.get(s, []) + [s]:
                if d not in plan:
                    plan.append(d)
        plan = [s for s in ALL_STAGES if s in plan]
        if "prepare" in plan and "download" not in plan and not dl.paths(self.cfg)["skab"].exists():
            plan.insert(0, "download")
        for st in plan:
            print(f"\n══ stage: {st} ══")
            getattr(self, f"stage_{st}")()
        return self.results
