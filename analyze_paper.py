"""
analyze_paper.py
----------------
One-shot analysis of a protocol-v2 campaign (runs/<tag>/eval/k<K>/<slug>/).

    python analyze_paper.py --run_root runs/paper-v2 [--ol 50 --cl 10]
                            [--metric success_strict] [--allow_mismatch]

Produces runs/<tag>/analysis/:
  provenance_report.txt     every run's code hash / checkpoint / matched set / settings,
                            and a hard failure if runs that are compared disagree
  table_conditions.{csv,md,tex}
                            per horizon: TSR + Wilson CI, raw sub-goal rates, pushed
                            successes, delta vs nominal (paired bootstrap CI), exact
                            McNemar p, Holm-adjusted p (family = all conditions at that k)
  table_ol_vs_cl.{csv,md,tex}
                            per condition: TSR at OL and CL, paired delta + McNemar, and
                            the difference-in-differences of the robustness drop
  table_camera_shapley.csv  per-camera Shapley contribution to TSR (if the 2^3 factorial exists)
  figures/*.png|pdf         see FIGURES below
  analysis_summary.json

FIGURES (each answers a "why" question)
  fig1_tsr_by_condition     where does performance drop, per horizon (Wilson CIs)
  fig2_delta_forest         effect size vs nominal with paired CIs, Holm-significant marked
  fig3_failure_taxonomy     WHICH stage fails (never reached / no grasp / dropped / not released)
  fig4_stage_timing         WHEN it fails: cumulative P(stage reached by step t)
  fig5_spatial_success      WHERE in the workspace it fails (initial cube position)
  fig6_dose_response        how fast performance degrades with perturbation magnitude
  fig7_horizon_sweep        TSR vs execution horizon k (if >= 3 horizons were run)
  fig8_min_dist_ecdf        threshold-free: how close the gripper ever got to the grasp pose
  fig9_jaw_paths            top-down end-effector paths, success vs failure
  fig10_chunk_boundary      action discontinuity at re-plan steps vs within a chunk
  fig11_probe_divergence    how much the perturbed view changes the action (if --probe_nominal)
  fig12_camera_shapley      contribution of each camera
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_analysis import (load_csv, wilson_ci, delta_ci_paired, delta_ci_unpaired,  # noqa: E402
                            mcnemar_exact, holm, did_paired)
from episode_metrics import FAILURE_MODES  # noqa: E402

# Settings that must be identical for two runs to be compared.
MUST_MATCH = ("protocol_version", "outcome_code_hash", "max_steps", "metric_config",
              "backend", "env_seed", "total_episodes")
FAMILY_ORDER = ["nominal", "camera availability", "token dropped", "slot routing", "camera pose",
                "object position", "distractors", "appearance", "other"]
FAMILY_COLORS = {"nominal": "#444444", "camera availability": "#1f77b4", "token dropped": "#17becf", 
                 "slot routing": "#8c564b", "camera pose": "#d62728", "object position": "#9467bd", 
                 "distractors": "#2ca02c", "appearance": "#ff7f0e", "other": "#7f7f7f"}
MODE_COLORS = {"success": "#2ca02c", "over_target_not_released": "#98df8a",
               "grasped_not_delivered": "#ffbb78", "reached_not_grasped": "#ff7f0e",
               "never_reached": "#d62728"}


def family(slug: str) -> str:
    if slug == "nominal":
        return "nominal"
    if slug.startswith("abs-"):
        return "token dropped"
    if slug.startswith("swap-"):
        return "slot routing"
    if any(slug.startswith(p) for p in ("ctx-only", "wrist-only", "top-only", "no-", "all-black",
                                         "ctx-black")):
        return "appearance" if slug.endswith(("-gray", "-noise")) else "camera availability"
    if slug.startswith(("wpos", "wtilt", "ctxpos", "ctxtilt", "toppos", "toptilt")):
        return "camera pose"
    if slug.startswith(("cpos-", "tpos-")):
        return "object position"
    if slug.startswith("sph-"):
        return "distractors"
    if slug.startswith("cube-"):
        return "appearance"
    return "other"


# ══════════════════════════════════════════════════════════════════════════
#  Loading + provenance
# ══════════════════════════════════════════════════════════════════════════

class Run:
    def __init__(self, summary_path: Path, metric: str):
        self.dir = summary_path.parent
        self.s = json.loads(summary_path.read_text())
        self.k = int(self.s.get("n_action_steps") or self.dir.parent.name.lstrip("k"))
        self.slug = self.dir.name
        self.records = load_csv(self.dir / "zeroshot.csv", metric=metric)
        self.y = np.array([r["y"] for r in self.records], dtype=float)
        self.episodes = [r["episode"] for r in self.records]
        self.paired = self.s.get("spawn_source") == "matched"
        tp = self.dir / "trajectories.npz"
        self.traj_path = tp if tp.exists() else None

    @property
    def tsr(self):
        return float(self.y.mean())

    def col(self, name, default=np.nan):
        return np.array([default if r.get(name) is None else r[name] for r in self.records], dtype=float)


def discover(root: Path, metric: str) -> dict:
    runs = {}
    for sp in sorted(root.glob("eval/k*/*/run_summary.json")):
        r = Run(sp, metric)
        runs[(r.k, r.slug)] = r
    if not runs:
        sys.exit(f"[error] no runs found under {root}/eval/k*/<slug>/run_summary.json")
    return runs


def provenance(runs: dict, allow_mismatch: bool, out: Path) -> list[str]:
    lines, problems = [], []
    ref = next(iter(runs.values())).s
    hashes = {r.s.get("matched_eval_set", {}).get("content_hash")
              for r in runs.values() if r.s.get("matched_eval_set")}
    ckpts = {json.dumps(r.s.get("checkpoint_fingerprint"), sort_keys=True) for r in runs.values()}
    lines.append(f"runs: {len(runs)}   horizons: {sorted({k for k, _ in runs})}")
    lines.append(f"matched-set hashes: {sorted(h for h in hashes if h)}")
    lines.append(f"checkpoint fingerprints: {len(ckpts)} distinct")
    if len(hashes) > 1:
        problems.append(f"different matched eval sets in one campaign: {hashes}")
    if len(ckpts) > 1:
        problems.append("more than one checkpoint fingerprint; analyse per checkpoint "
                        "(or use a hierarchical analysis across seeds)")
    for key, r in sorted(runs.items()):
        s = r.s
        lines.append(f"k={r.k:<3} {r.slug:<24} N={s.get('total_episodes'):<4} "
                     f"code={s.get('outcome_code_hash')} git={str(s.get('git_commit'))[:8]}"
                     f"{'+dirty' if s.get('git_dirty') else ''} spawn={s.get('spawn_source')} "
                     f"backend={s.get('backend')}")
        for m in MUST_MATCH:
            if json.dumps(s.get(m), sort_keys=True) != json.dumps(ref.get(m), sort_keys=True):
                problems.append(f"k={r.k} {r.slug}: {m} = {s.get(m)!r} differs from {ref.get(m)!r}")
        if s.get("git_dirty"):
            lines.append("      warning: produced from a dirty git tree")
        if s.get("spawn_source") == "random":
            problems.append(f"k={r.k} {r.slug}: no matched eval set (unpaired)")
    lines.append("")
    lines += ["PROBLEMS:"] + [f"  - {p}" for p in problems] if problems else ["provenance: OK"]
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-max(1, len(problems) + 1):]))
    if problems and not allow_mismatch:
        sys.exit(f"[error] provenance problems (see {out}); fix them or pass --allow_mismatch "
                 f"and report the deviation in the paper.")
    return problems


# ══════════════════════════════════════════════════════════════════════════
#  Tables
# ══════════════════════════════════════════════════════════════════════════

def _pair_ok(a: Run, b: Run) -> bool:
    return a.paired and b.paired and a.episodes == b.episodes


def condition_table(runs: dict, metric: str) -> list[dict]:
    rows = []
    for k in sorted({k for k, _ in runs}):
        nom = runs.get((k, "nominal"))
        block = []
        for (kk, slug), r in sorted(runs.items(), key=lambda kv: (FAMILY_ORDER.index(family(kv[0][1])), kv[0][1])):
            if kk != k:
                continue
            n = len(r.y)
            c = int(r.y.sum())
            lo, hi = wilson_ci(c, n)
            row = {"k": k, "condition": slug, "family": family(slug), "N": n, "TSR": c / n,
                   "TSR_lo": lo, "TSR_hi": hi}
            for name in ("reached_raw", "grasped_raw", "success_carry", "pushed_success"):
                v = r.col(name)
                row[f"P_{name}"] = float(np.nanmean(v)) if not np.all(np.isnan(v)) else None
            row["violations"] = int(np.nansum(r.col("violation_grasp_without_reach", 0)))
            if nom is not None and slug != "nominal":
                if _pair_ok(nom, r):
                    pt, dlo, dhi = delta_ci_paired(nom.y, r.y)
                    mc = mcnemar_exact(nom.y.astype(bool), r.y.astype(bool))
                    row.update(delta=pt, delta_lo=dlo, delta_hi=dhi, paired=True,
                               n01=mc["n01"], n10=mc["n10"], p=mc["p"])
                else:
                    pt, dlo, dhi = delta_ci_unpaired(nom.y, r.y)
                    # unpaired: Fisher-style exact test is the analogue; we use a
                    # permutation-free normal approximation only as a fallback
                    row.update(delta=pt, delta_lo=dlo, delta_hi=dhi, paired=False,
                               p=_two_prop_p(nom.y, r.y))
            block.append(row)
        tested = [r for r in block if r.get("p") is not None]
        for r_, adj in zip(tested, holm([r["p"] for r in tested]) if tested else []):
            r_["p_holm"] = adj
        rows += block
    return rows


def _two_prop_p(a, b):
    """Exact two-sided Fisher test for two independent proportions."""
    from math import comb
    a1, b1, na, nb = int(a.sum()), int(b.sum()), len(a), len(b)
    tot = a1 + b1
    N = na + nb
    def pmf(x):
        return comb(na, x) * comb(nb, tot - x) / comb(N, tot)
    p_obs = pmf(a1)
    lo, hi = max(0, tot - nb), min(tot, na)
    return float(min(1.0, sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p_obs * (1 + 1e-9))))


def ol_cl_table(runs: dict, k_ol: int, k_cl: int) -> list[dict]:
    rows = []
    nom_ol, nom_cl = runs.get((k_ol, "nominal")), runs.get((k_cl, "nominal"))
    slugs = sorted({s for k, s in runs if k == k_ol} & {s for k, s in runs if k == k_cl},
                   key=lambda s: (FAMILY_ORDER.index(family(s)), s))
    for slug in slugs:
        a, b = runs[(k_ol, slug)], runs[(k_cl, slug)]
        row = {"condition": slug, "family": family(slug), f"TSR_k{k_ol}": a.tsr, f"TSR_k{k_cl}": b.tsr}
        if _pair_ok(a, b):
            pt, lo, hi = delta_ci_paired(a.y, b.y)
            mc = mcnemar_exact(a.y.astype(bool), b.y.astype(bool))
            row.update(delta_cl_minus_ol=pt, lo=lo, hi=hi, p=mc["p"])
        if (slug != "nominal" and nom_ol is not None and nom_cl is not None
                and all(_pair_ok(nom_ol, x) for x in (a, nom_cl, b))):
            d, dlo, dhi = did_paired(nom_ol.y, a.y, nom_cl.y, b.y)
            row.update(drop_ol=a.tsr - nom_ol.tsr, drop_cl=b.tsr - nom_cl.tsr,
                       did=d, did_lo=dlo, did_hi=dhi)
        rows.append(row)
    tested = [r for r in rows if r.get("p") is not None]
    for r_, adj in zip(tested, holm([r["p"] for r in tested]) if tested else []):
        r_["p_holm"] = adj
    return rows


CAMERA_SUBSETS = {"nominal": "cwt", "no-top": "cw", "no-wrist": "ct", "no-ctx": "wt",
                  "ctx-only": "c", "wrist-only": "w", "top-only": "t", "all-black": ""}


def camera_shapley(runs: dict, k: int, n_boot: int = 2000, seed: int = 0):
    """Shapley value of each camera for TSR, from the 2^3 blanking factorial.
    Bootstrap CI resamples episode indices jointly across the 8 runs (paired)."""
    sub = {CAMERA_SUBSETS[s]: runs[(k, s)] for s in CAMERA_SUBSETS if (k, s) in runs}
    if len(sub) != 8 or not all(_pair_ok(sub["cwt"], r) for r in sub.values()):
        return None
    Y = {key: r.y for key, r in sub.items()}
    n = len(Y["cwt"])
    players = "cwt"

    def shap(idx):
        v = {key: y[idx].mean() for key, y in Y.items()}
        out = {}
        for p in players:
            others = [q for q in players if q != p]
            tot = 0.0
            for size in range(3):
                for S in combinations(others, size):
                    S_key = "".join(q for q in players if q in S)
                    Sp_key = "".join(q for q in players if q in S or q == p)
                    w = factorial(size) * factorial(3 - size - 1) / factorial(3)
                    tot += w * (v[Sp_key] - v[S_key])
            out[p] = tot
        return out

    point = shap(np.arange(n))
    rng = np.random.default_rng(seed)
    boots = [shap(rng.integers(0, n, n)) for _ in range(n_boot)]
    names = {"c": "context", "w": "wrist", "t": "top"}
    return [{"k": k, "camera": names[p], "shapley": point[p],
             "lo": float(np.percentile([b[p] for b in boots], 2.5)),
             "hi": float(np.percentile([b[p] for b in boots], 97.5)),
             "v_all": float(Y["cwt"].mean()), "v_none": float(Y[""].mean())} for p in players]


def write_table(rows: list[dict], stem: Path, caption: str):
    if not rows:
        return
    import csv
    cols = list(dict.fromkeys(c for r in rows for c in r))
    with open(stem.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        if isinstance(v, float):
            return f"{v:.3g}" if abs(v) < 1e-3 and v != 0 else f"{v:.3f}"
        return str(v)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    stem.with_suffix(".md").write_text(f"{caption}\n\n" + "\n".join(md) + "\n")
    tex = ["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}",
           "\\begin{tabular}{" + "l" * 2 + "r" * (len(cols) - 2) + "}", "\\toprule",
           " & ".join(c.replace("_", "\\_") for c in cols) + " \\\\", "\\midrule"]
    tex += [" & ".join(fmt(r.get(c)).replace("_", "\\_") for c in cols) + " \\\\" for r in rows]
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    stem.with_suffix(".tex").write_text("\n".join(tex) + "\n")


# ══════════════════════════════════════════════════════════════════════════
#  Figures
# ══════════════════════════════════════════════════════════════════════════

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25})
    return plt


def _save(fig, figdir: Path, name: str):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"{name}.{ext}", bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  [fig] {name}")


def _k_style(ks):
    markers = ["o", "s", "^", "D", "v", "P"]
    cols = ["#1f4e79", "#c55a11", "#548235", "#7030a0", "#bf9000", "#2f5597"]
    return {k: (markers[i % 6], cols[i % 6]) for i, k in enumerate(sorted(ks, reverse=True))}


def _k_label(k, chunk=50):
    return f"k={k} (open loop)" if k >= chunk else f"k={k} (closed loop)"


def fig_tsr(rows, figdir, ks):
    plt = _plt()
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    style = _k_style(ks)
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(conds) + 1.2))
    off = np.linspace(-0.25, 0.25, len(ks)) if len(ks) > 1 else [0]
    for o, k in zip(off, sorted(ks, reverse=True)):
        rk = {r["condition"]: r for r in rows if r["k"] == k}
        ys, xs, lo, hi = [], [], [], []
        for i, c in enumerate(conds):
            if c in rk:
                ys.append(i + o); xs.append(rk[c]["TSR"])
                lo.append(rk[c]["TSR"] - rk[c]["TSR_lo"]); hi.append(rk[c]["TSR_hi"] - rk[c]["TSR"])
        m, col = style[k]
        ax.errorbar(xs, ys, xerr=[lo, hi], fmt=m, color=col, ms=4, capsize=2, lw=1, label=_k_label(k))
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels(conds)
    for t, c in zip(ax.get_yticklabels(), conds):
        t.set_color(FAMILY_COLORS[family(c)])
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Task success rate (95% Wilson CI)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(ks), fontsize=8, frameon=False)
    _save(fig, figdir, "fig1_tsr_by_condition")


def fig_forest(rows, figdir, ks):
    plt = _plt()
    rr = [r for r in rows if r.get("delta") is not None]
    if not rr:
        return
    conds = list(dict.fromkeys(r["condition"] for r in rr))
    style = _k_style(ks)
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(conds) + 1.2))
    off = np.linspace(-0.25, 0.25, len(ks)) if len(ks) > 1 else [0]
    for o, k in zip(off, sorted(ks, reverse=True)):
        rk = {r["condition"]: r for r in rr if r["k"] == k}
        m, col = style[k]
        for i, c in enumerate(conds):
            if c not in rk:
                continue
            r = rk[c]
            sig = r.get("p_holm", 1) < 0.05
            ax.errorbar(r["delta"], i + o, xerr=[[r["delta"] - r["delta_lo"]], [r["delta_hi"] - r["delta"]]],
                        fmt=m, color=col, ms=4, capsize=2, lw=1, mfc=col if sig else "white",
                        label=_k_label(k) if i == next(j for j, cc in enumerate(conds) if cc in rk) else None)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels(conds)
    ax.invert_yaxis()
    ax.set_xlabel("Δ TSR vs nominal (paired bootstrap 95% CI; filled = Holm p<0.05)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(ks), fontsize=8, frameon=False)
    _save(fig, figdir, "fig2_delta_forest")


def fig_taxonomy(runs, figdir, ks, order):
    plt = _plt()
    fig, axes = plt.subplots(1, len(ks), figsize=(4.2 * len(ks), 0.26 * len(order) + 1.4), sharey=True,
                             squeeze=False)
    for ax, k in zip(axes[0], sorted(ks, reverse=True)):
        conds = [c for c in order if (k, c) in runs]
        left = np.zeros(len(conds))
        for mode in FAILURE_MODES:
            vals = np.array([np.mean([r.get("failure_mode") == mode for r in runs[(k, c)].records])
                             for c in conds])
            ax.barh(range(len(conds)), vals, left=left, color=MODE_COLORS[mode],
                    label=mode.replace("_", " "), height=0.8)
            left += vals
        ax.set_yticks(range(len(conds)))
        ax.set_yticklabels(conds)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title(_k_label(k))
        ax.set_xlabel("fraction of episodes")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(FAILURE_MODES), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    _save(fig, figdir, "fig3_failure_taxonomy")


def _selected(runs, k, n=5):
    """
    nominal + up to n INFORMATIVE perturbed conditions for the per-episode
    plots: degraded but not trivially dead (0 < TSR < nominal, or TSR = 0 but
    the gripper still reaches the cube). Conditions at 0% everywhere carry no
    timing/geometry information. Picks are spread across the range of TSR.
    """
    nom = runs.get((k, "nominal"))
    ref = nom.tsr if nom else 1.0
    cands = []
    for (kk, s), r in runs.items():
        if kk != k or s == "nominal":
            continue
        reach = np.nanmean(r.col("reached_raw", 0.0))
        if 0 < r.tsr < ref or (r.tsr == 0 and reach > 0):
            cands.append((r.tsr, s))
    cands.sort()
    if len(cands) > n:
        idx = sorted(set(np.linspace(0, len(cands) - 1, n).round().astype(int)))
        cands = [cands[i] for i in idx]
    return (["nominal"] if nom else []) + [s for _, s in cands]


def _selection_colors(runs, ks):
    """One condition selection (chosen at the closed-loop horizon) and one
    colour per condition, shared by every panel and figure."""
    sel = _selected(runs, min(ks))
    pal = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    return sel, {s: ("k" if s == "nominal" else pal[i % len(pal)]) for i, s in enumerate(sel)}


def fig_stage_timing(runs, figdir, ks, max_steps):
    plt = _plt()
    stages = [("t_reach", "reached grasp pose"), ("t_grasp", "stable grasp"), ("t_success", "success")]
    fig, axes = plt.subplots(len(ks), 3, figsize=(10, 2.8 * len(ks)), squeeze=False, sharex=True, sharey=True)
    t = np.arange(max_steps + 1)
    sel, colors = _selection_colors(runs, ks)
    for row, k in enumerate(sorted(ks, reverse=True)):
        for col, (key, name) in enumerate(stages):
            ax = axes[row][col]
            for s in sel:
                if (k, s) not in runs:
                    continue
                v = runs[(k, s)].col(key)
                cdf = [(np.nan_to_num(v, nan=np.inf) <= tt).mean() for tt in t]
                ax.step(t * 0.1, cdf, where="post", lw=1.6 if s == "nominal" else 1.1,
                        color=colors[s], label=s)
            ax.set_title(f"{_k_label(k)}: {name}", fontsize=9)
            if row == len(ks) - 1:
                ax.set_xlabel("time (s)")
            if col == 0:
                ax.set_ylabel("cumulative fraction")
            if col == 2:
                ax.legend(fontsize=7, loc="upper left")
            if k < 50:
                for x in np.arange(0, max_steps, k) * 0.1:
                    ax.axvline(x, color="orange", lw=0.4, alpha=0.5)
    _save(fig, figdir, "fig4_stage_timing")


def fig_spatial(runs, figdir, k, order):
    plt = _plt()
    conds = [c for c in order if (k, c) in runs and runs[(k, c)].s.get("spawn_source") != "preset"]
    if not conds:
        return
    nc = min(6, len(conds))
    nr = int(np.ceil(len(conds) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(2.2 * nc, 2.2 * nr), squeeze=False, sharex=True, sharey=True)
    nom = runs.get((k, "nominal"))
    for ax, c in zip(axes.flat, conds):
        r = runs[(k, c)]
        x, y = r.col("cube_x"), r.col("cube_y")
        ok = r.y.astype(bool)
        ax.scatter(x[ok], y[ok], s=9, c="#2ca02c", label="success")
        ax.scatter(x[~ok], y[~ok], s=9, c="#d62728", marker="x", label="failure")
        if nom is not None and _pair_ok(nom, r) and c != "nominal":
            lost = nom.y.astype(bool) & ~ok
            ax.scatter(x[lost], y[lost], s=40, facecolors="none", edgecolors="k", lw=0.6,
                       label="nominal ✓ → ✗")
        ax.set_title(f"{c}  ({r.tsr:.0%})", fontsize=8)
        ax.set_aspect("equal")
    for ax in axes.flat[len(conds):]:
        ax.axis("off")
    axes[0][0].legend(fontsize=6, loc="lower left")
    for ax in axes[-1]:
        ax.set_xlabel("cube x (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("cube y (m)")
    fig.suptitle(f"Initial cube position vs outcome, {_k_label(k)}", fontsize=10)
    _save(fig, figdir, f"fig5_spatial_success_k{k}")


def _dose_key(r: Run):
    p = r.s.get("camera_perturbation_params") or {}
    if p.get("blackout_mode") not in (None, "none", "3-cam"):
        return None
    if p.get("tilt_target") and not p.get("position_target"):
        return (f"{p['tilt_target']} camera tilt (deg)", float(p["tilt_deg"]))
    if p.get("position_target") and not p.get("tilt_target"):
        off = np.asarray(p["position_offset"], float)
        nz = np.flatnonzero(off)
        if len(nz) == 1:
            return (f"{p['position_target']} camera shift along {'xyz'[nz[0]]} (cm)", 100 * off[nz[0]])
    return None


def fig_dose(runs, figdir, ks):
    plt = _plt()
    fams = {}
    for (k, s), r in runs.items():
        dk = _dose_key(r)
        if dk:
            fams.setdefault(dk[0], {}).setdefault(k, []).append((dk[1], r))
    fams = {f: v for f, v in fams.items() if max(len(x) for x in v.values()) >= 2}
    if not fams:
        return
    style = _k_style(ks)
    fig, axes = plt.subplots(1, len(fams), figsize=(3.6 * len(fams), 3), squeeze=False)
    for ax, (fam, byk) in zip(axes[0], sorted(fams.items())):
        for k, pts in sorted(byk.items(), reverse=True):
            nom = runs.get((k, "nominal"))
            pts = sorted(pts + ([(0.0, nom)] if nom else []), key=lambda t: t[0])
            span = (max(p[0] for p in pts) - min(p[0] for p in pts)) or 1.0
            jit = 0.012 * span * (sorted(byk, reverse=True).index(k) - (len(byk) - 1) / 2)
            x = [p[0] + jit for p in pts]                   # small offset so horizons don't overlap
            y = [p[1].tsr for p in pts]
            ci = [wilson_ci(int(p[1].y.sum()), len(p[1].y)) for p in pts]
            m, col = style[k]
            ax.errorbar(x, y, yerr=[[a - c[0] for a, c in zip(y, ci)], [c[1] - a for a, c in zip(y, ci)]],
                        fmt=m + "-", color=col, ms=4, capsize=2, lw=1, label=_k_label(k))
        ax.set_xlabel(fam)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("TSR")
        ax.axvline(0, color="k", lw=0.5)
    axes[0][0].legend(fontsize=7)
    _save(fig, figdir, "fig6_dose_response")


def fig_horizon(runs, figdir, ks):
    if len(ks) < 3:
        return
    plt = _plt()
    slugs = sorted({s for k, s in runs if all((kk, s) in runs for kk in ks)},
                   key=lambda s: (FAMILY_ORDER.index(family(s)), s))
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for s in slugs:
        ax.plot(sorted(ks), [runs[(k, s)].tsr for k in sorted(ks)], "o-", ms=3,
                lw=2 if s == "nominal" else 1, color="k" if s == "nominal" else FAMILY_COLORS[family(s)],
                alpha=1 if s == "nominal" else 0.7, label=s if s == "nominal" else None)
    ax.set_xscale("log")
    ax.set_xticks(sorted(ks))
    ax.set_xticklabels(sorted(ks))
    ax.set_xlabel("execution horizon k (actions per inference)")
    ax.set_ylabel("TSR")
    ax.legend(fontsize=7)
    _save(fig, figdir, "fig7_horizon_sweep")


def fig_ecdf(runs, figdir, ks, reach_radius):
    plt = _plt()
    fig, axes = plt.subplots(1, len(ks), figsize=(4.2 * len(ks), 3), squeeze=False, sharey=True)
    sel, colors = _selection_colors(runs, ks)
    for ax, k in zip(axes[0], sorted(ks, reverse=True)):
        for s in sel:
            if (k, s) not in runs:
                continue
            v = np.sort(runs[(k, s)].col("min_grasp_dist") * 100)
            v = v[~np.isnan(v)]
            if len(v):
                ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post",
                        color=colors[s], lw=1.6 if s == "nominal" else 1, label=s)
        ax.axvline(reach_radius * 100, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("closest approach to grasp pose (cm)")
        ax.set_title(_k_label(k))
        ax.set_xlim(0, None)
    axes[0][0].set_ylabel("cumulative fraction of episodes")
    axes[0][-1].legend(fontsize=7)
    _save(fig, figdir, "fig8_min_dist_ecdf")


def fig_paths(runs, figdir, ks, slugs):
    plt = _plt()
    panels = [(k, s) for s in slugs for k in sorted(ks, reverse=True)
              if (k, s) in runs and runs[(k, s)].traj_path]
    if not panels:
        return
    nc = len(ks)
    nr = int(np.ceil(len(panels) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.4 * nc, 3.2 * nr), squeeze=False, sharex=True, sharey=True)
    for ax, (k, s) in zip(axes.flat, panels):
        r = runs[(k, s)]
        tr = np.load(r.traj_path)
        succ = {rec["episode"]: rec["y"] for rec in r.records}
        for ep in np.unique(tr["episode"]):
            m = tr["episode"] == ep
            jaw = tr["jaw"][m]
            ax.plot(jaw[:, 0], jaw[:, 1], lw=0.6, alpha=0.5, color="#2ca02c" if succ.get(int(ep)) else "#d62728")
        ax.scatter([rec["cube_x"] for rec in r.records], [rec["cube_y"] for rec in r.records],
                   s=4, c="k", label="cube start")
        ax.scatter(tr["target_xyz"][:, 0], tr["target_xyz"][:, 1], s=4, c="#17becf", label="target")
        ax.set_title(f"{s}, {_k_label(k)} ({r.tsr:.0%})", fontsize=8)
        ax.set_aspect("equal")
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    axes[0][0].legend(fontsize=6)
    fig.suptitle("Top-down jaw paths (green = success, red = failure)", fontsize=10)
    _save(fig, figdir, "fig9_jaw_paths")


def fig_chunk_boundary(runs, figdir, ks):
    plt = _plt()
    data = []
    for k in sorted(ks, reverse=True):
        r = runs.get((k, "nominal"))
        if r is None or r.traj_path is None:
            continue
        tr = np.load(r.traj_path)
        ep, act, rep = tr["episode"], tr["action"][:, :5], tr["replan"].astype(bool)
        same = np.r_[False, ep[1:] == ep[:-1]]
        jump = np.r_[np.nan, np.linalg.norm(np.diff(act, axis=0), axis=1)]
        data.append((k, jump[same & rep], jump[same & ~rep]))
    if not data:
        return
    fig, ax = plt.subplots(figsize=(4.5, 3))
    pos, labels = [], []
    for i, (k, at, within) in enumerate(data):
        for j, (vals, name) in enumerate(((within, "within chunk"), (at, "at re-plan"))):
            if len(vals):
                ax.boxplot(vals, positions=[3 * i + j], widths=0.7, showfliers=False)
            pos.append(3 * i + j)
            labels.append(f"k={k}\n{name}")
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("‖Δ action‖ between steps (rad, arm)")
    ax.set_title("Action discontinuity at chunk boundaries (nominal)", fontsize=9)
    _save(fig, figdir, "fig10_chunk_boundary")


def fig_probe(runs, figdir, ks):
    plt = _plt()
    have = [(k, s) for (k, s), r in runs.items() if r.traj_path is not None
            and not np.all(np.isnan(r.col("probe_div_mean")))]
    if not have:
        return
    fig, axes = plt.subplots(1, len(ks), figsize=(4.4 * len(ks), 3), squeeze=False, sharey=True)
    for ax, k in zip(axes[0], sorted(ks, reverse=True)):
        for kk, s in sorted(have):
            if kk != k:
                continue
            tr = np.load(runs[(k, s)].traj_path)
            m = tr["replan"].astype(bool) & ~np.isnan(tr["probe_div"])
            t, d = tr["t"][m], tr["probe_div"][m]
            ts = np.unique(t)
            ax.plot(ts * 0.1, [d[t == x].mean() for x in ts], "o-", ms=3, lw=1, label=s)
        ax.set_title(_k_label(k))
        ax.set_xlabel("re-plan time (s)")
    axes[0][0].set_ylabel("‖a(perturbed) − a(nominal view)‖ (rad)")
    axes[0][-1].legend(fontsize=7)
    _save(fig, figdir, "fig11_probe_divergence")


def fig_shapley(rows, figdir):
    if not rows:
        return
    plt = _plt()
    ks = sorted({r["k"] for r in rows}, reverse=True)
    cams = ["context", "wrist", "top"]
    style = _k_style(ks)
    fig, ax = plt.subplots(figsize=(4.5, 3))
    w = 0.8 / len(ks)
    for i, k in enumerate(ks):
        rk = {r["camera"]: r for r in rows if r["k"] == k}
        x = np.arange(3) + i * w
        ax.bar(x, [rk[c]["shapley"] for c in cams], width=w, color=style[k][1], label=_k_label(k),
               yerr=[[rk[c]["shapley"] - rk[c]["lo"] for c in cams], [rk[c]["hi"] - rk[c]["shapley"] for c in cams]],
               capsize=2)
    ax.set_xticks(np.arange(3) + w * (len(ks) - 1) / 2)
    ax.set_xticklabels(cams)
    ax.set_ylabel("Shapley contribution to TSR")
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(fontsize=7)
    _save(fig, figdir, "fig12_camera_shapley")


# ══════════════════════════════════════════════════════════════════════════

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_root", required=True, help="e.g. runs/paper-v2")
    ap.add_argument("--metric", default="success",
                    help="CSV column used as the outcome (success = the run's --success_metric).")
    ap.add_argument("--ol", type=int, default=None, help="open-loop horizon (default: largest k)")
    ap.add_argument("--cl", type=int, default=None, help="closed-loop horizon (default: smallest k)")
    ap.add_argument("--paths_for", default=None, help="comma-separated slugs for fig9")
    ap.add_argument("--allow_mismatch", action="store_true")
    ap.add_argument("--no_figures", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.run_root)
    out = root / "analysis"
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    runs = discover(root, args.metric)
    ks = sorted({k for k, _ in runs})
    problems = provenance(runs, args.allow_mismatch, out / "provenance_report.txt")

    rows = condition_table(runs, args.metric)
    write_table(rows, out / "table_conditions",
                f"Task success per condition and execution horizon (metric: {args.metric}). "
                f"Delta vs nominal with paired bootstrap 95% CI; exact McNemar p, Holm-adjusted per horizon.")
    k_ol = args.ol or max(ks)
    k_cl = args.cl or min(ks)
    olcl = ol_cl_table(runs, k_ol, k_cl) if k_ol != k_cl else []
    write_table(olcl, out / "table_ol_vs_cl",
                f"Open loop (k={k_ol}) vs closed loop (k={k_cl}). did = (pert-nom)_CL - (pert-nom)_OL "
                f"per episode, paired bootstrap 95% CI; negative = closed loop loses more.")
    shap = [row for k in ks for row in (camera_shapley(runs, k) or [])]
    write_table(shap, out / "table_camera_shapley", "Camera Shapley values for TSR")

    order = list(dict.fromkeys(r["condition"] for r in rows))
    if not args.no_figures:
        any_run = next(iter(runs.values()))
        max_steps = int(any_run.s.get("max_steps", 60))
        reach = float((any_run.s.get("metric_config") or {}).get("reach_radius", 0.04))
        fig_tsr(rows, figdir, ks)
        fig_forest(rows, figdir, ks)
        if "failure_mode" in any_run.records[0]:
            fig_taxonomy(runs, figdir, ks, order)
            fig_stage_timing(runs, figdir, ks, max_steps)
            fig_ecdf(runs, figdir, ks, reach)
        for k in ks:
            fig_spatial(runs, figdir, k, order)
        fig_dose(runs, figdir, ks)
        fig_horizon(runs, figdir, ks)
        paths_for = (args.paths_for.split(",") if args.paths_for
                     else ["nominal"] + [s for s in _selected(runs, k_cl, 2) if s != "nominal"][:2])
        fig_paths(runs, figdir, ks, paths_for)
        fig_chunk_boundary(runs, figdir, ks)
        fig_probe(runs, figdir, ks)
        fig_shapley(shap, figdir)

    (out / "analysis_summary.json").write_text(json.dumps(
        {"metric": args.metric, "horizons": ks, "ol": k_ol, "cl": k_cl,
         "provenance_problems": problems, "conditions": rows, "ol_vs_cl": olcl,
         "camera_shapley": shap}, indent=2, default=float))
    print(f"[ok] analysis written to {out}")


if __name__ == "__main__":
    main()