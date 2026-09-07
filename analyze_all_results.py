"""
analyze_all_results.py
-----------------------
Reads ALL evaluation CSV files produced by zeroshot_eval.py and generates
publication-quality figures suitable for a robotics paper.

Usage (run from the project root, where results/ lives):
    python training/analyze_all_results.py --results_root results/

Output (all written to results/figures/):
    fig1_overview_bar.pdf/png       — TSR + PSS for every condition
    fig2_camera_ablation.pdf/png    — Single-camera-only ablation
    fig3_camera_combo.pdf/png       — Two-camera-present ablation
    fig4_missing_camera.pdf/png     — "Absent" (dropped-input) ablation
    fig5_scene_perturb.pdf/png      — Scene-level perturbations
    fig6_combined_4panel.pdf/png    — Paper-ready 4-panel summary
    summary_table.txt               — LaTeX-ready table

Requirements: numpy, matplotlib, scipy (for Wilson CI)
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   9.5,
    "ytick.labelsize":   9.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

# Palette (colourblind-friendly)
C_BLUE   = "#4C8BF5"
C_GREEN  = "#34A853"
C_AMBER  = "#FBBC04"
C_RED    = "#EA4335"
C_PURPLE = "#9B59B6"
C_TEAL   = "#17A589"
C_GRAY   = "#95A5A6"
C_DARK   = "#2C3E50"


# ══════════════════════════════════════════════════════════════════════════════
#  Data registry  — maps condition label → CSV path
#  Edit these paths if you stored files elsewhere.
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  Data registry — built from the SAME CameraPerturbation / ScenePerturbation
#  objects and the SAME condition_slug() function that zeroshot_eval.py uses
#  to pick its --run_root-derived output path. This means the two scripts
#  can never disagree about where a condition's CSV lives — there is no
#  hand-typed path string here that could drift out of sync with what
#  zeroshot_eval.py actually wrote.
#
#  Add a new perturbation to the sweep by adding one entry here; its CSV
#  path is computed automatically from --results_root + its slug.
# ══════════════════════════════════════════════════════════════════════════════

from scene_params import CameraPerturbation, ScenePerturbation, SPHERE_PRESETS, CUBE_POSITIONS
from run_naming import condition_slug

CP = CameraPerturbation
SP = ScenePerturbation


def _cube_displacement_cm(preset: str) -> float:
    """Euclidean XY displacement of a named cube-position preset from the
    'nominal' spawn center, in cm — gives the S1 sweep a continuous x-axis
    for otherwise-categorical position presets."""
    nominal = np.array(CUBE_POSITIONS["nominal"]["cube_xy"])
    this    = np.array(CUBE_POSITIONS[preset]["cube_xy"])
    return float(np.linalg.norm(this - nominal) * 100)


CONDITION_SPECS = [
    # Each entry: (display_label, CameraPerturbation, ScenePerturbation,
    #              group, color, sweep) where sweep is either None or
    # {"axis": <name shared across a curve>, "value": <signed magnitude>}.
    # "axis" groups points into one S1 sensitivity curve; the baseline
    # condition is injected as the value=0 point on EVERY axis rather than
    # being listed as its own sweep member below.

    ("Baseline\n(all cams)",        CP(),                                                     SP(), "baseline",   C_GREEN,  None),

    # ── Single-camera-only ──────────────────────────────────────────────────
    ("Context\nonly",               CP(blackout_mode="context_only"),                         SP(), "single_cam", C_BLUE,   None),
    ("Wrist\nonly",                 CP(blackout_mode="wrist_only"),                            SP(), "single_cam", C_AMBER,  None),
    ("Top\nonly",                   CP(blackout_mode="top_only"),                              SP(), "single_cam", C_PURPLE, None),

    # ── Two-camera combos ───────────────────────────────────────────────────
    ("No wrist\n(ctx+top)",         CP(blackout_mode="no_wrist"),                              SP(), "two_cam",    C_BLUE,   None),
    ("No top\n(ctx+wrist)",         CP(blackout_mode="no_top"),                                SP(), "two_cam",    C_TEAL,   None),
    ("No context\n(wrist+top)",     CP(blackout_mode="no_context"),                            SP(), "two_cam",    C_AMBER,  None),

    # ── Absent (tensor dropped, not zeroed) ─────────────────────────────────
    ("Absent top",                  CP(blackout_mode="absent_top"),                            SP(), "absent",     C_TEAL,   None),
    ("Absent\nwrist+top",           CP(blackout_mode="absent_wrist_top"),                       SP(), "absent",     C_RED,    None),

    # ── S1 sensitivity sweep: wrist camera translation, both sides of nominal
    ("Wrist shift\n-3 cm",          CP(position_target="wrist", position_offset=(-0.03, 0, 0)), SP(), "cam_pos",    C_AMBER,  {"axis": "wrist_pos_cm",   "value": -3.0}),
    ("Wrist shift\n+1.5 cm",        CP(position_target="wrist", position_offset=(0.015, 0, 0)), SP(), "cam_pos",    C_AMBER,  {"axis": "wrist_pos_cm",   "value": 1.5}),
    ("Wrist shift\n+3 cm",          CP(position_target="wrist", position_offset=(0.03, 0, 0)),  SP(), "cam_pos",    C_AMBER,  {"axis": "wrist_pos_cm",   "value": 3.0}),

    # ── S1 sensitivity sweep: wrist camera tilt, both sides of nominal ──────
    ("Wrist tilt\n-10 deg",         CP(tilt_target="wrist", tilt_deg=-10.0),                    SP(), "cam_tilt",   C_TEAL,   {"axis": "wrist_tilt_deg", "value": -10.0}),
    ("Wrist tilt\n+10 deg",         CP(tilt_target="wrist", tilt_deg=10.0),                      SP(), "cam_tilt",   C_TEAL,   {"axis": "wrist_tilt_deg", "value": 10.0}),

    # ── S1 sensitivity sweep: object-position displacement (one-sided —
    #    CUBE_POSITIONS has no "near_left"-and-mirror pair on this axis;
    #    add far_right/near_right here too if you run them) ────────────────
    ("Cube pos\nnear_left",         CP(),  SP(cube_position_preset="near_left"),                "cube_pos",   C_PURPLE, {"axis": "cube_pos_cm", "value": _cube_displacement_cm("near_left")}),
    ("Cube pos\nfar_left",          CP(),  SP(cube_position_preset="far_left"),                  "cube_pos",   C_PURPLE, {"axis": "cube_pos_cm", "value": _cube_displacement_cm("far_left")}),

    # ── S3: distractor confound decomposition ───────────────────────────────
    ("Two flanking\nspheres",       CP(),  SP(spheres=SPHERE_PRESETS["two_flanking"]),           "scene",      C_RED,    None),
    ("Color-matched\nsphere",       CP(),  SP(spheres=SPHERE_PRESETS["color_match"]),             "scene",      C_PURPLE, None),
]


def build_conditions(results_root: Path) -> dict:
    """
    Turn CONDITION_SPECS into the {label: {csv, group, color, sweep}} dict
    the rest of this script expects, resolving each CSV path as
    <results_root>/<condition_slug>/zeroshot.csv — the exact same slug
    zeroshot_eval.py computes for the same CameraPerturbation/
    ScenePerturbation pair when run with --run_root.
    """
    conditions = {}
    for label, cam_pert, scene_pert, group, color, sweep in CONDITION_SPECS:
        slug = condition_slug(cam_pert, scene_pert)
        conditions[label] = {
            "csv":   str(results_root / slug / "zeroshot.csv"),
            "slug":  slug,
            "group": group,
            "color": color,
            "sweep": sweep,
        }
    return conditions


N_EPISODES = 100   # expected episode count per condition; used only to WARN
                    # on mismatch, never to override the CI calculation


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def wilson_ci(p, n, z=1.96):
    denom  = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half   = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def load_csv(path):
    path = Path(path)
    if not path.exists():
        return None
    records = []
    with open(path) as f:
        for row in csv.DictReader(f):
            records.append({
                "success": row["success"].strip().lower() == "true",
                "pss":     float(row["pss"]),
                "steps":   int(row["steps"]),
                "reached": row.get("reached", "false").strip().lower() == "true",
                "lifted":  row.get("lifted",  "false").strip().lower() == "true",
                "placed":  row.get("placed",  "false").strip().lower() == "true",
            })
    return records


def load_all_conditions(conditions, expected_n=N_EPISODES):
    """
    Return a dict mapping label → {tsr, pss, lo, hi, n, records}.

    Every number is computed directly from the CSV that was actually
    loaded. There is no fallback to hardcoded values: a missing or
    unreadable CSV is a hard error, because a plot built from an
    unverifiable number is worse than no plot at all.

    The Wilson CI always uses the *actual* row count of the loaded CSV,
    not `expected_n`. `expected_n` is only used to print a warning if
    the file's row count doesn't match what you thought you ran.
    """
    data = {}
    missing = []

    for label, cfg in conditions.items():
        csv_path = cfg.get("csv")
        records  = load_csv(csv_path) if csv_path else None

        if not records:
            missing.append((label, csv_path))
            continue

        n   = len(records)
        tsr = sum(r["success"] for r in records) / n
        pss = float(np.mean([r["pss"] for r in records]))
        lo, hi = wilson_ci(tsr, n)

        data[label] = {
            "tsr": tsr, "pss": pss, "lo": lo, "hi": hi, "n": n,
            "records": records, "group": cfg["group"], "color": cfg["color"],
            "sweep": cfg.get("sweep"),
        }

        flag = "" if n == expected_n else f"  [warn: n={n}, expected {expected_n}]"
        print(f"  [CSV] {label.replace(chr(10),' '):35s}  "
              f"TSR={tsr:.0%}  PSS={pss:.3f}  N={n}{flag}")

    if missing:
        print("\n[error] The following conditions have no readable CSV and "
              "were NOT included in any figure or table:")
        for label, path in missing:
            print(f"    - {label.replace(chr(10), ' '):35s}  expected at: {path}")
        print("\nRun the corresponding zeroshot_eval.py command for each "
              "before re-running this script. Figures are only generated "
              "for conditions with real data — nothing is silently "
              "substituted.\n")

    return data


def _save(fig, stem, out_dir):
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}")
    plt.close(fig)
    print(f"[ok] {stem}.pdf/png")


def _require(data, keys, fig_name):
    """Return True only if every key needed for this figure has real data."""
    missing = [k for k in keys if k not in data]
    if missing:
        print(f"[skip] {fig_name}: missing data for "
              f"{[m.replace(chr(10), ' ') for m in missing]}")
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 1 — Full overview bar chart
# ══════════════════════════════════════════════════════════════════════════════

def fig_overview(data, out_dir):
    if not data:
        print("[skip] fig1_overview_bar: no conditions loaded")
        return

    labels = list(data.keys())
    tsrs   = [data[l]["tsr"] for l in labels]
    psss   = [data[l]["pss"] for l in labels]
    los    = [data[l]["tsr"] - data[l]["lo"] for l in labels]
    his    = [data[l]["hi"] - data[l]["tsr"] for l in labels]
    colors = [data[l]["color"] for l in labels]

    n = len(labels)
    x = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={"hspace": 0.08})

    # TSR
    bars = ax1.bar(x, tsrs, color=colors, alpha=0.88, width=0.6,
                   yerr=[los, his], capsize=5,
                   error_kw={"elinewidth": 1.4, "ecolor": "#555"})
    ax1.set_ylim(0, 1.18)
    ax1.set_ylabel("Task Success Rate (TSR)", fontweight="bold")
    baseline_key = "Baseline\n(all cams)"
    if baseline_key in data:
        ax1.axhline(data[baseline_key]["tsr"], color=C_GRAY,
                    ls="--", lw=1.2, label="Baseline TSR")
        ax1.legend(fontsize=9, loc="upper right")
    for bar, tsr in zip(bars, tsrs):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 min(tsr + 0.06, 1.08),
                 f"{tsr:.0%}", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold", color=C_DARK)

    # PSS
    ax2.bar(x, psss, color=colors, alpha=0.75, width=0.6)
    ax2.set_ylim(0, 1.65)
    ax2.set_ylabel("Mean PSS", fontweight="bold")
    if baseline_key in data:
        ax2.axhline(data[baseline_key]["pss"], color=C_GRAY, ls="--", lw=1.2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    for i, pss in enumerate(psss):
        ax2.text(i, pss + 0.04, f"{pss:.2f}", ha="center",
                 fontsize=8, color=C_DARK)

    # N varies per condition when re-running incrementally, so the title
    # reports the actual range rather than assuming a fixed N for everyone.
    ns = sorted({data[l]["n"] for l in labels})
    n_label = f"N={ns[0]}" if len(ns) == 1 else f"N={ns[0]}–{ns[-1]} (varies by condition)"
    fig.suptitle(
        f"SmolVLA Fine-tuned — Evaluation Across All Conditions  ({n_label})",
        fontsize=13, fontweight="bold", y=1.01,
    )
    _save(fig, "fig1_overview_bar", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 2 — Single-camera ablation
# ══════════════════════════════════════════════════════════════════════════════

def fig_single_cam(data, out_dir):
    groups = {
        "Baseline\n(all cams)":  ("All 3 cameras", C_GREEN),
        "Context\nonly":         ("Context only",  C_BLUE),
        "Wrist\nonly":           ("Wrist only",    C_AMBER),
        "Top\nonly":             ("Top only",      C_PURPLE),
    }
    keys = list(groups.keys())
    if not _require(data, keys, "fig2_camera_ablation"):
        return
    labels_out = ["All 3\ncameras", "Context\nonly", "Wrist\nonly", "Top\nonly"]
    colors = [v[1] for v in groups.values()]

    tsrs = [data[k]["tsr"] for k in keys]
    psss = [data[k]["pss"] for k in keys]
    los  = [data[k]["tsr"] - data[k]["lo"] for k in keys]
    his  = [data[k]["hi"] - data[k]["tsr"] for k in keys]

    x = np.arange(len(keys))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    for ax, vals, ylabel, title in [
        (ax1, (tsrs, [los, his]), "Task Success Rate (TSR)", "(A) TSR"),
        (ax2, (psss, None),       "Mean PSS",               "(B) PSS"),
    ]:
        if vals[1] is not None:
            bars = ax.bar(x, vals[0], color=colors, alpha=0.88, width=0.55,
                          yerr=vals[1], capsize=6,
                          error_kw={"elinewidth": 1.5, "ecolor": "#555"})
            ax.set_ylim(0, 1.2)
        else:
            bars = ax.bar(x, vals[0], color=colors, alpha=0.88, width=0.55)
            ax.set_ylim(0, 1.7)

        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_out, fontsize=9.5)
        for bar, v in zip(bars, vals[0]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (0.03 if vals[1] is None else 0.06),
                    f"{v:.0%}" if vals[1] is not None else f"{v:.3f}",
                    ha="center", fontsize=10, fontweight="bold", color=C_DARK)

    fig.suptitle(
        "Single-Camera Ablation — Which camera matters most?\n"
        "(Each condition: only that camera's stream is shown to the policy)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, "fig2_camera_ablation", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 3 — Two-camera combo ablation
# ══════════════════════════════════════════════════════════════════════════════

def fig_two_cam(data, out_dir):
    keys  = [
        "Baseline\n(all cams)",
        "No top\n(ctx+wrist)",
        "No wrist\n(ctx+top)",
        "No context\n(wrist+top)",
    ]
    if not _require(data, keys, "fig3_camera_combo"):
        return
    labels_out = [
        "All 3\ncams",
        "Ctx + Wrist\n(no top)",
        "Ctx + Top\n(no wrist)",
        "Wrist + Top\n(no ctx)",
    ]
    colors = [data[k]["color"] for k in keys]
    tsrs   = [data[k]["tsr"] for k in keys]
    psss   = [data[k]["pss"] for k in keys]
    los    = [data[k]["tsr"] - data[k]["lo"] for k in keys]
    his    = [data[k]["hi"] - data[k]["tsr"] for k in keys]

    x = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, vals, yerr, ylabel, title, ylim in [
        (axes[0], tsrs, [los, his], "TSR", "(A) Task Success Rate", (0, 1.2)),
        (axes[1], psss, None,       "Mean PSS", "(B) Partial Success Score", (0, 1.7)),
    ]:
        if yerr:
            bars = ax.bar(x, vals, color=colors, alpha=0.88, width=0.55,
                          yerr=yerr, capsize=6,
                          error_kw={"elinewidth": 1.5, "ecolor": "#555"})
        else:
            bars = ax.bar(x, vals, color=colors, alpha=0.88, width=0.55)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_out, fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (0.06 if yerr else 0.03),
                    f"{v:.0%}" if yerr else f"{v:.3f}",
                    ha="center", fontsize=10, fontweight="bold", color=C_DARK)

    fig.suptitle(
        "Two-Camera Combination Ablation\n"
        "(One camera zeroed per condition; policy still receives all 3 tensor slots)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, "fig3_camera_combo", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 4 — Absent / missing camera ablation
# ══════════════════════════════════════════════════════════════════════════════

def fig_absent(data, out_dir):
    keys = [
        "Baseline\n(all cams)",
        "Absent top",
        "Absent\nwrist+top",
    ]
    if not _require(data, keys, "fig4_missing_camera"):
        return
    labels_out = [
        "All 3 cams\n(baseline)",
        "Top absent\n(ctx+wrist remain)",
        "Wrist+Top absent\n(ctx only)",
    ]
    colors = [data[k]["color"] for k in keys]
    tsrs   = [data[k]["tsr"] for k in keys]
    psss   = [data[k]["pss"] for k in keys]
    los    = [data[k]["tsr"] - data[k]["lo"] for k in keys]
    his    = [data[k]["hi"] - data[k]["tsr"] for k in keys]

    x = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, vals, yerr, ylabel, title, ylim in [
        (axes[0], tsrs, [los, his], "TSR", "(A) Task Success Rate", (0, 1.2)),
        (axes[1], psss, None,       "Mean PSS", "(B) Partial Success Score", (0, 1.7)),
    ]:
        if yerr:
            bars = ax.bar(x, vals, color=colors, alpha=0.88, width=0.55,
                          yerr=yerr, capsize=6,
                          error_kw={"elinewidth": 1.5, "ecolor": "#555"})
        else:
            bars = ax.bar(x, vals, color=colors, alpha=0.88, width=0.55)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_out, fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (0.06 if yerr else 0.03),
                    f"{v:.0%}" if yerr else f"{v:.3f}",
                    ha="center", fontsize=10, fontweight="bold", color=C_DARK)

    fig.suptitle(
        "Missing / Absent Camera Inputs\n"
        "(Camera tensor omitted entirely — tests robustness to missing modalities)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, "fig4_missing_camera", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 5 — Scene perturbations
# ══════════════════════════════════════════════════════════════════════════════

def fig_scene(data, out_dir):
    keys = [
        "Baseline\n(all cams)",
        "Wrist shift\n+3 cm",
        "Two flanking\nspheres",
        "Color-matched\nsphere",
    ]
    if not _require(data, keys, "fig5_scene_perturb"):
        return
    labels_out = [
        "Baseline\n(nominal)",
        "Wrist cam\nshifted +3 cm",
        "Two distractor\nspheres",
        "Color-matched\nsphere",
    ]
    colors = [data[k]["color"] for k in keys]
    tsrs   = [data[k]["tsr"] for k in keys]
    psss   = [data[k]["pss"] for k in keys]
    los    = [data[k]["tsr"] - data[k]["lo"] for k in keys]
    his    = [data[k]["hi"] - data[k]["tsr"] for k in keys]

    x = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, vals, yerr, ylabel, title, ylim in [
        (axes[0], tsrs, [los, his], "TSR", "(A) Task Success Rate", (0, 1.2)),
        (axes[1], psss, None,       "Mean PSS", "(B) Partial Success Score", (0, 1.7)),
    ]:
        if yerr:
            bars = ax.bar(x, vals, color=colors, alpha=0.88, width=0.55,
                          yerr=yerr, capsize=6,
                          error_kw={"elinewidth": 1.5, "ecolor": "#555"})
        else:
            bars = ax.bar(x, vals, color=colors, alpha=0.88, width=0.55)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_out, fontsize=9.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (0.06 if yerr else 0.03),
                    f"{v:.0%}" if yerr else f"{v:.3f}",
                    ha="center", fontsize=10.5, fontweight="bold", color=C_DARK)

    fig.suptitle(
        "Scene-Level Perturbations\n"
        "(Camera position shift vs. visual distractors: color salience vs. color-matched)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, "fig5_scene_perturb", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  Figure S1 — sensitivity curves J(delta), not just J(0) vs J(delta*)
#
#  This is the direct answer to worksheet item S1: for every perturbation
#  axis with multiple levels, plot TSR (and PSS) AS A FUNCTION OF the
#  perturbation magnitude, with the nominal condition injected as the
#  delta=0 point on every curve, rather than a bar chart comparing only
#  two points.
# ══════════════════════════════════════════════════════════════════════════════

_SWEEP_AXIS_META = {
    "wrist_pos_cm":   {"xlabel": "Wrist camera shift (cm)", "title": "Camera translation"},
    "wrist_tilt_deg": {"xlabel": "Wrist camera tilt (deg)", "title": "Camera rotation / tilt"},
    "cube_pos_cm":    {"xlabel": "Cube displacement from nominal (cm)", "title": "Object-position displacement"},
}


def fig_sensitivity_sweep(data, out_dir):
    baseline_key = "Baseline\n(all cams)"
    if baseline_key not in data:
        print("[skip] fig7_sensitivity_sweep: no baseline data to anchor delta=0")
        return

    axes_present = sorted({d["sweep"]["axis"] for d in data.values() if d.get("sweep")})
    if not axes_present:
        print("[skip] fig7_sensitivity_sweep: no conditions carry sweep metadata "
              "(need at least one CONDITION_SPECS entry with a 'sweep' dict)")
        return

    n_axes = len(axes_present)
    fig, axes = plt.subplots(1, n_axes, figsize=(5.5 * n_axes, 4.5), squeeze=False)
    axes = axes[0]

    base = data[baseline_key]

    for ax, axis_name in zip(axes, axes_present):
        meta = _SWEEP_AXIS_META.get(axis_name, {"xlabel": axis_name, "title": axis_name})

        points = [(0.0, base["tsr"], base["lo"], base["hi"])]
        for label, d in data.items():
            sw = d.get("sweep")
            if sw and sw["axis"] == axis_name:
                points.append((sw["value"], d["tsr"], d["lo"], d["hi"]))
        points.sort(key=lambda p: p[0])

        xs   = [p[0] for p in points]
        tsrs = [p[1] for p in points]
        los  = [p[1] - p[2] for p in points]
        his  = [p[3] - p[1] for p in points]

        ax.errorbar(xs, tsrs, yerr=[los, his], marker="o", markersize=7,
                    linewidth=1.8, capsize=4, color=C_BLUE, ecolor="#555",
                    elinewidth=1.2)
        ax.axvline(0, color=C_GRAY, ls="--", lw=1.0, alpha=0.7)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(meta["xlabel"])
        ax.set_ylabel("TSR")
        ax.set_title(meta["title"], fontweight="bold")
        for x, y in zip(xs, tsrs):
            ax.annotate(f"{y:.0%}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8.5, fontweight="bold")

        n_points = len(xs)
        if n_points < 3:
            ax.text(0.5, -0.22, f"[only {n_points} points incl. nominal — "
                     f"add more levels for a fuller curve]",
                     transform=ax.transAxes, ha="center", fontsize=7.5,
                     color=C_RED, style="italic")

    fig.suptitle("S1 — Sensitivity Curves: TSR as a Function of Perturbation Magnitude",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    _save(fig, "fig7_sensitivity_sweep", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 6 — 4-panel paper summary
# ══════════════════════════════════════════════════════════════════════════════

def fig_4panel(data, out_dir):
    cam_keys   = ["Baseline\n(all cams)", "Context\nonly", "Wrist\nonly", "Top\nonly"]
    two_keys   = ["Baseline\n(all cams)", "No top\n(ctx+wrist)", "No wrist\n(ctx+top)", "No context\n(wrist+top)"]
    sc_keys    = ["Baseline\n(all cams)", "Wrist shift\n+3 cm", "Two flanking\nspheres", "Color-matched\nsphere"]
    summary_keys = [
        "Baseline\n(all cams)", "Context\nonly", "Wrist\nonly", "Top\nonly",
        "No wrist\n(ctx+top)", "No top\n(ctx+wrist)", "No context\n(wrist+top)",
        "Absent top", "Absent\nwrist+top", "Wrist shift\n+3 cm",
        "Two flanking\nspheres", "Color-matched\nsphere",
    ]
    needed = set(cam_keys) | set(two_keys) | set(sc_keys) | set(summary_keys)
    if not _require(data, needed, "fig6_combined_4panel"):
        return

    fig = plt.figure(figsize=(16, 10))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.3)

    ax_a = fig.add_subplot(gs[0, 0])   # Camera importance (single)
    ax_b = fig.add_subplot(gs[0, 1])   # Two-cam combos
    ax_c = fig.add_subplot(gs[1, 0])   # Scene perturbations
    ax_d = fig.add_subplot(gs[1, 1])   # PSS heatmap strip

    # ── Panel A: single-camera TSR ────────────────────────────────────────────
    cam_labels = ["All 3", "Ctx only", "Wrist only", "Top only"]
    cam_colors = [C_GREEN, C_BLUE, C_AMBER, C_PURPLE]
    cam_tsrs   = [data[k]["tsr"] for k in cam_keys]
    cam_lo     = [data[k]["tsr"] - data[k]["lo"] for k in cam_keys]
    cam_hi     = [data[k]["hi"] - data[k]["tsr"] for k in cam_keys]
    x = np.arange(len(cam_keys))

    bars = ax_a.bar(x, cam_tsrs, color=cam_colors, alpha=0.88, width=0.6,
                    yerr=[cam_lo, cam_hi], capsize=5,
                    error_kw={"elinewidth": 1.3, "ecolor": "#555"})
    ax_a.set_ylim(0, 1.2)
    ax_a.set_title("(A) Single-Camera Ablation", fontweight="bold")
    ax_a.set_ylabel("TSR")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(cam_labels, fontsize=9)
    for bar, v in zip(bars, cam_tsrs):
        ax_a.text(bar.get_x() + bar.get_width() / 2, v + 0.06,
                  f"{v:.0%}", ha="center", fontsize=9, fontweight="bold", color=C_DARK)

    # ── Panel B: two-cam TSR ──────────────────────────────────────────────────
    two_labels = ["All 3", "Ctx+Wrist", "Ctx+Top", "Wrist+Top"]
    two_colors = [C_GREEN, C_TEAL, C_BLUE, C_AMBER]
    two_tsrs   = [data[k]["tsr"] for k in two_keys]
    two_lo     = [data[k]["tsr"] - data[k]["lo"] for k in two_keys]
    two_hi     = [data[k]["hi"] - data[k]["tsr"] for k in two_keys]

    bars = ax_b.bar(np.arange(len(two_keys)), two_tsrs, color=two_colors, alpha=0.88, width=0.6,
                    yerr=[two_lo, two_hi], capsize=5,
                    error_kw={"elinewidth": 1.3, "ecolor": "#555"})
    ax_b.set_ylim(0, 1.2)
    ax_b.set_title("(B) Two-Camera Combos (one zeroed)", fontweight="bold")
    ax_b.set_ylabel("TSR")
    ax_b.set_xticks(np.arange(len(two_keys)))
    ax_b.set_xticklabels(two_labels, fontsize=9)
    for bar, v in zip(bars, two_tsrs):
        ax_b.text(bar.get_x() + bar.get_width() / 2, v + 0.06,
                  f"{v:.0%}", ha="center", fontsize=9, fontweight="bold", color=C_DARK)

    # ── Panel C: scene perturbations ──────────────────────────────────────────
    sc_labels = ["Baseline", "Wrist +3 cm", "2 Spheres", "Color-match"]
    sc_colors = [C_GREEN, C_AMBER, C_RED, C_PURPLE]
    sc_tsrs   = [data[k]["tsr"] for k in sc_keys]
    sc_lo     = [data[k]["tsr"] - data[k]["lo"] for k in sc_keys]
    sc_hi     = [data[k]["hi"] - data[k]["tsr"] for k in sc_keys]

    bars = ax_c.bar(np.arange(len(sc_keys)), sc_tsrs, color=sc_colors, alpha=0.88, width=0.55,
                    yerr=[sc_lo, sc_hi], capsize=5,
                    error_kw={"elinewidth": 1.3, "ecolor": "#555"})
    ax_c.set_ylim(0, 1.2)
    ax_c.set_title("(C) Scene Perturbations", fontweight="bold")
    ax_c.set_ylabel("TSR")
    ax_c.set_xticks(np.arange(len(sc_keys)))
    ax_c.set_xticklabels(sc_labels, fontsize=9)
    for bar, v in zip(bars, sc_tsrs):
        ax_c.text(bar.get_x() + bar.get_width() / 2, v + 0.06,
                  f"{v:.0%}", ha="center", fontsize=9, fontweight="bold", color=C_DARK)

    # ── Panel D: summary heatmap (TSR + PSS strip) ────────────────────────────
    summary_labels = [
        "Baseline", "Ctx only", "Wrist only", "Top only",
        "No wrist", "No top", "No context", "Absent top", "Absent W+T",
        "Wrist +3cm", "2 Spheres", "Color-match",
    ]
    tsr_vals = np.array([data[k]["tsr"] for k in summary_keys])
    pss_vals = np.array([data[k]["pss"] / 1.6 for k in summary_keys])  # normalise PSS to [0,1]

    heatmap = np.vstack([tsr_vals, pss_vals])
    im = ax_d.imshow(heatmap, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                     interpolation="nearest")
    ax_d.set_yticks([0, 1])
    ax_d.set_yticklabels(["TSR", "PSS\n(norm.)"], fontsize=9)
    ax_d.set_xticks(np.arange(len(summary_labels)))
    ax_d.set_xticklabels(summary_labels, fontsize=7.5, rotation=40, ha="right")
    ax_d.set_title("(D) Summary Heatmap (all conditions)", fontweight="bold")

    for j in range(heatmap.shape[1]):
        for i in range(heatmap.shape[0]):
            v = heatmap[i, j]
            ax_d.text(j, i, f"{v:.2f}", ha="center", va="center",
                      fontsize=7, color="black" if 0.3 < v < 0.8 else "white",
                      fontweight="bold")

    plt.colorbar(im, ax=ax_d, shrink=0.7, pad=0.02)

    ns = sorted({data[k]["n"] for k in summary_keys})
    n_label = f"N={ns[0]}" if len(ns) == 1 else f"N varies ({ns[0]}–{ns[-1]}) — see summary table"
    fig.suptitle(
        f"SmolVLA SO-101 — Perturbation Robustness Study  ({n_label})",
        fontsize=14, fontweight="bold", y=1.01,
    )
    _save(fig, "fig6_combined_4panel", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  LaTeX summary table
# ══════════════════════════════════════════════════════════════════════════════

def write_latex_table(data, out_dir):
    groups = {
        "Baseline":             ["Baseline\n(all cams)"],
        "Single-camera":        ["Context\nonly", "Wrist\nonly", "Top\nonly"],
        "Two-camera combos":    ["No top\n(ctx+wrist)", "No wrist\n(ctx+top)", "No context\n(wrist+top)"],
        "Absent inputs":        ["Absent top", "Absent\nwrist+top"],
        "Scene perturbations":  ["Wrist shift\n+3 cm", "Two flanking\nspheres", "Color-matched\nsphere"],
    }

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{SmolVLA robustness evaluation (95\\% Wilson CI; N per condition given in the N column).}",
        "\\label{tab:robustness}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Group & Condition & N & TSR (\\%) & Mean PSS \\\\",
        "\\midrule",
    ]

    any_missing = False
    for group, keys in groups.items():
        wrote_group_row = False
        for key in keys:
            if key not in data:
                any_missing = True
                continue
            d    = data[key]
            name = key.replace("\n", " ")
            lo   = d["lo"] * 100
            hi   = d["hi"] * 100
            tsr  = d["tsr"] * 100
            pss  = d["pss"]
            ci   = f"[{lo:.1f}, {hi:.1f}]"
            grp  = group if not wrote_group_row else ""
            wrote_group_row = True
            lines.append(f"{grp} & {name} & {d['n']} & ${tsr:.0f}$ {ci} & ${pss:.3f}$ \\\\")
        if wrote_group_row:
            lines.append("\\midrule")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]

    text = "\n".join(lines)
    path = out_dir / "summary_table.txt"
    path.write_text(text)
    print(f"[ok] summary_table.txt" + ("  (some conditions omitted — missing CSVs)" if any_missing else ""))

    # Also write a human-readable version
    hr = ["=" * 80,
          f"SmolVLA Perturbation Robustness — computed directly from CSVs",
          "=" * 80,
          f"{'Condition':<35}  {'N':>5}  {'TSR':>6}  {'95% CI':>15}  {'PSS':>6}",
          "-" * 80]
    for label, d in data.items():
        name = label.replace("\n", " ")
        ci   = f"[{d['lo']:.3f},{d['hi']:.3f}]"
        hr.append(f"{name:<35}  {d['n']:>5}  {d['tsr']:>6.1%}  {ci:>15}  {d['pss']:>6.3f}")
    hr.append("=" * 80)
    (out_dir / "summary_readable.txt").write_text("\n".join(hr))
    print("[ok] summary_readable.txt")
    print("\n" + "\n".join(hr))


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", required=True,
                        help="Root directory holding one subfolder per condition "
                             "slug, e.g. runs/icra-final/eval/ (the --run_root "
                             "eval output of zeroshot_eval.py). No default: this "
                             "must point at a real campaign root, since the old "
                             "hardcoded results/ layout no longer applies.")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory for figures (default: <results_root>/figures)")
    args = parser.parse_args()

    root    = Path(args.results_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = build_conditions(root)

    print(f"\nLoading conditions from {root}...\n")
    data = load_all_conditions(conditions)

    if not data:
        print("[fatal] No conditions loaded — nothing to plot. "
              "Run zeroshot_eval.py --run_root <tag> for at least the "
              "baseline condition first.")
        return

    print(f"\nGenerating figures → {out_dir}/\n")
    fig_overview(data, out_dir)
    fig_single_cam(data, out_dir)
    fig_two_cam(data, out_dir)
    fig_absent(data, out_dir)
    fig_scene(data, out_dir)
    fig_sensitivity_sweep(data, out_dir)
    fig_4panel(data, out_dir)
    write_latex_table(data, out_dir)

    print(f"\n✓  Figures written to {out_dir}/ for {len(data)}/{len(conditions)} conditions.")
    if len(data) < len(conditions):
        print("  Some figures were skipped because not every condition has data yet — "
              "see [error]/[skip] messages above.")


if __name__ == "__main__":
    main()