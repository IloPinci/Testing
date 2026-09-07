"""
run_naming.py
--------------
Central naming policy so every dataset, checkpoint, and eval run lands
under its own clearly-labeled folder, and NOTHING from a previous run
(e.g. your existing demos/, outputs/, results/ trees) can ever be
silently overwritten.

Directory layout under one run root:

    runs/<tag>/
        data/<dataset_slug>/lerobot/     # two_camera_setup.py / oracle_direct.py output
        train/<checkpoint_id>/           # finetune.py output
        matched_eval_set.json            # generate_matched_eval_set.py output
        eval/<condition_slug>/
            zeroshot.csv
            run_summary.json
            videos/policy/...
            videos/witness/...

Every script that writes data now accepts --run_root <tag>. When set, it
derives all output paths from this module instead of a hardcoded default,
and refuses to overwrite anything that already exists unless --overwrite
is also passed. An explicit --out / --out_dir / --output_dir still wins
over the derived path, so nothing here breaks a fully manual workflow.

Slugs are intentionally short (a few words, hyphenated) but derived
directly from the actual perturbation parameters, not a hand-maintained
label — so a new perturbation combination always gets SOME reasonable
slug instead of crashing, even if it's not as pretty as a hand-tuned one.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

# ── Short human-readable tags for known perturbation values ────────────────
# Anything not listed here falls back to _slugify() of the raw label, so
# adding a new BLACKOUT_MODE / sphere preset / cube position never breaks
# this file — it just produces a slightly uglier (but still short) slug
# until someone adds a nicer entry below.

_BLACKOUT_SLUGS = {
    "none": "nominal", "3-cam": "nominal",
    "context_only": "ctx-only", "wrist_only": "wrist-only", "top_only": "top-only",
    "no_top": "no-top", "no_wrist": "no-wrist", "no_context": "no-ctx",
    "absent_top": "abs-top", "absent_context": "abs-ctx", "absent_wrist": "abs-wrist",
    "absent_all": "abs-all",
    "absent_wrist_top": "abs-w+t", "absent_context_top": "abs-c+t",
    "absent_wrist_context": "abs-w+c",
    "3rd_black": "ctx-black", "all_black": "all-black",
}

_CUBE_POS_SLUGS = {
    "near_left": "cpos-nearL", "far_left": "cpos-farL",
    "near_right": "cpos-nearR", "far_right": "cpos-farR",
    "workspace_edge": "cpos-edge", "target_shift": "tpos-shift",
    "close_to_target": "cpos-closeT",
    # "nominal" deliberately omitted — contributes nothing to the slug
}

_SPHERE_SLUGS = {
    "single_left": "sph-1L", "single_right": "sph-1R",
    "two_flanking": "sph-2flank", "high_clutter": "sph-clutter3",
    "near_target": "sph-nearT", "color_match": "sph-colormatch",
}
# SPHERE_PRESETS produce SphereConfig objects whose own .label is per-sphere
# ("flank_left", "flank_right", ...), not the preset name ("two_flanking").
# This maps the exact combination of per-sphere labels a preset produces
# back to the preset's short slug above, so scene_perturbation_slug() can
# recognize "this scene's spheres ARE the two_flanking preset" even though
# it only ever sees the individual SphereConfig objects, not the preset name.
_SPHERE_LABEL_COMBO_TO_PRESET = {
    ("distractor_left",):                      "single_left",
    ("distractor_right",):                     "single_right",
    ("flank_left", "flank_right"):             "two_flanking",
    ("clutter_a", "clutter_b", "clutter_c"):   "high_clutter",
    ("near_target",):                          "near_target",
    ("red_match",):                            "color_match",
}


def _slugify(text: str, max_len: int = 24) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text[:max_len] or "x"


def camera_perturbation_slug(perturbation) -> str:
    """Short slug for a CameraPerturbation: blackout mode + position/tilt offsets."""
    parts = [_BLACKOUT_SLUGS.get(perturbation.blackout_mode,
                                  _slugify(perturbation.blackout_mode))]

    if perturbation.position_target is not None:
        dx, dy, dz = perturbation.position_offset
        mag  = max(abs(dx), abs(dy), abs(dz))
        sign = "-" if (dx + dy + dz) < 0 else "+"
        parts.append(f"{perturbation.position_target[:1]}pos{sign}{mag*100:.1f}cm")

    if perturbation.tilt_target is not None:
        sign = "+" if perturbation.tilt_deg >= 0 else ""
        parts.append(f"{perturbation.tilt_target[:1]}tilt{sign}{perturbation.tilt_deg:.0f}")

    return "_".join(p for p in parts if p and p != "nominal") or "nominal"


def scene_perturbation_slug(scene_perturbation) -> str:
    """Short slug for a ScenePerturbation: cube color + position preset + spheres."""
    parts = []
    if scene_perturbation.cube_color != "red":
        parts.append(f"cube-{scene_perturbation.cube_color}")

    preset = scene_perturbation.cube_position_preset
    if preset is not None and preset != "nominal":
        parts.append(_CUBE_POS_SLUGS.get(preset, _slugify(f"cpos-{preset}")))

    if scene_perturbation.spheres:
        label_combo = tuple(s.label for s in scene_perturbation.spheres)
        matched_preset = _SPHERE_LABEL_COMBO_TO_PRESET.get(label_combo)
        if matched_preset is not None:
            parts.append(_SPHERE_SLUGS[matched_preset])
        else:
            # Unrecognized / custom sphere combination — still short, just
            # less pretty: sph-<n> plus a slug of the first label as a hint.
            n = len(scene_perturbation.spheres)
            hint = _slugify(scene_perturbation.spheres[0].label, max_len=10)
            parts.append(f"sph-{n}custom-{hint}")

    return "_".join(parts)


def condition_slug(camera_perturbation, scene_perturbation) -> str:
    """
    Full short slug for one eval condition, combining both perturbation
    axes, e.g. 'wrist-only', 'wpos+3.0cm', 'nominal_sph-colormatch'.
    """
    cam = camera_perturbation_slug(camera_perturbation)
    scn = scene_perturbation_slug(scene_perturbation)
    combined = "_".join(p for p in [cam, scn] if p and p != "nominal")
    return combined or "nominal"


def run_root(tag: str, base: str = "runs") -> Path:
    """
    One root directory per experiment campaign, e.g. runs/icra-final/.

    IMPORTANT: this does NOT include today's date. An earlier version of
    this function prefixed the path with date.today(), which meant the
    exact same --run_root tag resolved to a DIFFERENT path depending on
    what day you ran it — silently breaking every cross-day comparison
    in a multi-day campaign (dataset generated on day 1, evaluated on
    day 3, analyzed on day 5 would each compute a different root). The
    tag alone now determines the path, so it is stable for the life of
    the campaign no matter how many days it spans.

    If you want a date recorded somewhere, put it in the tag yourself
    (e.g. --run_root icra-final-2026-09-05) — that's a deliberate choice
    you make once, not something recomputed on every invocation.

    Nothing under here can collide with your pre-existing demos/,
    outputs/, or results/ trees from before this naming convention
    existed.
    """
    safe_tag = _slugify(tag, max_len=60)
    root = Path(base) / safe_tag
    # One-time, purely informational marker — never used for addressing,
    # so it can never cause a path mismatch. Just a human-readable record
    # of when this campaign's root was first touched.
    marker = root / ".created_at"
    if not marker.exists():
        root.mkdir(parents=True, exist_ok=True)
        marker.write_text(date.today().isoformat() + "\n")
    return root


def eval_paths(root: Path, slug: str) -> dict:
    cond_dir = root / "eval" / slug
    return {
        "dir":                cond_dir,
        "csv":                cond_dir / "zeroshot.csv",
        "policy_video_dir":   cond_dir / "videos" / "policy",
        "witness_video_dir":  cond_dir / "videos" / "witness",
    }


def dataset_dir(root: Path, dataset_slug: str) -> Path:
    return root / "data" / dataset_slug / "lerobot"


def train_dir(root: Path, checkpoint_id: str) -> Path:
    return root / "train" / checkpoint_id


def matched_eval_set_path(root: Path) -> Path:
    return root / "matched_eval_set.json"


def refuse_if_exists(path: Path, what: str, overwrite: bool):
    """
    Common overwrite guard used by every script in this project. Checks
    both a file path and a directory path (existence + non-empty for
    directories, so an empty pre-created folder doesn't trigger a false
    refusal).
    """
    exists = path.is_file() or (path.is_dir() and any(path.iterdir()))
    if exists and not overwrite:
        raise SystemExit(
            f"[error] {what} already exists at {path} and is non-empty. "
            f"Refusing to overwrite existing data. Pass --overwrite if you "
            f"really mean to replace it, or choose a different --run_root "
            f"/ output path so old data is never silently lost."
        )