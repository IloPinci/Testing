"""
test_static.py -- fast pre-flight checks (protocol v2, section 3). Run after
every code change and before every campaign:

    python icra/tests/test_static.py            # from the folder containing icra/
    python icra/tests/test_static.py --require-real   # fail (not skip) if lerobot/genesis missing

No GPU and no simulation: genesis/lerobot are stubbed for the in-process
checks, so the only real-package check (section G) runs in a subprocess.

What each section protects against
  A  compile          a syntax error that would crash every run of a sweep
  B  sweep <-> CLI    a typo in run_sweep.sh that argparse rejects (every run
                      of that condition FAILS), or two conditions resolving to
                      the SAME output folder (the second is silently logged
                      ALREADY_DONE and never runs), or a label that differs from
                      the folder analyze_paper.py / make_comparison_video.py use
  C  provenance       outcome-relevant files missing from the code hash, or a
                      hash that is not stable (analyze_paper.py would refuse runs)
  D  matched set      matched eval set not deterministic for a given seed
  E  statistics       stats functions disagree with hand-computed reference values
  F  metrics/config   frozen thresholds and index parsing changed by accident
  G  real packages    installed lerobot/genesis lack the calls zeroshot_eval uses
Exit code 0 = all passed (skips allowed unless --require-real), 1 = a failure.
"""
from __future__ import annotations

import argparse
import py_compile
import re
import shlex
import subprocess
import sys
import time
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
ICRA = HERE.parent
sys.path.insert(0, str(ICRA)); sys.path.insert(0, str(HERE))

RESULTS: list[tuple[str, str, str]] = []   # (status, name, detail)


def check(name: str, ok: bool, detail: str = ""):
    RESULTS.append(("PASS" if ok else "FAIL", name, detail))


def skip(name: str, detail: str):
    RESULTS.append(("SKIP", name, detail))


def close(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(a - b) <= tol


# A. compile 
def section_compile():
    bad = []
    for f in sorted(list(ICRA.glob("*.py")) + list(HERE.glob("*.py"))):
        try:
            py_compile.compile(str(f), doraise=True)
        except py_compile.PyCompileError as e:
            bad.append(f"{f.name}: {e.msg.strip().splitlines()[-1]}")
    check("A1 all .py files compile", not bad, "; ".join(bad))
    r = subprocess.run(["bash", "-n", str(ICRA / "run_sweep.sh")], capture_output=True, text=True)
    check("A2 run_sweep.sh bash syntax", r.returncode == 0, r.stderr.strip())


# B. run_sweep.sh conditions vs zeroshot_eval CLI and naming 
def parse_sweep_sets(text: str) -> dict[str, list[tuple[str, str]]]:
    sets = {}
    for name in ("CORE", "DOSE", "CONTROLS"):
        m = re.search(rf"^{name}=\((.*?)^\)", text, re.S | re.M)
        if not m:
            continue
        entries = re.findall(r'^\s*"([^"|]+)\|([^"]*)"', m.group(1), re.M)
        sets[name] = [(lbl.strip(), args.strip()) for lbl, args in entries]
    return sets


def slug_for(ze, args) -> str:
    # Mirrors the CameraPerturbation / ScenePerturbation construction in
    # zeroshot_eval.main(). If main() changes, update this together with it.
    from run_naming import condition_slug
    cam = ze.CameraPerturbation(
        blackout_mode=args.blackout_mode, position_target=args.pos_target,
        position_offset=tuple(args.pos_offset), tilt_target=args.tilt_target,
        tilt_deg=args.tilt_deg, blank_fill=args.blank_fill)
    scn = ze.ScenePerturbation(
        cube_color=args.cube_color, cube_position_preset=args.cube_pos,
        spheres=ze.SPHERE_PRESETS[args.spheres])
    return condition_slug(cam, scn)


def section_sweep(ze):
    import contextlib, io
    import analyze_paper
    text = (ICRA / "run_sweep.sh").read_text()
    sets = parse_sweep_sets(text)
    check("B1 CORE/DOSE/CONTROLS arrays found", set(sets) == {"CORE", "DOSE", "CONTROLS"},
          f"found {sorted(sets)}")
    counts = {k: len(v) for k, v in sets.items()}
    check("B2 condition counts (core 21, dose 19, controls 6)",
          counts == {"CORE": 21, "DOSE": 19, "CONTROLS": 6}, str(counts))

    common = ["--run_root", "static-check", "--checkpoint", "dummy", "--n", "200",
              "--matched_eval_set", "dummy.json", "--n_action_steps", "10"]
    forbidden = ("--n_action_steps", "--run_root", "--n ", "--out", "--overwrite", "--episodes")
    parse_err, label_mismatch, other_family, forbidden_hit = [], [], [], []
    slug_owner: dict[str, str] = {}
    collisions = []
    for set_name, entries in sets.items():
        for label, extra in entries:
            if any(f in extra + " " for f in forbidden):
                forbidden_hit.append(label)
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    a = ze.parse_args(common + shlex.split(extra))
            except SystemExit:
                parse_err.append(f"{label} ({extra})")
                continue
            slug = slug_for(ze, a)
            if slug != label:
                label_mismatch.append(f"{label} -> folder '{slug}'")
            if slug in slug_owner:
                collisions.append(f"{slug_owner[slug]} and {set_name}:{label} -> {slug}")
            slug_owner[slug] = f"{set_name}:{label}"
            if analyze_paper.family(slug) == "other":
                other_family.append(slug)
    check("B3 every condition is accepted by zeroshot_eval's argparse", not parse_err,
          "; ".join(parse_err))
    check("B4 no output-folder collisions across all sets", not collisions, "; ".join(collisions))
    check("B5 sweep label == output folder slug", not label_mismatch, "; ".join(label_mismatch))
    check("B6 every slug has an analysis family (not 'other')", not other_family,
          ", ".join(other_family))
    check("B7 no per-run flags inside condition args", not forbidden_hit, ", ".join(forbidden_hit))
    core_slugs = {l for l, _ in sets.get("CORE", [])}
    missing = sorted(set(analyze_paper.CAMERA_SUBSETS) - core_slugs)
    check("B8 CORE contains the full 2^3 camera factorial (Shapley)", not missing,
          f"missing {missing}")
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            ze.parse_args(common[:-2])
        check("B9 --n_action_steps is required (no silent default)", False, "parsed without it")
    except SystemExit:
        check("B9 --n_action_steps is required (no silent default)", True)


# C. provenance 
def section_provenance(ze):
    missing = [f for f in ze.OUTCOME_SOURCE_FILES if not (ICRA / f).exists()]
    check("C1 all OUTCOME_SOURCE_FILES exist", not missing, f"missing {missing}")
    a = ze.parse_args(["--out", "x.csv", "--n_action_steps", "10", "--checkpoint", "dummy"])
    p1, p2 = ze.collect_provenance(a), ze.collect_provenance(a)
    check("C2 outcome_code_hash is stable", p1["outcome_code_hash"] == p2["outcome_code_hash"],
          f"{p1['outcome_code_hash']} vs {p2['outcome_code_hash']}")
    commit, dirty = p1.get("git_commit"), p1.get("git_dirty")
    # Not a failure (tests must pass on a dev checkout), but must be clean for the campaign.
    RESULTS.append(("INFO", "C3 git state seen by run_summary.json",
                    f"commit={commit or 'NONE (icra/ is not a git checkout)'} dirty={dirty} "
                    f"code_hash={p1['outcome_code_hash']}"))


# D. matched eval set 
def section_matched():
    import generate_matched_eval_set as g
    a, b = g.generate(50, 1234), g.generate(50, 1234)
    check("D1 matched set deterministic for a seed", a == b)
    check("D2 different seed -> different set", g.generate(50, 999) != a)
    check("D3 prefix-stable (n=20 is the first 20 of n=50)", g.generate(20, 1234) == a[:20])


# E. statistics vs hand-computed references 
def section_stats():
    import numpy as np
    import stats_analysis as st

    lo, hi = st.wilson_ci(91, 100)
    check("E1 Wilson 91/100 = (0.8377, 0.9519)", close(lo, 0.83774) and close(hi, 0.95193),
          f"got ({lo:.5f}, {hi:.5f})")
    lo, hi = st.wilson_ci(0, 10)
    check("E2 Wilson 0/10 = (0, 0.2775)", close(lo, 0.0) and close(hi, 0.27753),
          f"got ({lo:.5f}, {hi:.5f})")

    def pairs(n11, n10, n01, n00):
        a = [1] * n11 + [1] * n10 + [0] * n01 + [0] * n00
        b = [1] * n11 + [0] * n10 + [1] * n01 + [0] * n00
        return a, b
    cases = [((50, 0, 6, 44), 2 / 64), ((50, 1, 1, 48), 1.0),
             ((40, 2, 8, 50), 2 * (comb(10, 0) + comb(10, 1) + comb(10, 2)) / 2 ** 10),
             ((90, 0, 0, 10), 1.0)]
    bad = []
    for (n11, n10, n01, n00), ref in cases:
        r = st.mcnemar_exact(*pairs(n11, n10, n01, n00))
        if not close(r["p"], ref, 1e-12) or r["n01"] != n10 or r["n10"] != n01:
            bad.append(f"{(n10, n01)}: p={r['p']} ref={ref} counts={r['n01']},{r['n10']}")
    check("E3 exact McNemar p-values and discordant counts", not bad, "; ".join(bad))

    adj = st.holm([0.01, 0.04, 0.03, 0.005])
    check("E4 Holm [.01,.04,.03,.005] -> [.03,.06,.06,.02]",
          all(close(x, y, 1e-12) for x, y in zip(adj, [0.03, 0.06, 0.06, 0.02])), str(adj))

    x = np.array([1, 0, 1, 1, 0, 1, 0, 1] * 10)
    check("E5 paired delta of identical arrays = (0, 0, 0)",
          st.delta_ci_paired(x, x) == (0.0, 0.0, 0.0))
    nom_a, pert_a = np.ones(40), np.ones(40)               # open loop: no drop
    nom_b, pert_b = np.ones(40), np.r_[np.zeros(10), np.ones(30)]   # closed loop: -25 pts
    d, dlo, dhi = st.did_paired(nom_a, pert_a, nom_b, pert_b)
    check("E6 DiD sign and value (-0.25, CI below 0)", close(d, -0.25) and dhi < 0,
          f"got {d:.3f} [{dlo:.3f}, {dhi:.3f}]")
    pt, clo, chi = st.cluster_bootstrap_mean([np.ones(10), np.zeros(10), np.ones(10)], n_boot=2000)
    check("E7 cluster bootstrap point = mean of seed means", close(pt, 2 / 3) and clo <= pt <= chi,
          f"got {pt:.3f} [{clo:.3f}, {chi:.3f}]")
    t0 = time.time()
    p_null = st.power_mcnemar(0.9, 0.9, 100, rho=0.6, alpha=0.05, n_sim=1500)
    p_big = st.power_mcnemar(0.9, 0.5, 100, rho=0.6, alpha=0.05, n_sim=300)
    check("E8 power: type-I <= alpha under H0, ~1 for a 40-pt drop", p_null <= 0.06 and p_big > 0.99,
          f"H0 rejection rate {p_null:.3f}, power {p_big:.3f} ({time.time() - t0:.1f}s)")


# ── F. frozen thresholds and index parsing ───────────────────────────────────
def section_metrics():
    import numpy as np
    import episode_metrics as em
    c = em.MetricConfig()
    frozen = {"reach_radius": 0.04, "gripper_closed_max": 0.49, "gripper_open_min": 0.49,
              "lift_min": 0.01, "target_radius": 0.05, "rest_max_height": 0.025,
              "rest_max_speed": 0.05, "policy_dt": 0.1, "success_hold_steps": 3,
              "legacy_near_radius": 0.12}
    diff = {k: (getattr(c, k), v) for k, v in frozen.items() if getattr(c, k) != v}
    # If you deliberately change a threshold (e.g. calibrated reach_radius as the
    # new default), update this table in the same commit and note it in the protocol.
    check("F1 MetricConfig defaults match the frozen protocol values", not diff, str(diff))
    check("F2 grasp point offset = (0.016, 0, 0.07)",
          np.allclose(em.GRASP_POINT_OFFSET, [0.016, 0.0, 0.07]))
    p = em.parse_index_spec
    ok = (p("0-4", 200) == [0, 1, 2, 3, 4] and p("none", 200) == [] and p("all", 5) == list(range(5))
          and p("3,17,40-42", 200) == [3, 17, 40, 41, 42])
    check("F3 parse_index_spec ('0-4', 'none', 'all', '3,17,40-42')", ok)


# G. real installed packages (subprocess, no stubs) 
REAL_PROBE = r"""
import inspect, json
out = {}
try:
    import genesis, importlib.metadata as md
    out["genesis"] = md.version("genesis-world")
except Exception as e:
    out["genesis_error"] = repr(e)
try:
    import lerobot, importlib.metadata as md
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
    from lerobot.policies.factory import make_pre_post_processors
    out["lerobot"] = md.version("lerobot"); out["lerobot_file"] = lerobot.__file__
    out["has_predict_action_chunk"] = hasattr(SmolVLAPolicy, "predict_action_chunk")
    out["predict_takes_noise"] = (out["has_predict_action_chunk"] and
        "noise" in inspect.signature(SmolVLAPolicy.predict_action_chunk).parameters)
    out["has_select_action"] = hasattr(SmolVLAPolicy, "select_action")
    out["has_sample_noise"] = hasattr(VLAFlowMatching, "sample_noise")
except Exception as e:
    out["lerobot_error"] = repr(e)
print("JSON" + json.dumps(out))
"""


def section_real(require: bool):
    import json
    r = subprocess.run([sys.executable, "-c", REAL_PROBE], capture_output=True, text=True,
                       timeout=300, cwd=str(ICRA.parent))
    line = next((l for l in r.stdout.splitlines() if l.startswith("JSON")), None)
    info = json.loads(line[4:]) if line else {"probe_error": r.stderr.strip()[-300:]}
    report = skip if not require else (lambda n, d: check(n, False, d))
    if "genesis" in info:
        check("G1 genesis importable", True, f"genesis-world {info['genesis']}")
    else:
        report("G1 genesis importable", info.get("genesis_error", info.get("probe_error", "")))
    if "lerobot" in info:
        check("G2 lerobot SmolVLA importable", True,
              f"lerobot {info['lerobot']} from {info['lerobot_file']}")
        check("G3 SmolVLAPolicy.predict_action_chunk(batch, noise=...) exists (chunk backend)",
              bool(info["predict_takes_noise"]))
        check("G4 SmolVLAPolicy.select_action exists (select_action backend)",
              bool(info["has_select_action"]))
        check("G5 VLAFlowMatching.sample_noise exists (shared noise stream)",
              bool(info["has_sample_noise"]))
    else:
        report("G2 lerobot SmolVLA importable", info.get("lerobot_error", info.get("probe_error", "")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--require-real", action="store_true",
                    help="Treat missing genesis/lerobot as a failure (use inside the campaign container).")
    ap.add_argument("--skip-real", action="store_true", help="Skip section G.")
    args = ap.parse_args()
    t0 = time.time()

    section_compile()
    if not args.skip_real:
        section_real(args.require_real)        # before stubs touch sys.modules

    import stub_env
    stub_env.install_fake_genesis(); stub_env.install_fake_lerobot()
    import zeroshot_eval as ze
    for sec in (lambda: section_sweep(ze), lambda: section_provenance(ze),
                section_matched, section_stats, section_metrics):
        try:
            sec()
        except Exception as e:                  # a crashing section is a failure, not a traceback
            check(f"section crashed: {getattr(sec, '__name__', 'section')}", False, repr(e))

    width = max(len(n) for _, n, _ in RESULTS)
    for status, name, detail in RESULTS:
        print(f"[{status}] {name:<{width}}  {detail}".rstrip())
    n_fail = sum(s == "FAIL" for s, _, _ in RESULTS)
    n_skip = sum(s == "SKIP" for s, _, _ in RESULTS)
    print(f"\n{len(RESULTS)} checks, {n_fail} failed, {n_skip} skipped, {time.time() - t0:.1f}s")
    print("STATIC CHECKS PASSED" if n_fail == 0 else "STATIC CHECKS FAILED")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()