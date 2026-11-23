"""
Builds a small TOY campaign with the stubbed evaluator (3 horizons x 14
conditions x 40 episodes) and runs analyze_paper.py on it. Exercises every
table and figure path; the numbers mean nothing about SmolVLA.

    python tests/test_analysis.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import stub_env  # noqa: E402
stub_env.install_fake_genesis(); stub_env.install_fake_lerobot()
import zeroshot_eval as ze  # noqa: E402
import analyze_paper  # noqa: E402

_w = {}
stub_env.FakePolicy.world_box = _w
ze.build_scene = stub_env.make_build_scene(_w)

CONDS = {
    "nominal": [], "wtilt+2.5": ["--tilt_target", "wrist", "--tilt_deg", "2.5"],
    "wtilt+5": ["--tilt_target", "wrist", "--tilt_deg", "5"],
    "wtilt+10": ["--tilt_target", "wrist", "--tilt_deg", "10"],
    "wtilt-5": ["--tilt_target", "wrist", "--tilt_deg", "-5"],
    "wpos+1.5cm": ["--pos_target", "wrist", "--pos_offset", "0.015", "0", "0"],
    "no-top": ["--blackout_mode", "no_top"], "no-wrist": ["--blackout_mode", "no_wrist"],
    "no-ctx": ["--blackout_mode", "no_context"], "ctx-only": ["--blackout_mode", "context_only"],
    "wrist-only": ["--blackout_mode", "wrist_only"], "top-only": ["--blackout_mode", "top_only"],
    "all-black": ["--blackout_mode", "all_black"], "abs-top": ["--blackout_mode", "absent_top"],
}


def main():
    tmp = Path(tempfile.mkdtemp(prefix="toy_campaign_"))
    os.chdir(tmp)
    import generate_matched_eval_set as g
    pairs = g.generate(40, 1234)
    Path("matched.json").write_text(json.dumps({"n": 40, "cube_target_pairs": pairs, "content_hash": "toy"}))
    for k in ("50", "25", "10"):
        for slug, extra in CONDS.items():
            probe = ["--probe_nominal"] if slug in ("wtilt+5", "wtilt+10") else []
            with redirect_stdout(StringIO()):
                ze.main(["--run_root", "toy", "--checkpoint", "fake", "--matched_eval_set", "matched.json",
                         "--n", "40", "--n_action_steps", k, *extra, *probe])
        print(f"[toy] k={k}: {len(CONDS)} conditions done")
    analyze_paper.main(["--run_root", "runs/toy"])
    out = tmp / "runs" / "toy" / "analysis"
    figs = sorted(p.name for p in (out / "figures").glob("*.png"))
    print(f"[ok] {len(figs)} figures: {figs}")
    for t in ("table_conditions.csv", "table_ol_vs_cl.csv", "table_camera_shapley.csv",
              "table_conditions.tex", "provenance_report.txt"):
        assert (out / t).exists(), t
    # provenance guard: tamper with one summary -> analysis must refuse
    sp = tmp / "runs/toy/eval/k10/no-top/run_summary.json"
    s = json.loads(sp.read_text()); s["outcome_code_hash"] = "different"; sp.write_text(json.dumps(s))
    try:
        with redirect_stdout(StringIO()):
            analyze_paper.main(["--run_root", "runs/toy", "--no_figures"])
        raise AssertionError("provenance mismatch not detected")
    except SystemExit:
        print("[ok] provenance mismatch is refused")
    print(f"ALL ANALYSIS TESTS PASSED ({out})")
    return out


if __name__ == "__main__":
    main()