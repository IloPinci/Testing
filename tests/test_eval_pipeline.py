"""
End-to-end test of zeroshot_eval.py's bookkeeping on a toy CPU world.

    python tests/test_eval_pipeline.py        # or: pytest tests/

Checks:
  1. chunk backend == select_action backend (identical CSV outcomes),
     for open loop (k=50) and closed loop (k=10)        -> the speed-up is exact
  2. the chunk backend renders far fewer training-camera frames
  3. recording videos does not change outcomes; videos + manifest are written
  4. --episodes replay reproduces the full run (with --reference_csv)
  5. --probe_nominal does not change outcomes and logs a divergence
  6. outputs: CSV columns, run_summary provenance, trajectories.npz
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import stub_env  # noqa: E402

stub_env.install_fake_genesis()
stub_env.install_fake_lerobot()
import zeroshot_eval as ze  # noqa: E402

_world = {}
stub_env.FakePolicy.world_box = _world
ze.build_scene = stub_env.make_build_scene(_world)

TMP = Path(tempfile.mkdtemp(prefix="ze_test_"))
MATCH = TMP / "matched.json"


def _make_matched(n=12):
    import generate_matched_eval_set as g
    pairs = g.generate(n, 1234)
    MATCH.write_text(json.dumps({"n": n, "cube_target_pairs": pairs, "content_hash": "test"}))


def run(name, *extra):
    out = TMP / name / "zeroshot.csv"
    stub_env.RENDER_COUNT["n"] = 0
    ze.main(["--out", str(out), "--checkpoint", "fake", "--matched_eval_set", str(MATCH),
             "--n", "12", *extra])
    rows = list(csv.DictReader(open(out)))
    return rows, json.loads((out.parent / "run_summary.json").read_text()), stub_env.RENDER_COUNT["n"]


def outcomes(rows):
    return [(r["episode"], r["success_strict"], r["success_carry"], r["failure_mode"], r["steps"])
            for r in rows]


def test_all():
    _make_matched()
    results = {}
    for k in ("50", "10"):
        rc, sc, nc = run(f"chunk_k{k}", "--n_action_steps", k)
        rs, ss, ns = run(f"sel_k{k}", "--n_action_steps", k, "--backend", "select_action")
        assert outcomes(rc) == outcomes(rs), f"backend mismatch at k={k}"
        assert nc < ns, (nc, ns)
        results[k] = (rc, sc, nc, ns)
        print(f"[ok] k={k}: chunk == select_action on {len(rc)} episodes; "
              f"training renders {nc} vs {ns}; TSR strict {sc['tsr_strict']:.2f}")

    # perturbed condition actually changes behaviour in the toy world
    rp, sp, _ = run("tilt_k10", "--n_action_steps", "10", "--tilt_target", "wrist", "--tilt_deg", "10")
    print(f"[ok] perturbed run: TSR strict {sp['tsr_strict']:.2f}, modes {sp['failure_modes']}")

    # videos do not change outcomes
    rv, sv, _ = run("video_k10", "--n_action_steps", "10", "--video_episodes", "0-2")
    assert outcomes(rv) == outcomes(results["10"][0]), "recording changed outcomes"
    man = json.loads((TMP / "video_k10" / "videos" / "manifest.json").read_text())
    assert [e["episode"] for e in man["episodes"]] == [0, 1, 2]
    for e in man["episodes"]:
        for stream in ("presentation", "witness/context", "witness/wrist", "witness/top", "policy/combined"):
            assert (TMP / "video_k10" / "videos" / e["files"][stream]["video"]).exists(), stream
    assert not list((TMP / "video_k10" / "videos").rglob("*.part.mp4"))
    print(f"[ok] videos: {sum(1 for _ in (TMP / 'video_k10' / 'videos').rglob('*.mp4'))} mp4 files, "
          f"manifest has {len(man['episodes'])} episodes")

    # replay a subset and verify against the full run
    full_csv = TMP / "chunk_k10" / "zeroshot.csv"
    rr, sr, _ = run("replay_k10", "--n_action_steps", "10", "--episodes", "3,7",
                    "--reference_csv", str(full_csv))
    assert [r["episode"] for r in rr] == ["3", "7"]
    assert sr["replay_check"]["n_mismatches"] == 0, sr["replay_check"]
    print("[ok] replay of episodes 3,7 matches the full run")

    # probe: same outcomes, divergence logged (perturbed wrist => nonzero)
    rpr, spr, _ = run("probe_tilt", "--n_action_steps", "10", "--tilt_target", "wrist",
                      "--tilt_deg", "10", "--probe_nominal")
    assert outcomes(rpr) == outcomes(rp), "probe changed outcomes"
    assert all(r["probe_div_mean"] not in ("", None) for r in rpr)
    assert max(float(r["probe_div_mean"]) for r in rpr) > 0
    print(f"[ok] probe: outcomes unchanged, mean divergence "
          f"{np.mean([float(r['probe_div_mean']) for r in rpr]):.4f}")

    # outputs
    rc, sc, _, _ = results["10"]
    for col in ("success", "success_strict", "success_carry", "reached_raw", "grasped_raw",
                "failure_mode", "stage", "t_reach", "min_grasp_dist", "reached", "lifted",
                "placed", "pss", "cube_x", "target_y", "n_replans", "terminated_reason"):
        assert col in rc[0], col
    for key in ("outcome_code_hash", "versions", "metric_config", "matched_eval_set",
                "n_action_steps", "loop_mode", "failure_modes", "timing"):
        assert key in sc, key
    assert sc["loop_mode"] == "closed_loop" and results["50"][1]["loop_mode"] == "open_loop"
    tr = np.load(TMP / "chunk_k10" / "trajectories.npz")
    assert tr["qpos"].shape[1] == 6 and len(tr["episode"]) == len(tr["t"])
    assert int(tr["replan"].sum()) == sum(int(r["n_replans"]) for r in rc)
    print(f"[ok] outputs: {len(rc[0])} CSV columns, trajectories {tr['qpos'].shape[0]} steps")

    # refuse to overwrite
    try:
        run("chunk_k10", "--n_action_steps", "10")
        raise AssertionError("overwrite guard did not trigger")
    except SystemExit:
        print("[ok] overwrite guard")
    return TMP


if __name__ == "__main__":
    tmp = test_all()
    print(f"\nALL PIPELINE TESTS PASSED  (artifacts in {tmp})")