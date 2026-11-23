"""
oracle_feasibility.py
---------------------
Control experiment: is the task still SOLVABLE under a scene perturbation?

Runs the privileged scripted oracle (oracle_direct.SO101Oracle, which uses
ground-truth cube/target poses and no cameras) on the same matched initial
states the policy is evaluated on. If the oracle's success drops under, e.g.,
--spheres two_flanking, part of the policy's drop there is physical
obstruction, not a perception failure. Camera perturbations cannot affect
the oracle, so this is only meaningful for scene perturbations (spheres,
cube position, cube colour).

    python oracle_feasibility.py --matched_eval_set $MATCH --n 100 \\
        --spheres two_flanking --out runs/paper-v2/oracle/sph-2flank.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zeroshot_eval as ze  # noqa: E402  (scene construction + reset, identical to the eval)
from episode_metrics import MetricConfig, strict_success, sphere_geometry  # noqa: E402
from scene_params import (ScenePerturbation, SPHERE_PRESETS, CUBE_POSITIONS,  # noqa: E402
                          CUBE_COLORS, NOMINAL_PERTURBATION)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matched_eval_set", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0, help="oracle waypoint-noise seed")
    ap.add_argument("--spheres", default="none", choices=list(SPHERE_PRESETS))
    ap.add_argument("--cube_pos", default=None, choices=[None] + list(CUBE_POSITIONS))
    ap.add_argument("--cube_color", default="red", choices=list(CUBE_COLORS))
    ap.add_argument("--xml", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    from oracle_direct import SO101Oracle, OracleConfig
    _setup = ze._setup

    scene_p = ScenePerturbation(cube_color=a.cube_color, cube_position_preset=a.cube_pos,
                                spheres=SPHERE_PRESETS[a.spheres])
    pairs = json.loads(Path(a.matched_eval_set).read_text())["cube_target_pairs"]
    spawns = ze.resolve_spawns(a.n, scene_p, pairs, seed=42)
    xml = a.xml or str((ze._script_dir / ".." / "so101_arm" / "so101_new_calib.xml").resolve())
    (scene, so101, cube, target_zone, spheres, _, _, _, th) = ze.build_scene(
        xml, NOMINAL_PERTURBATION, scene_p, with_witness=False, with_probe=False)
    oracle = SO101Oracle(so101, scene, cameras=None, cfg=OracleConfig(), seed=a.seed)
    cfg = MetricConfig()

    def sub_goals(robot, c, tz, h, latch=None):
        return _setup.check_sub_goals(robot, c, tz, h, _latch=latch)

    rows = []
    for ep in range(a.n):
        cxy, txy = spawns[ep]
        ze.reset_episode(scene, so101, cube, target_zone, th, cxy, txy, scene_p, spheres)
        carry = bool(oracle.run_episode(cube, target_zone, th, _setup.is_success, sub_goals,
                                        language_instruction=scene_p.language_instruction))
        for _ in range(ze.HOLD_STEPS):          # one more policy step to measure cube speed
            scene.step()
        prev = ze._np(cube.get_pos())
        for _ in range(ze.HOLD_STEPS):
            scene.step()
        cube_p = ze._np(cube.get_pos())
        strict = strict_success(cube_p, prev, ze._np(target_zone.get_pos()),
                                float(ze._np(so101.get_dofs_position())[5]), th, cfg)
        rows.append({"episode": ep, "oracle_success_carry": carry, "oracle_success_strict": strict,
                     "cube_x": float(cxy[0]), "cube_y": float(cxy[1]),
                     "target_x": float(txy[0]), "target_y": float(txy[1]),
                     **sphere_geometry(scene_p.spheres, cxy, txy)})
        print(f"{ep:4d} carry={carry} strict={strict}", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    s = {k: float(np.mean([r[k] for r in rows])) for k in ("oracle_success_carry", "oracle_success_strict")}
    out.with_suffix(".json").write_text(json.dumps({"scene_perturbation": scene_p.label, "n": a.n, **s}, indent=2))
    print(f"[ok] {out}: oracle TSR carry={s['oracle_success_carry']:.1%} strict={s['oracle_success_strict']:.1%}")


if __name__ == "__main__":
    main()