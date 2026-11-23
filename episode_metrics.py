"""
episode_metrics.py
------------------
Per-episode task metrics for zeroshot_eval.py, kept free of any Genesis /
torch import so they can be unit-tested on plain numpy arrays and reused by
other scripts (e.g. an oracle evaluation).

Why this module exists (what changed vs. the old in-loop logic)
---------------------------------------------------------------
1. RAW sub-goal latches. The old rollout forced
       lifted  = lifted or placed
       reached = reached or lifted or placed
   before writing the CSV, which made stats_analysis.py's ordering check
   pass by construction. Here every latch records only what was observed.
   The legacy (forced) columns are still produced, under their old names,
   so earlier analysis scripts and earlier numbers remain reproducible.

2. A STRICT success definition. The legacy criterion ("carry": cube centre
   within 5 cm of the target and < 5 cm above the table) is met while the
   cube is still held above the pad. `strict_success` additionally requires
   the gripper to be open and the cube to be at rest on the table. Both are
   recorded; the analysis picks the primary one (default: strict).

3. Geometry-grounded "reached". The oracle commands the jaw-link origin to
   cube_centre + GRASP_POINT_OFFSET at the grasp waypoint
   (oracle_direct._compute_waypoints: x_offset=0.016, z = 0.01 + 0.06).
   The jaw origin therefore sits ~7.2 cm from the cube centre even during a
   PERFECT grasp, which is why a jaw-to-cube radius of 8 cm produced
   "placed without reached" episodes and was then widened to 12 cm. "Reached"
   is now the distance from the jaw origin to that grasp point, with a small
   radius (default 4 cm) that should be calibrated on oracle rollouts before
   the campaign. The continuous minimum distance is logged as well, so
   threshold-free plots (ECDFs) are possible.

4. Failure taxonomy, time-to-event and continuous diagnostics per episode,
   so the paper can say WHERE in the pipeline each condition fails.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np

CUBE_HALF = 0.015                                 # cube is 3 cm on a side
GRASP_POINT_OFFSET = np.array([0.016, 0.0, 0.07])  # jaw-origin target at grasp, from the oracle

# Ordered, mutually exclusive outcome categories (best -> worst).
FAILURE_MODES = (
    "success",                    # strict success
    "over_target_not_released",   # carry criterion met at some point, but never released at rest on target
    "grasped_not_delivered",      # held the cube, never brought it low over the target
    "reached_not_grasped",        # got to the grasp pose, no stable grasp
    "never_reached",              # never got to the grasp pose
)


@dataclass
class MetricConfig:
    """All thresholds in one place; stored in run_summary.json verbatim."""
    reach_radius:        float = 0.04   # m, jaw origin -> oracle grasp point
    gripper_closed_max:  float = 0.49   # joint value; open=0.8, close cmd=0.18 (midpoint)
    gripper_open_min:    float = 0.49   # joint value above which the gripper counts as open
    lift_min:            float = 0.01   # m above the resting height to count as lifted
    co_move_window:      int   = 5      # policy steps
    co_move_tol:         float = 0.01   # m, max per-axis std of cube-minus-jaw position
    target_radius:       float = 0.05   # m, XY radius of the target pad
    rest_max_height:     float = 0.025  # m, cube centre above table to count as resting (rest = 0.015)
    rest_max_speed:      float = 0.05   # m/s, cube speed to count as at rest
    policy_dt:           float = 0.1    # s per policy step (HOLD_STEPS * sim dt)
    success_hold_steps:  int   = 3      # strict success must hold this many consecutive steps to stop early
    off_table_margin:    float = 0.02   # m below the table surface => cube fell off
    # Legacy thresholds (reproduce the old GraspDetector / is_success exactly)
    legacy_near_radius:      float = 0.12
    legacy_close_target:     float = 0.18
    legacy_close_tol:        float = 0.05
    legacy_elevation_thresh: float = 0.02

    def to_dict(self) -> dict:
        return asdict(self)


def legacy_carry_success(cube_pos, target_pos, table_height, radius=0.05) -> bool:
    """Exact numpy mirror of two_camera_setup.is_success (the old TSR)."""
    xy = float(np.linalg.norm(np.asarray(cube_pos[:2]) - np.asarray(target_pos[:2])))
    z = float(cube_pos[2])
    return xy < radius and z > table_height + 0.005 and z < table_height + 0.05


def strict_success(cube_pos, prev_cube_pos, target_pos, gripper_q, table_height,
                   cfg: MetricConfig) -> bool:
    """Cube resting on the target pad with the gripper open."""
    cube_pos = np.asarray(cube_pos, dtype=float)
    on_target = np.linalg.norm(cube_pos[:2] - np.asarray(target_pos[:2], dtype=float)) < cfg.target_radius
    resting_h = (cube_pos[2] - table_height) < cfg.rest_max_height
    speed = (0.0 if prev_cube_pos is None
             else float(np.linalg.norm(cube_pos - np.asarray(prev_cube_pos, dtype=float))) / cfg.policy_dt)
    at_rest = speed < cfg.rest_max_speed
    released = float(gripper_q) > cfg.gripper_open_min
    return bool(on_target and resting_h and at_rest and released)


@dataclass
class EpisodeMonitor:
    """
    Feed one observation per policy step via update(); call summary() at
    the end. Also decides early termination (should_stop()).
    """
    table_height: float
    cfg: MetricConfig = field(default_factory=MetricConfig)

    def __post_init__(self):
        self.t = -1
        self._rel_hist: list[np.ndarray] = []
        self._prev_cube: Optional[np.ndarray] = None
        self._strict_run = 0
        self.lat = {k: False for k in ("reached", "grasped", "carry", "strict",
                                       "legacy_near", "legacy_grasped")}
        self.first = {k: None for k in ("reached", "grasped", "carry", "strict")}
        self.min_grasp_dist = np.inf
        self.min_jaw_cube_dist = np.inf
        self.max_cube_lift = 0.0
        self.strict_now = False
        self.last_cube = None
        self.last_target = None
        self.cube_off_table = False
        self.drop_after_grasp = False   # grasped, then cube back on table away from target


    def update(self, jaw_pos, cube_pos, target_pos, gripper_q) -> dict:
        c = self.cfg
        self.t += 1
        jaw = np.asarray(jaw_pos, dtype=float)
        cube = np.asarray(cube_pos, dtype=float)
        tgt = np.asarray(target_pos, dtype=float)
        q = float(gripper_q)

        d_grasp = float(np.linalg.norm(jaw - (cube + GRASP_POINT_OFFSET)))
        d_jaw_cube = float(np.linalg.norm(jaw - cube))
        lift = float(cube[2] - (self.table_height + CUBE_HALF))
        self.min_grasp_dist = min(self.min_grasp_dist, d_grasp)
        self.min_jaw_cube_dist = min(self.min_jaw_cube_dist, d_jaw_cube)
        self.max_cube_lift = max(self.max_cube_lift, lift)

        self._rel_hist.append(cube - jaw)
        if len(self._rel_hist) > c.co_move_window:
            self._rel_hist.pop(0)
        co_moving = (len(self._rel_hist) == c.co_move_window and
                     float(np.std(np.stack(self._rel_hist), axis=0).max()) < c.co_move_tol)

        reached = d_grasp < c.reach_radius
        grasped = (q < c.gripper_closed_max) and (lift > c.lift_min) and co_moving
        carry = legacy_carry_success(cube, tgt, self.table_height, radius=c.target_radius)
        strict = strict_success(cube, self._prev_cube, tgt, q, self.table_height, c)

        # Legacy flags (old GraspDetector semantics, for reproducibility)
        legacy_near = d_jaw_cube < c.legacy_near_radius
        legacy_grasped = (abs(q - c.legacy_close_target) < c.legacy_close_tol and
                          cube[2] > self.table_height + c.legacy_elevation_thresh and co_moving)

        for k, v in (("reached", reached), ("grasped", grasped), ("carry", carry), ("strict", strict)):
            if v and not self.lat[k]:
                self.first[k] = self.t
            self.lat[k] = self.lat[k] or v
        self.lat["legacy_near"] |= legacy_near
        self.lat["legacy_grasped"] |= legacy_grasped

        on_table_away = (lift < 0.005 and
                         np.linalg.norm(cube[:2] - tgt[:2]) >= c.target_radius)
        if self.lat["grasped"] and on_table_away and not grasped:
            self.drop_after_grasp = True

        self._strict_run = self._strict_run + 1 if strict else 0
        self.strict_now = strict
        self.cube_off_table = bool(cube[2] < self.table_height - c.off_table_margin)
        self._prev_cube = cube
        self.last_cube, self.last_target = cube, tgt

        return {"d_grasp": d_grasp, "d_jaw_cube": d_jaw_cube, "lift": lift,
                "reached": reached, "grasped": grasped, "carry": carry,
                "strict": strict, "co_moving": co_moving, "gripper_q": q}

    def should_stop(self) -> Optional[str]:
        if self._strict_run >= self.cfg.success_hold_steps:
            return "success_hold"
        if self.cube_off_table:
            return "cube_off_table"
        return None


    def summary(self, terminated_reason: str) -> dict:
        lat, c = self.lat, self.cfg
        success_strict = bool(self.strict_now)       # state at termination
        success_carry = bool(lat["carry"])           # ever met (== the old TSR)

        if success_strict:
            mode = "success"
        elif lat["carry"]:
            mode = "over_target_not_released"
        elif lat["grasped"]:
            mode = "grasped_not_delivered"
        elif lat["reached"]:
            mode = "reached_not_grasped"
        else:
            mode = "never_reached"
        stage = len(FAILURE_MODES) - 1 - FAILURE_MODES.index(mode)   # 4=success ... 0=never_reached

        # Legacy columns, computed EXACTLY as before (including the forced
        # implication), so old scripts / old numbers are reproducible.
        placed_l = success_carry
        lifted_l = bool(lat["legacy_grasped"] or placed_l)
        reached_l = bool(lat["legacy_near"] or lifted_l or placed_l)
        pss_legacy = round(0.2 * reached_l + 0.4 * lifted_l + 1.0 * placed_l, 3)

        final_xy = (float(np.linalg.norm(self.last_cube[:2] - self.last_target[:2]))
                    if self.last_cube is not None else float("nan"))
        return {
            # primary / new
            "success_strict": success_strict,
            "success_carry": success_carry,
            "reached_raw": bool(lat["reached"]),
            "grasped_raw": bool(lat["grasped"]),
            "pushed_success": bool(success_strict and not lat["grasped"]),
            "failure_mode": mode,
            "stage": stage,
            "t_reach": self.first["reached"],
            "t_grasp": self.first["grasped"],
            "t_carry": self.first["carry"],
            "t_success": self.first["strict"],
            "min_grasp_dist": round(float(self.min_grasp_dist), 5),
            "min_jaw_cube_dist": round(float(self.min_jaw_cube_dist), 5),
            "max_cube_lift": round(float(self.max_cube_lift), 5),
            "final_cube_target_dist": round(final_xy, 5),
            "drop_after_grasp": bool(self.drop_after_grasp),
            "terminated_reason": terminated_reason,
            "violation_grasp_without_reach": bool(lat["grasped"] and not lat["reached"]),
            "violation_success_without_grasp": bool(success_strict and not lat["grasped"]),
            # legacy (old names, old semantics)
            "reached": reached_l,
            "lifted": lifted_l,
            "placed": placed_l,
            "pss": pss_legacy,
        }


def parse_index_spec(spec: str | None, n: int) -> list[int]:
    """
    Parse an episode-index spec: "", "none" -> []; "all" -> 0..n-1;
    "0-4,10,17" -> [0,1,2,3,4,10,17]. Indices >= n are an error.
    """
    if spec is None:
        return []
    spec = spec.strip().lower()
    if spec in ("", "none"):
        return []
    if spec == "all":
        return list(range(n))
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    bad = [i for i in out if i < 0 or i >= n]
    if bad:
        raise ValueError(f"episode indices {sorted(bad)} out of range [0, {n})")
    return sorted(out)


def sphere_geometry(spheres, cube_xy, target_xy) -> dict:
    """Initial-state distances between distractor spheres and cube/target edges (m).
    Negative target_clearance means the sphere footprint overlaps the target pad."""
    if not spheres:
        return {"sphere_target_clearance": None, "sphere_cube_clearance": None}
    t_clear, c_clear = [], []
    for s in spheres:
        sxy = np.asarray(s.xy, dtype=float)
        t_clear.append(float(np.linalg.norm(sxy - np.asarray(target_xy))) - s.radius - 0.05)
        c_clear.append(float(np.linalg.norm(sxy - np.asarray(cube_xy))) - s.radius - CUBE_HALF)
    return {"sphere_target_clearance": round(min(t_clear), 5),
            "sphere_cube_clearance": round(min(c_clear), 5)}