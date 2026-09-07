"""
oracle_direct.py  —  SO-101 scripted oracle + LeRobot v3.0 dataset writer
=========================================================================

Runs a deterministic scripted policy (oracle) on the Genesis / SO-101
pick-and-place task and writes every successful (and optionally failed)
episode directly into a LeRobot v3.0 dataset on disk.

Key entry point
---------------
collect_demonstrations()
    Runs n_episodes oracle rollouts and writes them into a LeRobot v3.0
    dataset.  Accepts a ScenePerturbation to control cube color, fixed
    spawn positions, and distractor spheres.

Oracle behaviour
----------------
The oracle computes five Cartesian waypoints (pre-grasp → grasp → lift
→ carry → place), interpolates joint configurations between them via IK,
and records joint state + camera images at every record_every_n_steps
simulator step.  Small Gaussian noise is added to waypoints and IK
orientations so the recorded trajectories are not perfectly identical.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import imageio.v3 as iio
import numpy as np
import torch

from scene_params import (
    language_instruction_for, DEFAULT_CUBE_COLOR,
    CameraPerturbation, NOMINAL_PERTURBATION,
    ScenePerturbation, NOMINAL_SCENE,
    SphereConfig,
)

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as e:
    raise ImportError(
        "lerobot not found. Install from source with SmolVLA extras:\n"
        "  pip install -e '.[smolvla]'  (inside lerobot repo)\n"
        f"Original error: {e}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset schema constants
# ══════════════════════════════════════════════════════════════════════════════

CAMERA_KEYS = ["camera1", "camera2", "camera3"]
STATE_DIM   = 6   # six joint positions (5 arm + 1 gripper)
ACTION_DIM  = 6   # six joint position targets
FPS         = 10  # recording frame rate

# Map human-readable camera roles to LeRobot dataset keys
_CAM_REMAP   = {"context": "camera1", "wrist": "camera2", "top": "camera3"}
# Reverse mapping (camera key → role name)
_KEY_TO_ROLE = {v: k for k, v in _CAM_REMAP.items()}

# Joint index slices used for arm vs. gripper control
_ARM_DOFS    = np.arange(5)
_GRIPPER_DOF = np.array([5])


def make_features(img_h: int = 256, img_w: int = 256) -> dict:
    """Build the LeRobot v3.0 features schema for this task."""
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": {"motors": [f"motor_{i}" for i in range(STATE_DIM)]},
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": {"motors": [f"motor_{i}" for i in range(ACTION_DIM)]},
        },
    }
    for cam_key in CAMERA_KEYS:
        features[f"observation.images.{cam_key}"] = {
            "dtype": "video",
            "shape": (img_h, img_w, 3),
            "names": ["height", "width", "channel"],
        }
    return features


# ══════════════════════════════════════════════════════════════════════════════
#  Oracle config
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OracleConfig:
    """Tunable parameters for the scripted oracle policy."""
    pos_noise_sigma:      float = 0.004          # std of XY waypoint jitter (m)
    rot_noise_sigma:      float = np.deg2rad(1)  # std of orientation jitter (rad)
    steps_per_segment:    int   = 30             # interpolation steps between waypoints
    pregrasp_clearance:   float = 0.12           # height above cube for approach (m)
    lift_height:          float = 0.15           # height above table after grasping (m)
    carry_clearance:      float = 0.15           # height above table during carry (m)
    gripper_open:         float = 0.8            # gripper joint value when open
    gripper_close_safe:   float = 0.18           # gripper joint value when grasping
    ik_pos_tol:           float = 1e-4           # IK position convergence tolerance (m)
    ik_rot_tol:           float = 1e-4           # IK rotation convergence tolerance (rad)
    # Number of settle steps after each waypoint (one entry per segment)
    settle_steps: list = field(default_factory=lambda: [10, 50, 100, 25, 25, 60])
    record_every_n_steps: int   = 10             # record a frame once every N sim steps


# ══════════════════════════════════════════════════════════════════════════════
#  Oracle policy
# ══════════════════════════════════════════════════════════════════════════════

class SO101Oracle:
    """
    Scripted oracle for the SO-101 pick-and-place task.

    Executes a fixed sequence of Cartesian waypoints via IK, recording
    joint state and camera frames at each step.  A ScenePerturbation can
    be passed so the language instruction embedded in each frame correctly
    reflects the current cube colour.
    """

    def __init__(
        self,
        so101,
        scene,
        cameras=None,
        witness_cameras=None,
        perturbation: CameraPerturbation = NOMINAL_PERTURBATION,
        cfg: OracleConfig = OracleConfig(),
        language_instruction: str | None = None,
        seed: int | None = None,
    ):
        self.robot           = so101
        self.scene           = scene
        self.cameras         = cameras
        self.witness_cameras = witness_cameras
        self.perturbation    = perturbation
        self.cfg             = cfg
        # Default instruction; can be overridden per-episode when cube color changes
        self.language_instruction = (
            language_instruction or language_instruction_for(DEFAULT_CUBE_COLOR)
        )
        # Dedicated RNG for waypoint/orientation noise (System Noise, README
        # section II-E). Seeded so the exact noise sequence for a given
        # dataset-generation run is reproducible and independent of the
        # env/rollout seed used elsewhere in the pipeline (ST5).
        self._rng = np.random.default_rng(seed)

        self.end_effector = so101.get_link("moving_jaw_so101_v1")
        # Downward-facing grasp orientation (quaternion: w, x, y, z)
        self.grasp_quat   = np.array([0.707107, 0.0, -0.707107, 0.0])

        self._frames:         list[dict] = []
        self._witness_frames: list[dict] = []
        self._last_qpos:      torch.Tensor | None = None
        self._step_counter:   int = 0

    # ── Public API ──────────────────────────────────────────────────────────

    def run_episode(self, cube, target_zone, table_height,
                    is_success_fn, check_sub_goals_fn,
                    language_instruction: str | None = None) -> bool:
        """
        Execute one pick-and-place episode and return whether it succeeded.

        language_instruction overrides the instance default for this episode,
        which is useful when cube colour changes between episodes.
        """
        self._frames         = []
        self._witness_frames = []
        self._last_qpos      = None
        self._step_counter   = 0
        self._subgoal_latch: dict = {"lifted": False}

        if language_instruction is not None:
            self.language_instruction = language_instruction

        cube_pos   = cube.get_pos().cpu().numpy()
        target_pos = target_zone.get_pos().cpu().numpy()

        waypoints = self._compute_waypoints(cube_pos, target_pos, table_height)
        # Insert a duplicate of waypoint 1 (grasp pose) so the gripper has
        # extra settle time before closing
        waypoints.insert(2, waypoints[1].copy())

        # Gripper state for each movement segment (open during approach, closed after grasp)
        gripper_targets = [
            self.cfg.gripper_open,
            self.cfg.gripper_open,
            self.cfg.gripper_close_safe,
            self.cfg.gripper_close_safe,
            self.cfg.gripper_close_safe,
        ]

        prev_qpos = self.robot.get_dofs_position()
        success   = False

        # Execute each approach/carry segment; stop early if the block is placed
        for wp_pos, g_target, s_steps in zip(
                waypoints[:-1], gripper_targets, self.cfg.settle_steps[:-1]):
            prev_qpos, success = self._execute_segment(
                wp_pos, g_target, s_steps, prev_qpos,
                cube, target_zone, table_height, check_sub_goals_fn,
                self._subgoal_latch,
            )
            if success:
                break

        # Final segment: move to release position and open gripper
        if not success:
            success = self._execute_release_segment(
                waypoints[-1], self.cfg.settle_steps[-1], prev_qpos,
                cube, target_zone, table_height, is_success_fn,
            )

        # Last-chance success check after the gripper has fully opened
        if not success:
            success = is_success_fn(cube, target_zone, table_height)

        return success

    def get_recorded_frames(self) -> list[dict]:
        return self._frames

    def get_witness_frames(self) -> list[dict]:
        return self._witness_frames

    # ── Frame recording ──────────────────────────────────────────────────────

    def _record_step(self, qpos: torch.Tensor):
        """Render cameras and store a dataset frame (called every N sim steps)."""
        self._step_counter += 1
        if self._step_counter % self.cfg.record_every_n_steps != 0:
            return
        if self.cameras is None:
            return

        frame  = {}
        # Wrist camera is physically attached to the gripper link and must be
        # moved to its current attach position before rendering
        ATTACHED = {"wrist"}

        for role, cam in self.cameras.items():
            if role in ATTACHED:
                cam.move_to_attach()
            rgb, _, _, _ = cam.render()
            cam_key      = _CAM_REMAP.get(role, role)
            # Apply any blackout perturbation (e.g. zero out this camera stream)
            rgb          = self.perturbation.apply_blackout(rgb, cam_key)
            frame[f"observation.images.{cam_key}"] = rgb

        # Record the *actual* joint positions as state and the *commanded*
        # positions as action, so the dataset captures real execution noise
        actual_qpos = self.robot.get_dofs_position()
        frame["observation.state"] = actual_qpos[:6].cpu().numpy().astype(np.float32)
        frame["action"]            = qpos[:6].cpu().numpy().astype(np.float32)
        frame["task"]              = self.language_instruction
        self._frames.append(frame)

        # Witness cameras are only for debugging video; they are never in the dataset
        if self.witness_cameras:
            wf = {}
            for w_role, w_cam in self.witness_cameras.items():
                if "wrist" in w_role:
                    w_cam.move_to_attach()
                rgb_w, _, _, _ = w_cam.render()
                wf[w_role] = rgb_w
            self._witness_frames.append(wf)

    # ── Segment execution ────────────────────────────────────────────────────

    def _execute_segment(self, target_cart_pos, gripper_target, settle_steps,
                         prev_qpos, cube, target_zone, table_height,
                         check_sub_goals_fn, latch):
        """
        Interpolate from prev_qpos to the IK solution for target_cart_pos,
        then settle for settle_steps.  Returns (final_qpos, placed).
        """
        cfg = self.cfg
        # Add small orientation noise so trajectories vary across episodes
        noisy_quat = self._perturb_quat_z(
            self.grasp_quat, self._rng.normal(0, cfg.rot_noise_sigma)
        )
        target_qpos = self.robot.inverse_kinematics(
            link=self.end_effector, pos=target_cart_pos, quat=noisy_quat,
            pos_tol=cfg.ik_pos_tol, rot_tol=cfg.ik_rot_tol,
        )
        target_qpos[5] = gripper_target

        # Linear interpolation: shape (steps_per_segment, n_dofs)
        alphas = torch.linspace(0.0, 1.0, cfg.steps_per_segment,
                                device=target_qpos.device).unsqueeze(1)
        interp_configs       = prev_qpos + alphas * (target_qpos - prev_qpos)
        # Keep gripper at its current value throughout the arm motion;
        # the gripper opens/closes only during settle steps
        interp_configs[:, 5] = prev_qpos[5]

        def _step(qpos) -> bool:
            """Send one command, step the sim, and check if the block is placed."""
            self._record_step(qpos)
            self.robot.control_dofs_position(qpos[_ARM_DOFS],    dofs_idx_local=_ARM_DOFS)
            self.robot.control_dofs_position(qpos[_GRIPPER_DOF], dofs_idx_local=_GRIPPER_DOF)
            self.scene.step()
            return check_sub_goals_fn(
                self.robot, cube, target_zone, table_height, latch
            )["placed"]

        # Move along the interpolated path
        for interp_qpos in interp_configs:
            if _step(interp_qpos):
                return interp_qpos, True
        # Hold the final pose for settle_steps to let physics stabilise
        for _ in range(settle_steps):
            if _step(target_qpos):
                return target_qpos, True
        return target_qpos, False

    def _execute_release_segment(self, target_cart_pos, settle_steps,
                                  prev_qpos, cube, target_zone, table_height,
                                  is_success_fn):
        """
        Move to the release position with the gripper closed, then open
        it and settle.  Returns True if the block ends on the target.
        """
        cfg = self.cfg
        noisy_quat = self._perturb_quat_z(
            self.grasp_quat, self._rng.normal(0, cfg.rot_noise_sigma)
        )
        target_qpos = self.robot.inverse_kinematics(
            link=self.end_effector, pos=target_cart_pos, quat=noisy_quat,
            pos_tol=cfg.ik_pos_tol, rot_tol=cfg.ik_rot_tol,
        )

        # Approach with gripper still closed (carry the block to the release pose)
        closed_qpos    = target_qpos.clone()
        closed_qpos[5] = prev_qpos[5]
        alphas = torch.linspace(0.0, 1.0, cfg.steps_per_segment,
                                device=target_qpos.device).unsqueeze(1)
        interp       = prev_qpos + alphas * (closed_qpos - prev_qpos)
        interp[:, 5] = prev_qpos[5]

        for interp_qpos in interp:
            self._record_step(interp_qpos)
            self.robot.control_dofs_position(interp_qpos[_ARM_DOFS],    dofs_idx_local=_ARM_DOFS)
            self.robot.control_dofs_position(interp_qpos[_GRIPPER_DOF], dofs_idx_local=_GRIPPER_DOF)
            self.scene.step()

        # Brief hold at the release pose before opening
        for _ in range(15):
            self._record_step(closed_qpos)
            self.robot.control_dofs_position(closed_qpos[_ARM_DOFS],    dofs_idx_local=_ARM_DOFS)
            self.robot.control_dofs_position(closed_qpos[_GRIPPER_DOF], dofs_idx_local=_GRIPPER_DOF)
            self.scene.step()

        # Open the gripper to release the block
        open_qpos    = closed_qpos.clone()
        open_qpos[5] = cfg.gripper_open

        for _ in range(20):
            self._record_step(open_qpos)
            self.robot.control_dofs_position(open_qpos[_ARM_DOFS],    dofs_idx_local=_ARM_DOFS)
            self.robot.control_dofs_position(open_qpos[_GRIPPER_DOF], dofs_idx_local=_GRIPPER_DOF)
            self.scene.step()

        # Let the block settle on the surface before checking success
        for _ in range(settle_steps):
            self._record_step(open_qpos)
            self.robot.control_dofs_position(open_qpos[_ARM_DOFS],    dofs_idx_local=_ARM_DOFS)
            self.robot.control_dofs_position(open_qpos[_GRIPPER_DOF], dofs_idx_local=_GRIPPER_DOF)
            self.scene.step()

        return is_success_fn(cube, target_zone, table_height)

    # ── Waypoint computation ─────────────────────────────────────────────────

    def _compute_waypoints(self, cube_pos, target_pos, table_height):
        """
        Compute the five Cartesian end-effector waypoints for one episode.

        Waypoints (before noise):
          wp0 — pre-grasp: above the cube
          wp1 — grasp: at the cube
          wp2 — lift: above the cube after grasping
          wp3 — carry: above the target zone
          wp4 — place: just above the target zone
        """
        cfg            = self.cfg
        gripper_length = 0.06   # distance from wrist to fingertip (m)
        # Small offsets tuned empirically to align the gripper with the cube
        x_offset, y_offset = 0.016, -0.000

        wp0 = cube_pos   + np.array([x_offset, y_offset, cfg.pregrasp_clearance + gripper_length])
        wp1 = cube_pos   + np.array([x_offset, y_offset, 0.01 + gripper_length])
        wp2 = cube_pos   + np.array([x_offset, y_offset, cfg.lift_height + gripper_length])
        wp3 = target_pos + np.array([0.0,      0.0,      cfg.carry_clearance + gripper_length])
        wp4 = target_pos + np.array([x_offset, 0.0,      0.04 + gripper_length])

        waypoints = []
        for nominal in [wp0, wp1, wp2, wp3, wp4]:
            noise    = self._rng.normal(0, cfg.pos_noise_sigma, size=3)
            noise[2] = abs(noise[2])    # never push waypoints into the table
            noise[0] *= 0.3             # reduce noise along the reach axis
            noise[1] *= 0.3
            waypoints.append(nominal + noise)
        return waypoints

    @staticmethod
    def _perturb_quat_z(base_quat, angle_rad):
        """Compose base_quat with a small rotation around the world Z axis."""
        half = angle_rad / 2.0
        dq   = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
        w1, x1, y1, z1 = base_quat
        w2, x2, y2, z2 = dq
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])


# ══════════════════════════════════════════════════════════════════════════════
#  Witness video writer
# ══════════════════════════════════════════════════════════════════════════════

def _save_witness_video(witness_frames, episode_id, success, witness_dir,
                         fps=10, prefix="episode_"):
    """Save per-camera and side-by-side combined witness videos for one episode."""
    if not witness_frames:
        return
    label     = "pass" if success else "fail"
    stem      = f"{prefix}{episode_id:04d}_{label}"
    cam_roles = list(witness_frames[0].keys())

    # Write one video per camera
    for role in cam_roles:
        short = role.replace("witness_", "")
        out   = witness_dir / short
        out.mkdir(parents=True, exist_ok=True)
        frames_np = np.stack([wf[role] for wf in witness_frames])
        iio.imwrite(str(out / f"{stem}.mp4"), frames_np, fps=fps,
                    codec="libx264",
                    output_params=["-crf", "18", "-pix_fmt", "yuv420p"])

    # Write a combined side-by-side video for quick visual review
    combined_dir = witness_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_frames = [
        np.concatenate([wf[r] for r in cam_roles], axis=1)
        for wf in witness_frames
    ]
    iio.imwrite(str(combined_dir / f"{stem}.mp4"), np.stack(combined_frames),
                fps=fps, codec="libx264",
                output_params=["-crf", "18", "-pix_fmt", "yuv420p"])


# ══════════════════════════════════════════════════════════════════════════════
#  collect_demonstrations
# ══════════════════════════════════════════════════════════════════════════════

def collect_demonstrations(
    so101,
    scene,
    cameras,
    cube,
    target_zone,
    table_height,
    is_success_fn,
    check_sub_goals_fn,
    n_episodes: int                              = 200,
    output_dir: str                              = "demos/lerobot/",
    fps: int                                     = FPS,
    repo_id: str                                 = "local/genesis_pickplace",
    push_to_hub: bool                            = False,
    save_failures: bool                          = True,
    # Kept for backwards compatibility — ScenePerturbation takes precedence
    cube_color: str                              = DEFAULT_CUBE_COLOR,
    img_h: int                                   = 256,
    img_w: int                                   = 256,
    witness_cameras: dict | None                 = None,
    perturbation: CameraPerturbation             = NOMINAL_PERTURBATION,
    scene_perturbation: ScenePerturbation | None = None,
    sphere_entities: list | None                 = None,
    seed: int | None                             = None,
) -> list[bool]:
    """
    Run the oracle for n_episodes and write directly into a LeRobot v3.0 dataset.

    Parameters
    ----------
    so101, scene, cameras, cube, target_zone, table_height
        Genesis scene objects from build_environment() and scene.build().
    is_success_fn, check_sub_goals_fn
        Callables that evaluate task completion and sub-goal progress.
    n_episodes : int
        Number of oracle rollouts to attempt.
    output_dir : str
        Directory to write the LeRobot dataset into.
    push_to_hub : bool
        If True, upload the finished dataset to the Hugging Face Hub.
    save_failures : bool
        If True, failed episodes are written to a separate failures dataset.
    cube_color : str
        Fallback cube color used when scene_perturbation is None.
    scene_perturbation : ScenePerturbation or None
        Controls cube colour, fixed spawn positions, and distractor spheres.
        When None, falls back to the cube_color kwarg.
    sphere_entities : list[gs.Entity] or None
        Genesis entity handles for distractor spheres returned by
        build_environment().  These are repositioned at the start of each
        episode to match scene_perturbation.spheres.
    seed : int or None
        Dataset-generation RNG seed (ST5: kept distinct from the
        environment/rollout seed used in zeroshot_eval.py, and from the
        --train_seed passed to finetune.py). Controls both the spawn
        rejection-sampling draws in this loop AND the oracle's own
        waypoint/orientation noise (passed through to SO101Oracle). None
        means "unseeded" (previous, non-reproducible default) — pass an
        explicit int for any run you intend to report or reproduce.

    Returns
    -------
    list[bool]
        Per-episode success flags.
    """
    # Build a default ScenePerturbation from the legacy cube_color kwarg if needed
    if scene_perturbation is None:
        scene_perturbation = ScenePerturbation(cube_color=cube_color)

    eff_color = scene_perturbation.cube_color
    print(f"[info] Scene perturbation : {scene_perturbation.label}")
    print(f"[info] Cube colour        : {eff_color}")
    print(f"[info] Dataset-gen seed   : {seed if seed is not None else 'UNSEEDED (not reproducible)'}")
    if scene_perturbation.cube_position_preset:
        print(f"[info] Cube position      : {scene_perturbation.cube_position_preset}")
    if scene_perturbation.spheres:
        print(f"[info] Distractor spheres : "
              f"{[s.label for s in scene_perturbation.spheres]}")

    cfg    = OracleConfig()
    oracle = SO101Oracle(
        so101, scene, cameras,
        witness_cameras=witness_cameras,
        perturbation=perturbation,
        cfg=cfg,
        language_instruction=scene_perturbation.language_instruction,
        seed=seed,
    )
    features = make_features(img_h, img_w)
    out_dir  = Path(output_dir).resolve()
    fail_dir = out_dir.parent / "failures_lerobot"

    witness_dir = out_dir.parent / "witness_videos"
    if witness_cameras:
        witness_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Writing dataset to : {out_dir}")
    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=fps, root=str(out_dir),
        features=features, use_videos=True, image_writer_threads=4,
    )

    # Optionally create a parallel dataset for failed episodes
    fail_dataset = None
    if save_failures:
        fail_dataset = LeRobotDataset.create(
            repo_id=repo_id + "_failures", fps=fps, root=str(fail_dir),
            features=features, use_videos=True, image_writer_threads=4,
        )

    home_pose  = np.zeros(so101.n_dofs)
    # Same seed drives spawn rejection-sampling as drives the oracle's own
    # waypoint noise above — one dataset-generation seed controls the whole
    # run, deliberately, so the run is fully reproducible from one number.
    rng        = np.random.default_rng(seed)
    results: list[bool] = []
    successes  = 0
    saved_ep   = 0    # count of episodes saved to the success dataset
    saved_fail = 0    # count of episodes saved to the failure dataset

    # Sphere configs used for per-episode repositioning
    sphere_cfgs = scene_perturbation.spheres if scene_perturbation else []

    print(f"\nCollecting {n_episodes} episodes  "
          f"(fps={fps}, perturbation={perturbation.label}, "
          f"scene={scene_perturbation.label})\n")

    for ep in range(n_episodes):
        # ── Resolve spawn positions ─────────────────────────────────────────
        # Returns fixed coords if cube_position_preset is set, else samples randomly
        cube_xy, target_xy = scene_perturbation.resolve_spawn(rng)

        # ── Reset robot to home pose ────────────────────────────────────────
        so101.set_dofs_position(home_pose)
        so101.set_dofs_velocity(np.zeros(so101.n_dofs))
        so101.control_dofs_position(home_pose)

        # ── Reset cube and target zone positions ─────────────────────────────
        cube.set_pos(np.array([cube_xy[0], cube_xy[1], table_height + 0.015]))
        cube.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))
        target_zone.set_pos(np.array([target_xy[0], target_xy[1], table_height + 0.001]))

        # ── Reposition distractor spheres to their configured locations ──────
        if sphere_entities and sphere_cfgs:
            for entity, cfg_s in zip(sphere_entities, sphere_cfgs):
                z = table_height + cfg_s.radius
                entity.set_pos(np.array([cfg_s.xy[0], cfg_s.xy[1], z]))
                entity.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))

        # ── Settle the scene before the oracle starts ─────────────────────────
        for _ in range(20):
            so101.set_dofs_velocity(np.zeros(so101.n_dofs))
            so101.control_dofs_position(home_pose)
            scene.step()

        # ── Run oracle ───────────────────────────────────────────────────────
        success = oracle.run_episode(
            cube, target_zone, table_height,
            is_success_fn, check_sub_goals_fn,
            language_instruction=scene_perturbation.language_instruction,
        )

        label   = "PASS ✓" if success else "FAIL ✗"
        frames  = oracle.get_recorded_frames()
        wframes = oracle.get_witness_frames()
        print(f"[{ep + 1:3d}/{n_episodes}] {label}  T={len(frames)}", flush=True)

        results.append(success)

        if not frames:
            print("           [warn] no frames recorded — skipping")
            continue

        # Write frames to the appropriate dataset (success or failure)
        target_ds = dataset if success else fail_dataset
        if target_ds is not None:
            for frame in frames:
                target_ds.add_frame(frame)
            target_ds.save_episode()

        if witness_cameras and wframes:
            ep_index = saved_ep if success else saved_fail
            _save_witness_video(wframes, ep_index, success, witness_dir, fps=fps)

        if success:
            successes += 1
            saved_ep  += 1
        else:
            saved_fail += 1

    # ── Finalize datasets (write meta/info.json, close video writers) ─────────
    print("\n[info] Finalizing dataset …")
    dataset.finalize()
    if fail_dataset is not None:
        fail_dataset.finalize()

    if push_to_hub:
        dataset.push_to_hub()
        if fail_dataset is not None:
            fail_dataset.push_to_hub()

    # ── Write a manifest capturing exactly how this dataset was generated ────
    # (M8: reproducibility manifest needs this to exist somewhere; this is
    # the dataset-generation half of it.)
    manifest = {
        "seed": seed,
        "n_episodes_requested": n_episodes,
        "n_episodes_succeeded": successes,
        "repo_id": repo_id,
        "output_dir": str(out_dir),
        "camera_perturbation": perturbation.label,
        "scene_perturbation": scene_perturbation.label,
        "fps": fps,
        "img_h": img_h,
        "img_w": img_w,
    }
    import json as _json
    (out_dir / "generation_manifest.json").write_text(_json.dumps(manifest, indent=2))

    print(f"\n{'─' * 50}")
    print(f"Collection complete : {successes}/{n_episodes} "
          f"({successes / max(n_episodes, 1):.1%})")
    print(f"Scene perturbation  : {scene_perturbation.label}")
    print(f"Dataset-gen seed    : {seed if seed is not None else 'UNSEEDED'}")
    print(f"Dataset written to  : {out_dir}")
    if save_failures:
        print(f"Failures written to : {fail_dir}")
    if witness_cameras:
        print(f"Witness videos      : {witness_dir}")
    print(f"{'─' * 50}")

    return results