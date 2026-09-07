"""
zeroshot_eval.py
----------------
Run N rollouts of a SmolVLA checkpoint in the Genesis scene and record
Task Success Rate (TSR) and Partial Success Score (PSS).

Three perturbation axes are exposed via CLI (and via ScenePerturbation):

  --cube_color   <color>      Change the block colour (red/blue/green/yellow).
                               Also changes the language instruction fed to
                               the policy so it stays internally consistent.

  --cube_pos     <preset>     Fix cube + target spawn to a named preset from
                               CUBE_POSITIONS (e.g. workspace_edge, far_left).
                               Omit to use the normal randomised distribution.

  --spheres      <preset>     Add distractor spheres from SPHERE_PRESETS
                               (e.g. two_flanking, high_clutter).  Spheres
                               are re-placed at the start of every episode.

Usage
-----
    # Nominal
    python zeroshot_eval.py --checkpoint lerobot/smolvla_base \\
        --dataset_dir demos/lerobot/ --n 100 --run_root icra-final

    # Blue cube
    python zeroshot_eval.py ... --cube_color blue

    # Fixed OOD spawn position
    python zeroshot_eval.py ... --cube_pos workspace_edge

    # High clutter
    python zeroshot_eval.py ... --spheres high_clutter

    # Combined: blue cube + far_left spawn + two flanking spheres
    python zeroshot_eval.py ... \\
        --cube_color blue --cube_pos far_left --spheres two_flanking

    # Combined with camera perturbation
    python zeroshot_eval.py ... --blackout_mode wrist_only --spheres single_left

NOTE ON N, --out, AND --run_root
----------------------------------
--n controls exactly how many rows land in the output CSV; every
downstream Wilson CI is computed from the actual row count, so keep
--n consistent across conditions you intend to compare directly.

Either --out or --run_root is required (there is no default path).
--run_root <tag> auto-derives a unique path per condition under
runs/<tag>/eval/<condition_slug>/ (see run_naming.py) from the actual
CameraPerturbation/ScenePerturbation values you passed, so two different
conditions can never collide. Either way, the script refuses to overwrite
an existing CSV or run_summary.json unless you also pass --overwrite.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

from scene_params import (
    IMG_RES, language_instruction_for, DEFAULT_CUBE_COLOR, CUBE_COLORS,
    CameraPerturbation, NOMINAL_PERTURBATION,
    ScenePerturbation, NOMINAL_SCENE,
    CUBE_POSITIONS, SPHERE_PRESETS,
    SphereConfig,
    BLACKOUT_MODES,
)

import genesis as gs

try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
except ImportError:
    print("[error] lerobot not installed or SmolVLA policy not found.")
    sys.exit(1)

import importlib.util
import pathlib

def _import_from(module_name, file_path):
    """Import a module from an explicit file path (used for sibling scripts)."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_script_dir = pathlib.Path(__file__).parent
_setup      = _import_from("two_camera_setup", _script_dir / "two_camera_setup.py")

build_environment                  = _setup.build_environment
attach_cameras                     = _setup.attach_cameras
attach_witness_cameras_with_robot  = _setup.attach_witness_cameras_with_robot
is_success                         = _setup.is_success
check_sub_goals                    = _setup.check_sub_goals


# ══════════════════════════════════════════════════════════════════════════════
#  Policy wrapper
# ══════════════════════════════════════════════════════════════════════════════

class SmolVLAAgent:
    """
    Thin wrapper around SmolVLAPolicy that delegates ALL observation
    preprocessing (renaming, batching, newline-terminating the language
    instruction, tokenising, device placement, state/action normalisation)
    and action postprocessing (unnormalising, device placement) to the
    REAL PolicyProcessorPipeline saved alongside the checkpoint
    (policy_preprocessor.json / policy_postprocessor.json + their
    .safetensors state files), loaded via make_pre_post_processors(...,
    pretrained_path=checkpoint).

    An earlier version of this class hand-reimplemented tokenisation and
    mean/std normalisation directly from stats.json. That reimplementation
    had drifted from what training actually used, in two confirmed ways:
      1. It never applied NewLineTaskProcessorStep, which SmolVLA's own
         factory (make_smolvla_pre_post_processors) always inserts to
         ensure the language instruction ends with a newline before
         tokenisation — every rollout was feeding the model a differently
         terminated instruction than it was fine-tuned on.
      2. It tokenised with max_length=64 / padding=True, while training
         used config.tokenizer_max_length / config.pad_language_to
         (confirmed 48 / "max_length" for this checkpoint) via
         TokenizerProcessorStep.
    Loading the pipeline directly, instead of re-copying its steps here,
    means this class can no longer drift from training the same way again
    — if the pipeline's steps change, this class picks the change up
    automatically instead of silently falling further out of sync.
    """

    def __init__(self, checkpoint: str, dataset_dir: str, device: str = "cuda",
                 language_instruction: str | None = None):
        self.device = device
        self._default_instruction = (
            language_instruction or language_instruction_for(DEFAULT_CUBE_COLOR)
        )
        print(f"[info] Loading SmolVLA checkpoint: {checkpoint}")
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint, device=device)
        self.policy.eval()

        # Load the checkpoint's OWN saved pre/post-processor pipeline rather
        # than reconstructing it from dataset_stats — this is the exact
        # pipeline object training produced, including its normalizer's
        # learned mean/std (from the .safetensors state file), so there is
        # no separate stats.json to keep in sync by hand any more.
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=checkpoint,
        )
        # dataset_dir is kept as a constructor arg for CLI/back-compat only;
        # it is no longer read directly (no more hand-parsed stats.json).
        self._dataset_dir = dataset_dir
        print("[info] SmolVLA + real pre/post-processor pipeline loaded successfully.")

    def reset(self):
        """Reset the policy's internal recurrent state (action queue) between episodes."""
        self.policy.reset()

    @torch.no_grad()
    def act(self, context_rgb, wrist_rgb, top_rgb, joint_state,
            language_instruction=None, absent_cameras: set[str] | None = None):
        """
        Compute one action from the current observation via the real
        preprocessor -> policy -> postprocessor chain.

        Cameras listed in absent_cameras are omitted from the observation
        dict entirely (tensor not passed), simulating a missing sensor.
        Cameras that are zeroed by a blackout mode still have their tensors
        passed; the policy sees a black image rather than nothing.

        Returns a (6,) float32 array of joint position targets.
        """
        instruction    = language_instruction or self._default_instruction
        absent_cameras = absent_cameras or set()

        def img_to_tensor(arr):
            # HWC uint8 -> CHW float32 in [0, 1]. No batch dim and no
            # .to(device) here: AddBatchDimensionProcessorStep and
            # DeviceProcessorStep inside self.preprocessor do both, exactly
            # as they did when this dataset was collected/trained on.
            t = torch.from_numpy(arr.copy()).float() / 255.0
            return t.permute(2, 0, 1)

        # Build the RAW observation dict — un-normalised, un-batched,
        # un-tokenised. self.preprocessor does all of that.
        obs = {}
        if "camera1" not in absent_cameras and context_rgb is not None:
            obs["observation.images.camera1"] = img_to_tensor(context_rgb)
        if "camera2" not in absent_cameras and wrist_rgb is not None:
            obs["observation.images.camera2"] = img_to_tensor(wrist_rgb)
        if "camera3" not in absent_cameras and top_rgb is not None:
            obs["observation.images.camera3"] = img_to_tensor(top_rgb)
        obs["observation.state"] = torch.from_numpy(joint_state.copy()).float()
        # "task" is the raw instruction string — NewLineTaskProcessorStep +
        # TokenizerProcessorStep (task_key="task") handle newline-termination
        # and tokenisation, matching training exactly.
        obs["task"] = instruction

        batch    = self.preprocessor(obs)
        action_t = self.policy.select_action(batch)
        # Defensive: policy may in principle return (B, T, A); take the
        # first timestep. In practice select_action() already pops a single
        # (B, A) timestep off its internal action-chunk queue.
        if action_t.ndim == 3:
            action_t = action_t[:, 0, :]
        action_t = self.postprocessor(action_t)
        return action_t.squeeze(0).cpu().numpy().astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Scene builder
# ══════════════════════════════════════════════════════════════════════════════

def build_scene(
    xml_path: str,
    perturbation: CameraPerturbation = NOMINAL_PERTURBATION,
    scene_perturbation: ScenePerturbation = NOMINAL_SCENE,
):
    """
    Initialise Genesis and build the full evaluation scene.

    Returns
    -------
    scene, so101, cube, target_zone, sphere_entities,
    training_cams, witness_cams, table_height
    """
    gs.init(backend=gs.gpu, seed=0)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=16),
        rigid_options=gs.options.RigidOptions(
            constraint_solver=gs.constraint_solver.Newton,
            iterations=100, tolerance=1e-9,
            constraint_timeconst=0.006,
            enable_self_collision=True, box_box_detection=True,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False, show_cameras=False,
            ambient_light=(0.25, 0.25, 0.25),
            lights=[
                {"type": "directional", "dir": (-0.5, -0.5, -1.0),
                 "intensity": 8.0, "color": (1.0, 0.97, 0.90)},
                {"type": "directional", "dir": (0.5, 0.5, -0.5),
                 "intensity": 2.5, "color": (0.6, 0.7, 1.0)},
            ],
        ),
        renderer=gs.renderers.Rasterizer(),
        show_viewer=False,
    )

    scene.add_entity(gs.morphs.Plane())

    table_height = 0.8
    cube, target_zone, sphere_entities = build_environment(
        scene, table_height,
        cube_color=scene_perturbation.cube_color,
        spheres=scene_perturbation.spheres,
    )

    so101 = scene.add_entity(
        gs.morphs.MJCF(file=xml_path, pos=(0.0, 0.0, table_height))
    )

    training_cams = attach_cameras(scene, so101, table_height, perturbation=perturbation)
    witness_cams  = attach_witness_cameras_with_robot(scene, so101, table_height)

    scene.build()

    # Controller gains tuned for the SO-101 at this table height
    arm_dofs    = np.arange(5)
    gripper_dof = np.array([5])
    so101.set_dofs_kp(np.array([4000, 4000, 3000, 2000, 2000]), dofs_idx_local=arm_dofs)
    so101.set_dofs_kv(np.array([400,  400,  300,  200,  200]),  dofs_idx_local=arm_dofs)
    so101.set_dofs_kp(np.array([800.0]), dofs_idx_local=gripper_dof)
    so101.set_dofs_kv(np.array([80.0]),  dofs_idx_local=gripper_dof)
    so101.set_dofs_force_range(
        lower=np.array([-100.0]), upper=np.array([100.0]),
        dofs_idx_local=gripper_dof,
    )
    for link_name in ["gripper", "moving_jaw_so101_v1"]:
        so101.get_link(link_name).set_friction(5.0)

    return (scene, so101, cube, target_zone, sphere_entities,
            training_cams, witness_cams, table_height)


# ══════════════════════════════════════════════════════════════════════════════
#  Episode reset
# ══════════════════════════════════════════════════════════════════════════════

def reset_episode(
    scene, so101, cube, target_zone, table_height, rng,
    scene_perturbation: ScenePerturbation = NOMINAL_SCENE,
    sphere_entities: list | None = None,
    matched_pair: dict | None = None,
):
    """
    Reset robot, cube, target zone, and spheres for a new rollout.

    matched_pair : dict or None
        If provided (as {"cube_xy": [x,y], "target_xy": [x,y]}, from a
        frozen matched-eval-set file — see generate_matched_eval_set.py),
        this OVERRIDES scene_perturbation.resolve_spawn()'s own draw so
        every condition starts this episode index from the identical
        physical state (ST4: paired comparisons). Ignored (with a
        one-time warning from the caller) for conditions that already
        use a fixed cube_position_preset, since those are already
        matched by construction and don't consume rng draws the same way.

    Returns (cube_xy, target_xy) so the caller can log spawn positions.
    """
    home = np.zeros(so101.n_dofs)

    if matched_pair is not None and scene_perturbation.cube_position_preset is None:
        cube_xy   = np.array(matched_pair["cube_xy"],   dtype=float)
        target_xy = np.array(matched_pair["target_xy"], dtype=float)
    else:
        # Returns fixed coords if cube_position_preset is set, else samples randomly
        cube_xy, target_xy = scene_perturbation.resolve_spawn(rng)

    so101.set_dofs_position(home)
    so101.set_dofs_velocity(np.zeros(so101.n_dofs))
    so101.control_dofs_position(home)

    cube.set_pos(np.array([cube_xy[0], cube_xy[1], table_height + 0.015]))
    cube.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))
    target_zone.set_pos(np.array([target_xy[0], target_xy[1], table_height + 0.001]))

    if sphere_entities and scene_perturbation.spheres:
        for entity, cfg_s in zip(sphere_entities, scene_perturbation.spheres):
            z = table_height + cfg_s.radius
            entity.set_pos(np.array([cfg_s.xy[0], cfg_s.xy[1], z]))
            entity.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))

    # Settle the scene before the policy starts acting
    for _ in range(20):
        so101.set_dofs_velocity(np.zeros(so101.n_dofs))
        so101.control_dofs_position(home)
        scene.step()

    return cube_xy, target_xy


# ══════════════════════════════════════════════════════════════════════════════
#  Rollout constants
# ══════════════════════════════════════════════════════════════════════════════

MAX_STEPS            = 50    # maximum policy steps per episode
HOLD_STEPS           = 10    # simulator steps per policy step (action hold duration)
RECORD_EVERY         = 5     # save a video frame once every N policy steps
ARM_DOFS             = np.arange(5)
GRIPPER_DOF          = np.array([5])
GRIPPER_CLOSE_TARGET = 0.18  # gripper joint value considered "closed"


# ══════════════════════════════════════════════════════════════════════════════
#  GraspDetector
# ══════════════════════════════════════════════════════════════════════════════

class GraspDetector:
    """
    Heuristic detector for whether the robot has successfully grasped the cube.

    A grasp is declared when all three conditions hold simultaneously:
      1. The gripper is in the closed position (joint value near target).
      2. The cube is elevated above the table.
      3. The cube is co-moving with the gripper (low relative-position variance
         over the last `window` steps), indicating physical contact.

    This avoids false positives from a closed gripper that missed the cube.

    NOTE: this "grasped" definition is intentionally stricter than
    check_sub_goals()'s "lifted" flag in two_camera_setup.py (which only
    checks elevation + proximity, with no gripper-closed or co-moving
    check). The two are not interchangeable metrics even though both feed
    into a PSS "lift" term — check_sub_goals() is used during oracle
    demonstration collection (where a loose check is fine, since the
    oracle's trajectory guarantees a real grasp), while GraspDetector is
    used here during zero-shot evaluation of a learned policy, where a
    closed-gripper-but-missed-the-cube false positive is a real risk.
    """

    def __init__(self, table_height, jaw_link_name="moving_jaw_so101_v1",
                 gripper_close_target=GRIPPER_CLOSE_TARGET, gripper_closed_tol=0.05,
                 near_radius=0.08, elevation_thresh=0.02, window=5, rel_pos_tol=0.01):
        self.table_height         = table_height
        self.jaw_link_name        = jaw_link_name
        self.gripper_close_target = gripper_close_target
        self.gripper_closed_tol   = gripper_closed_tol
        self.near_radius          = near_radius
        self.elevation_thresh     = elevation_thresh
        self.window               = window
        self.rel_pos_tol          = rel_pos_tol
        self._rel_pos_history     = []

    def reset(self):
        self._rel_pos_history = []

    def step(self, so101, cube):
        """
        Update the detector with the current state and return a status dict.

        Returns
        -------
        dict with keys: near, grasped, dist, is_closed, is_elevated, co_moving
        """
        jaw_pos   = so101.get_link(self.jaw_link_name).get_pos()
        cube_pos  = cube.get_pos()
        gripper_q = so101.get_dofs_position()[5].item()
        dist      = torch.norm(jaw_pos - cube_pos).item()
        near      = dist < self.near_radius
        is_closed = abs(gripper_q - self.gripper_close_target) < self.gripper_closed_tol
        is_elev   = cube_pos[2].item() > (self.table_height + self.elevation_thresh)

        # Track relative position of cube w.r.t. gripper over the last `window` steps
        rel_pos = (cube_pos - jaw_pos).cpu().numpy()
        self._rel_pos_history.append(rel_pos)
        if len(self._rel_pos_history) > self.window:
            self._rel_pos_history.pop(0)

        # Co-moving: max std across XYZ axes is below tolerance
        co_moving = False
        if len(self._rel_pos_history) == self.window:
            spread    = np.std(self._rel_pos_history, axis=0).max()
            co_moving = spread < self.rel_pos_tol

        grasped = is_closed and is_elev and co_moving
        return {"near": near, "grasped": grasped, "dist": dist,
                "is_closed": is_closed, "is_elevated": is_elev, "co_moving": co_moving}


# ══════════════════════════════════════════════════════════════════════════════
#  Run one rollout
# ══════════════════════════════════════════════════════════════════════════════

def run_rollout(
    agent,
    scene,
    so101,
    cube,
    target_zone,
    training_cams: dict,
    witness_cams: dict,
    table_height: float,
    perturbation: CameraPerturbation       = NOMINAL_PERTURBATION,
    scene_perturbation: ScenePerturbation  = NOMINAL_SCENE,
    cube_xy=None,
    target_xy=None,
) -> dict:
    """
    Execute one policy rollout and return metrics plus raw video frames.

    The returned dict contains: success, pss, steps, reached, lifted,
    placed, cube_xy, target_xy, _train_vid, _wit_vid.
    The _train_vid and _wit_vid entries hold raw frame lists and should
    be consumed (or discarded) by the caller before logging to CSV.
    """
    agent.reset()
    grasp_detector = GraspDetector(table_height=table_height)
    grasp_detector.reset()
    # Latch flags: once a sub-goal is achieved it stays True for the episode.
    # Keys match exactly what they latch — "reached" latches gd["near"],
    # "grasped" latches gd["grasped"] (GraspDetector's stricter definition,
    # see GraspDetector docstring). There is no separate "lifted" latch key:
    # the episode's `lifted` metric IS latch["grasped"], read below.
    latch = {"reached": False, "grasped": False}

    context_cam = training_cams["context"]
    wrist_cam   = training_cams["wrist"]
    top_cam     = training_cams["top"]

    reached = lifted = placed = False
    step    = 0

    # Accumulate video frames for optional saving
    train_vid = {"context": [], "wrist": [], "top": []}
    wit_vid   = {k: [] for k in witness_cams}

    # Language instruction reflects the current cube colour
    lang_instr = scene_perturbation.language_instruction

    while step < MAX_STEPS:

        # ── Render training cameras (apply blackout/absent modes) ─────────────
        ctx_rgb = None
        if not perturbation.camera_is_absent("camera1"):
            ctx_rgb, _, _, _ = context_cam.render()
            ctx_rgb = perturbation.apply_blackout(ctx_rgb, "camera1")

        wrist_rgb = None
        if not perturbation.camera_is_absent("camera2"):
            wrist_cam.move_to_attach()   # update wrist cam pose from gripper FK
            wrist_rgb, _, _, _ = wrist_cam.render()
            wrist_rgb = perturbation.apply_blackout(wrist_rgb, "camera2")

        top_rgb = None
        if not perturbation.camera_is_absent("camera3"):
            top_rgb, _, _, _ = top_cam.render()
            top_rgb = perturbation.apply_blackout(top_rgb, "camera3")

        # ── Render witness cameras (no perturbation applied) ──────────────────
        for w_role, w_cam in witness_cams.items():
            if "wrist" in w_role:
                w_cam.move_to_attach()
            w_rgb, _, _, _ = w_cam.render()
            if step % RECORD_EVERY == 0:
                wit_vid[w_role].append(w_rgb.copy())

        # ── Save training video frames at reduced frame rate ──────────────────
        if step % RECORD_EVERY == 0:
            if ctx_rgb   is not None: train_vid["context"].append(ctx_rgb.copy())
            if wrist_rgb is not None: train_vid["wrist"].append(wrist_rgb.copy())
            if top_rgb   is not None: train_vid["top"].append(top_rgb.copy())

        # ── Query policy ──────────────────────────────────────────────────────
        qpos   = so101.get_dofs_position().cpu().numpy()
        absent = {k for k in ["camera1", "camera2", "camera3"]
                  if perturbation.camera_is_absent(k)}
        # NOTE (S2): SmolVLA is fine-tuned with a FIXED 3-camera schema
        # (observation.images.camera1/2/3 all present in every training
        # frame). Standard LeRobot policies iterate over the camera keys
        # baked into the checkpoint's config at training time, so an
        # "absent_*" blackout mode (which omits a key from the obs dict
        # entirely, unlike "no_top"/"no_wrist"/"no_context" which still
        # pass a zeroed-but-present tensor) is very likely to raise inside
        # policy.select_action() rather than "gracefully degrade" — see
        # inspect_observation_schema.py, which must be run and confirmed
        # clean BEFORE any absent_* TSR number is trusted for the paper.
        #
        # There is deliberately no bare `except` around this call: a run
        # that can't complete must fail loudly (consistent with the
        # completeness check at the end of main()), not silently report a
        # 0% TSR that looks identical to "the policy tried and failed."
        # If you need every condition in a sweep to survive one bad
        # condition, catch a specific, narrow exception here and record it
        # as its own outcome (e.g. a `policy_raised` column) rather than
        # conflating it with `success=False`.
        action = agent.act(ctx_rgb, wrist_rgb, top_rgb, qpos[:6],
                           language_instruction=lang_instr,
                           absent_cameras=absent)

        # ── Execute action for HOLD_STEPS simulation steps ───────────────────
        arm_target     = action[:5]
        gripper_target = np.array([action[5]])
        for _ in range(HOLD_STEPS):
            so101.control_dofs_position(arm_target,     dofs_idx_local=ARM_DOFS)
            so101.control_dofs_position(gripper_target, dofs_idx_local=GRIPPER_DOF)
            scene.step()

        # ── Check sub-goals and update latch flags ────────────────────────────
        sg = check_sub_goals(so101, cube, target_zone, table_height)
        gd = grasp_detector.step(so101, cube)

        if gd["near"]:
            latch["reached"] = True
        if gd["grasped"]:
            latch["grasped"] = True

        # TEMPORARY DIAGNOSTIC (S-debug): set DEBUG_GRASP=1 to trace the
        # near/grasped geometry across a rollout. Added to investigate a
        # confirmed placed_without_reached / lifted_without_reached
        # violation (stats_analysis.py's internal-consistency check) —
        # i.e. episodes where the cube was successfully grasped and placed
        # WITHOUT GraspDetector's "near" latch (jaw-LINK-ORIGIN-to-cube
        # distance < near_radius=0.08) ever tripping. If dist stays above
        # 0.08 for the whole episode despite is_closed/is_elevated/
        # co_moving all being True, that confirms the jaw link's origin is
        # offset from the actual fingertip contact point by more than the
        # current near_radius, and near_radius (or the reference link/point
        # it's measured from) needs to change — not a one-off fluke.
        # Remove this block once the root cause is confirmed and fixed.
        if os.environ.get("DEBUG_GRASP"):
            print(f"    [debug step {step:2d}] dist={gd['dist']:.4f}  "
                  f"near={gd['near']}  is_closed={gd['is_closed']}  "
                  f"is_elevated={gd['is_elevated']}  co_moving={gd['co_moving']}  "
                  f"grasped={gd['grasped']}  "
                  f"latch_reached={latch['reached']}  latch_grasped={latch['grasped']}")

        reached = latch["reached"]
        lifted  = latch["grasped"]
        placed  = sg["placed"]

        if placed:
            break
        step += 1

    success = is_success(cube, target_zone, table_height)
    # PSS: 0.2 for reaching, +0.4 for lifting, +1.0 for placing
    pss     = 0.2 * reached + 0.4 * lifted + 1.0 * placed

    return {
        "success":    success,
        "pss":        round(pss, 3),
        "steps":      step,
        "reached":    reached,
        "lifted":     lifted,
        "placed":     placed,
        "cube_xy":    cube_xy.tolist() if cube_xy is not None else None,
        "target_xy":  target_xy.tolist() if target_xy is not None else None,
        "_train_vid": train_vid,
        "_wit_vid":   wit_vid,
    }


def _save_episode_videos(r, tag, policy_video_dir, witness_video_dir, fps=15):
    """Write per-camera and combined side-by-side videos for one episode."""
    def _write(frames, path):
        if frames:
            iio.imwrite(str(path), np.stack(frames), fps=fps,
                        codec="libx264",
                        output_params=["-crf", "22", "-pix_fmt", "yuv420p"])

    train_vid = r["_train_vid"]
    wit_vid   = r["_wit_vid"]

    # Per-camera training videos
    for role in ("context", "wrist", "top"):
        (policy_video_dir / role).mkdir(parents=True, exist_ok=True)
        _write(train_vid[role], policy_video_dir / role / f"{tag}.mp4")

    # Combined side-by-side training video (only if at least two cameras have frames)
    present = [train_vid[r] for r in ("context", "wrist", "top") if train_vid[r]]
    if len(present) > 1:
        combined_policy = [
            np.concatenate(frames, axis=1)
            for frames in zip(*present)
        ]
        # This mkdir was missing here (present on the witness-video equivalent
        # below, but not mirrored to this — policy — branch), which meant any
        # episode reaching this code path crashed on the first attempt to
        # write into videos/policy/combined/ before that directory ever
        # existed. Every OTHER per-role directory gets created just above
        # (line ~591), so this was purely a missed mirror, not an
        # intentional asymmetry.
        (policy_video_dir / "combined").mkdir(parents=True, exist_ok=True)
        _write(combined_policy, policy_video_dir / "combined" / f"{tag}.mp4")

    # Per-camera witness videos
    for w_role in wit_vid:
        short = w_role.replace("witness_", "")
        (witness_video_dir / short).mkdir(parents=True, exist_ok=True)
        _write(wit_vid[w_role], witness_video_dir / short / f"{tag}.mp4")

    # Combined side-by-side witness video
    w_roles = list(wit_vid.keys())
    if len(w_roles) > 1 and wit_vid[w_roles[0]]:
        combined_wit = [
            np.concatenate([wit_vid[r][i] for r in w_roles], axis=1)
            for i in range(len(wit_vid[w_roles[0]]))
        ]
        (witness_video_dir / "combined").mkdir(parents=True, exist_ok=True)
        _write(combined_wit, witness_video_dir / "combined" / f"{tag}.mp4")


# ══════════════════════════════════════════════════════════════════════════════
#  Run summary
# ══════════════════════════════════════════════════════════════════════════════

def build_run_summary(records, args, perturbation, scene_perturbation, csv_path,
                       matched_eval_set_info=None):
    """
    Build the aggregate + per-episode summary dict written to
    run_summary.json alongside the CSV.

    `grasped_count` is the canonical name for the "lifted" sub-goal — it's
    named after what GraspDetector actually checks (gripper closed AND
    cube elevated AND cube co-moving with the gripper), which is stricter
    than the "lifted" label alone suggests. `lifted_count` is kept as an
    exact alias (same value) since the CSV column is already named
    `lifted` and downstream tooling may expect that word.

    Every count here is a direct sum over `records` — nothing is
    recomputed from a different source or assumed from N, so this file
    can only report what was actually observed in this run.

    matched_eval_set_info : dict or None
        {"path": ..., "content_hash": ...} when --matched_eval_set was used.
        Stamping this lets you verify AFTER THE FACT that every run in a
        sweep you intend to compare actually used the identical frozen
        initial conditions (same content_hash), rather than trusting
        yourself not to have pointed two runs at different files by
        mistake. Compare content_hash across run_summary.json files
        before trusting any paired/matched comparison between them.
    """
    n = len(records)
    successes     = sum(1 for r in records if r["success"])
    reached_count = sum(1 for r in records if r["reached"])
    grasped_count = sum(1 for r in records if r["lifted"])   # see docstring
    placed_count  = sum(1 for r in records if r["placed"])
    mean_pss      = float(np.mean([r["pss"] for r in records]))

    episodes = [
        {
            "episode": r["episode"],
            "success": r["success"],
            "pss":     r["pss"],
            "steps":   r["steps"],
            "reached": r["reached"],
            "grasped": r["lifted"],   # alias, same underlying GraspDetector value
            "lifted":  r["lifted"],
            "placed":  r["placed"],
            "cube_xy":   r.get("cube_xy"),
            "target_xy": r.get("target_xy"),
        }
        for r in records
    ]

    return {
        "generated_at_utc":    datetime.now(timezone.utc).isoformat(),
        "checkpoint":          args.checkpoint,
        "checkpoint_id":       getattr(args, "checkpoint_id", None) or Path(args.checkpoint).name,
        "train_seed":          getattr(args, "train_seed", None),
        "env_seed":            args.seed,
        "matched_eval_set":    matched_eval_set_info,   # None if unmatched (e.g. --cube_pos preset)
        "dataset_dir":         args.dataset_dir,
        "csv_path":            str(csv_path),
        "camera_perturbation": perturbation.label,
        "scene_perturbation":  scene_perturbation.label,
        "total_episodes":      n,
        "successes":           successes,
        "reached_count":       reached_count,
        "grasped_count":       grasped_count,
        "lifted_count":        grasped_count,   # alias, identical value
        "placed_count":        placed_count,
        "tsr":                 successes / n,
        "mean_pss":            round(mean_pss, 6),
        "episodes":            episodes,
    }


def write_run_summary(summary, out_dir):
    """Write run_summary.json next to the CSV. Refuses to silently overwrite
    a summary from a different run size without saying so."""
    path = out_dir / "run_summary.json"
    if path.exists():
        try:
            prior_n = json.loads(path.read_text()).get("total_episodes")
        except Exception:
            prior_n = None
        if prior_n is not None and prior_n != summary["total_episodes"]:
            print(f"[warn] Overwriting {path} — previous run had "
                  f"total_episodes={prior_n}, this run has "
                  f"total_episodes={summary['total_episodes']}.")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[ok] Run summary saved → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default=None,
                        help="Path to the SO-101 MJCF XML. Defaults to "
                             "<this script's directory>/../so101_arm/so101_new_calib.xml "
                             "— i.e. assumes so101_arm/ is a sibling of the folder "
                             "containing this script, same convention two_camera_setup.py "
                             "uses. Resolved relative to THIS FILE, not your current "
                             "working directory, so it works the same regardless of "
                             "where you run `python zeroshot_eval.py` from. Override "
                             "with an absolute or explicit relative path if your "
                             "layout differs.")
    parser.add_argument("--checkpoint",  default="lerobot/smolvla_base")
    parser.add_argument("--checkpoint_id", default=None,
                        help="Human label for this checkpoint, e.g. 'seed0', "
                             "'seed1'. Distinct from --seed (which is the "
                             "environment/rollout RNG seed). REQUIRED once you "
                             "have more than one fine-tuned checkpoint, so "
                             "results from different training runs are never "
                             "pooled together implicitly (ST5). Defaults to the "
                             "last path component of --checkpoint if omitted.")
    parser.add_argument("--train_seed",  type=int, default=None,
                        help="The random seed used when fine-tuning this "
                             "checkpoint (dataset-generation and/or training "
                             "seed — record whichever your finetune.py run "
                             "actually varied). Stored in run_summary.json "
                             "for hierarchical analysis (ST5); purely "
                             "informational, not used by this script.")
    parser.add_argument("--n",           type=int, default=100,
                        help="Episodes to run. Every downstream Wilson CI is "
                             "computed from the actual CSV row count, so keep "
                             "this consistent across conditions you plan to "
                             "compare (analyze_all_results.py will warn, not "
                             "silently rescale, if N differs from what you expect).")
    parser.add_argument("--out",         default=None,
                        help="Output CSV path. Either this OR --run_root is "
                             "required. If both are given, --out wins. If only "
                             "--run_root is given, the path is derived automatically "
                             "as runs/<run_root>/eval/<condition_slug>/zeroshot.csv "
                             "(see run_naming.py) — this never collides with paths "
                             "used before this convention existed.")
    parser.add_argument("--run_root", default=None,
                        help="Short tag for this experiment campaign, e.g. "
                             "'icra-final'. When set (and --out is not explicitly "
                             "given), every output path is derived under "
                             "runs/<run_root>/, isolated from any prior "
                             "demos/outputs/results trees.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting an existing CSV/run_summary/video "
                             "directory at the resolved output path. Without this, "
                             "the script refuses to run if the target already has data.")
    parser.add_argument("--seed",        type=int, default=42,
                        help="Environment/rollout RNG seed (cube+target spawn "
                             "sampling, waypoint noise via ScenePerturbation). "
                             "This is the 'env_seed' in ST5's terminology — "
                             "separate from --train_seed.")
    parser.add_argument("--dataset_dir", default="demos/lerobot/")
    parser.add_argument("--matched_eval_set", default=None,
                        help="Path to a frozen matched-eval-set JSON from "
                             "generate_matched_eval_set.py (ST4). When set, "
                             "every episode's cube/target spawn comes from "
                             "this file by episode index instead of being "
                             "independently sampled, so this run is paired "
                             "with every other run that used the same file. "
                             "Ignored for --cube_pos conditions (already "
                             "fixed by construction). --n must not exceed "
                             "the file's episode count.")

    # ── Camera perturbation ───────────────────────────────────────────────────
    parser.add_argument(
        "--blackout_mode", default="none",
        choices=list(BLACKOUT_MODES),
    )
    parser.add_argument("--pos_target",  default=None,
                        choices=[None, "context", "wrist", "top"])
    parser.add_argument("--pos_offset",  nargs=3, type=float, default=[0.0, 0.0, 0.0],
                        metavar=("DX", "DY", "DZ"))
    parser.add_argument("--tilt_target", default=None,
                        choices=[None, "context", "wrist", "top"])
    parser.add_argument("--tilt_deg",    type=float, default=0.0)

    # ── Scene perturbation ────────────────────────────────────────────────────
    parser.add_argument(
        "--cube_color", default=DEFAULT_CUBE_COLOR,
        choices=list(CUBE_COLORS.keys()),
        help="Cube colour — also changes the language instruction to the policy.",
    )
    parser.add_argument(
        "--cube_pos", default=None,
        choices=[None] + list(CUBE_POSITIONS.keys()),
        help="Fix cube/target spawn to a named preset (omit for random spawn).",
    )
    parser.add_argument(
        "--spheres", default="none",
        choices=list(SPHERE_PRESETS.keys()),
        help="Distractor sphere layout from SPHERE_PRESETS.",
    )

    args = parser.parse_args()

    if args.out is None and args.run_root is None:
        print("[error] Either --out or --run_root is required. Every condition "
              "must write to its own explicit, non-colliding path.")
        sys.exit(1)

    perturbation = CameraPerturbation(
        blackout_mode   = args.blackout_mode,
        position_target = args.pos_target,
        position_offset = tuple(args.pos_offset),
        tilt_target     = args.tilt_target,
        tilt_deg        = args.tilt_deg,
    )
    scene_perturbation = ScenePerturbation(
        cube_color           = args.cube_color,
        cube_position_preset = args.cube_pos,
        spheres              = SPHERE_PRESETS[args.spheres],
    )

    # ── Resolve output paths: explicit --out wins; otherwise derive from
    #    --run_root using the shared naming convention in run_naming.py so
    #    every script agrees on where things go and nothing from an older
    #    run (or an older naming scheme) can collide with it. ──────────────
    if args.out is not None:
        out_csv           = Path(args.out)
        policy_video_dir  = out_csv.parent / "videos" / "policy"
        witness_video_dir = out_csv.parent / "videos" / "witness"
    else:
        from run_naming import run_root as _run_root, condition_slug, eval_paths
        slug  = condition_slug(perturbation, scene_perturbation)
        root  = _run_root(args.run_root)
        paths = eval_paths(root, slug)
        out_csv           = paths["csv"]
        policy_video_dir  = paths["policy_video_dir"]
        witness_video_dir = paths["witness_video_dir"]
        print(f"[run_root] {args.run_root} -> condition slug '{slug}' -> {out_csv}")

    from run_naming import refuse_if_exists
    refuse_if_exists(out_csv, "Output CSV", args.overwrite)
    refuse_if_exists(out_csv.parent / "run_summary.json", "Run summary", args.overwrite)

    args.out = str(out_csv)   # downstream code (build_run_summary etc.) reads args.out

    print(f"[camera perturbation] {perturbation.label}")
    print(f"[scene  perturbation] {scene_perturbation.label}")
    print(f"[N] {args.n} episodes  →  {args.out}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.xml is None:
        xml_path = str((_script_dir / ".." / "so101_arm" / "so101_new_calib.xml").resolve())
    else:
        xml_path = os.path.abspath(args.xml)
    (scene, so101, cube, target_zone, sphere_entities,
     training_cams, witness_cams, table_height) = build_scene(
        xml_path, perturbation=perturbation,
        scene_perturbation=scene_perturbation,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent  = SmolVLAAgent(
        checkpoint=args.checkpoint,
        dataset_dir=args.dataset_dir,
        device=device,
        language_instruction=scene_perturbation.language_instruction,
    )

    rng     = np.random.default_rng(args.seed)
    records = []

    matched_pairs = None
    if args.matched_eval_set:
        matched_data = json.loads(Path(args.matched_eval_set).read_text())
        matched_pairs = matched_data["cube_target_pairs"]
        if args.n > len(matched_pairs):
            print(f"[error] --n {args.n} exceeds the matched eval set's "
                  f"{len(matched_pairs)} episodes. Regenerate the matched "
                  f"set with a larger --n or lower --n here.")
            sys.exit(1)
        if scene_perturbation.cube_position_preset is not None:
            print(f"[warn] --matched_eval_set is ignored for this run: "
                  f"--cube_pos={scene_perturbation.cube_position_preset} "
                  f"already fixes the spawn for every episode.")
        else:
            print(f"[info] Using matched eval set {args.matched_eval_set} "
                  f"(content_hash={matched_data.get('content_hash')}) — "
                  f"this run is paired with every other run using the "
                  f"same file (ST4).")

    policy_video_dir.mkdir(parents=True, exist_ok=True)
    witness_video_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning {args.n} rollouts  (checkpoint: {args.checkpoint})\n")
    print(f"{'ep':>4}  {'result':>6}  {'pss':>5}  {'steps':>6}")
    print("─" * 40)

    # Limit success video saves to avoid filling disk
    MAX_SUCCESS_VIDEOS  = 5
    success_video_count = 0

    for ep in range(args.n):
        cube_xy, target_xy = reset_episode(
            scene, so101, cube, target_zone, table_height, rng,
            scene_perturbation=scene_perturbation,
            sphere_entities=sphere_entities,
            matched_pair=(matched_pairs[ep] if matched_pairs else None),
        )
        r = run_rollout(
            agent, scene, so101, cube, target_zone,
            training_cams, witness_cams, table_height,
            perturbation=perturbation,
            scene_perturbation=scene_perturbation,
            cube_xy=cube_xy, target_xy=target_xy,
        )
        r["episode"]       = ep
        r["checkpoint_id"] = args.checkpoint_id or Path(args.checkpoint).name
        r["train_seed"]    = args.train_seed
        r["env_seed"]      = args.seed

        res_str = "PASS ✓" if r["success"] else "FAIL ✗"
        print(f"{ep:4d}  {res_str:>6}  {r['pss']:>5.3f}  {r['steps']:>6}")

        # Save videos for all failures and the first few successes
        save_video = False
        if r["success"] and success_video_count < MAX_SUCCESS_VIDEOS:
            tag = f"ep{ep:02d}_pass"
            save_video = True
            success_video_count += 1
        elif not r["success"]:
            tag = f"ep{ep:02d}_fail"
            save_video = True

        if save_video:
            _save_episode_videos(r, tag, policy_video_dir, witness_video_dir)

        # Remove video buffers before appending to records (not CSV-serialisable)
        r.pop("_train_vid")
        r.pop("_wit_vid")
        records.append(r)

    # ── Completeness check ─────────────────────────────────────────────────────
    # A run that crashed or was killed partway through must NOT produce a CSV
    # or summary that looks complete. Fail loudly instead of writing partial
    # results under the filename a complete run would have used.
    if len(records) != args.n:
        print(f"\n[fatal] Recorded {len(records)} episodes but --n was "
              f"{args.n}. Refusing to write {args.out} or a run_summary.json "
              f"— a partial run must not silently look like a complete one.")
        sys.exit(1)

    # ── Aggregate and print summary ───────────────────────────────────────────
    n_success     = sum(r["success"] for r in records)
    reached_count = sum(1 for r in records if r["reached"])
    grasped_count = sum(1 for r in records if r["lifted"])
    placed_count  = sum(1 for r in records if r["placed"])
    tsr           = n_success / args.n
    mean_pss      = np.mean([r["pss"] for r in records])

    print("\n" + "─" * 40)
    print(f"Camera perturbation : {perturbation.label}")
    print(f"Scene  perturbation : {scene_perturbation.label}")
    print(f"Total episodes      : {args.n}")
    print(f"Successes           : {n_success}")
    print(f"Reached count       : {reached_count}")
    print(f"Grasped count       : {grasped_count}  (aka 'lifted' — see GraspDetector)")
    print(f"Placed count        : {placed_count}")
    print(f"Zero-shot TSR       : {n_success}/{args.n} = {tsr:.1%}")
    print(f"Mean PSS            : {mean_pss:.3f}")
    print("─" * 40)

    out_path = Path(args.out)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"\n[ok] Results saved → {out_path}")

    summary = build_run_summary(
        records, args, perturbation, scene_perturbation, out_path,
        matched_eval_set_info=(
            {"path": args.matched_eval_set, "content_hash": matched_data.get("content_hash")}
            if args.matched_eval_set and scene_perturbation.cube_position_preset is None
            else None
        ),
    )
    write_run_summary(summary, out_path.parent)

    print(f"\nPaper entry: {scene_perturbation.label} | "
          f"{perturbation.label} TSR = {tsr:.1%}  (N={args.n})")


if __name__ == "__main__":
    main()