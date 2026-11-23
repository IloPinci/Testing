"""
zeroshot_eval.py
----------------
Run N rollouts of a SmolVLA checkpoint in the Genesis scene under one
perturbation condition and one execution horizon, and record per-episode
outcomes, sub-goal diagnostics, trajectories and (optionally) videos.

What changed in protocol v2 (see docs/EXPERIMENT_PROTOCOL.md for the why)
-------------------------------------------------------------------------
SPEED
  * The policy only looks at the observation when it re-plans (every
    --n_action_steps steps). Cameras are now rendered ONLY on those steps
    (k=50: ~2 renders per episode instead of 60; k=10: 6 instead of 60).
    The action chunk is managed here via policy.predict_action_chunk(), which
    is exactly what select_action() does internally. `--backend select_action`
    keeps the old every-step path as a reference for an equivalence check.
  * Blanked cameras are not rendered at all (the policy gets the same blank).
  * Witness cameras are only CREATED when some episode is recorded, and only
    rendered for recorded episodes. Frames are streamed to disk, not kept in RAM.
  * Episodes end early once strict success has held for 3 steps, or the cube
    has fallen off the table.

VIDEOS
  * --video_episodes "0-4" records those episode indices in EVERY condition
    (same matched initial states), so videos line up across conditions/horizons
    for side-by-side figures (make_comparison_video.py).
  * --episodes "37,52" re-runs only those episodes (seeded identically) to
    render failures after a fast video-free sweep; --reference_csv checks
    the replay reproduces the original outcomes.

METRICS
  * success_strict (released, at rest, on target) and success_carry (the old
    criterion) are both recorded; --success_metric picks which fills the
    `success` column (default strict). Raw sub-goal latches, a failure
    taxonomy, time-to-event and continuous distances are logged; legacy
    columns (reached/lifted/placed/pss) keep their exact old semantics.
  * trajectories.npz: per-step joint state, action, jaw/cube position, flags.
  * --probe_nominal: at every re-plan, also run the policy on the UNPERTURBED
    camera views (same state, same noise) and log how far the actions diverge.

PROVENANCE
  * run_summary.json records code hashes, git commit, package versions, GPU,
    checkpoint fingerprint, matched-set hash, all thresholds and CLI args.
    analyze_paper.py refuses to compare runs whose provenance differs.

Usage
-----
    python zeroshot_eval.py --run_root paper-v2 --checkpoint $CKPT \\
        --matched_eval_set $MATCH --n 100 --n_action_steps 10 \\
        --tilt_target wrist --tilt_deg 10 --video_episodes 0-4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from scene_params import (
    IMG_RES, DEFAULT_CUBE_COLOR, CUBE_COLORS,
    CameraPerturbation, NOMINAL_PERTURBATION,
    ScenePerturbation, NOMINAL_SCENE,
    CUBE_POSITIONS, SPHERE_PRESETS, BLACKOUT_MODES, BLANK_FILLS,
)
from episode_metrics import (
    EpisodeMonitor, MetricConfig, FAILURE_MODES, parse_index_spec, sphere_geometry,
)
from video_utils import (
    VideoRecorder, policy_strip, presentation_grid, annotate_grid, append_manifest,
)

import genesis as gs

try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
except ImportError:
    print("[error] lerobot not installed or SmolVLA policy not found.")
    sys.exit(1)

import importlib.util


def _import_from(module_name, file_path):
    """Import a module from an explicit file path (used for sibling scripts)."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_script_dir = Path(__file__).resolve().parent
_setup = _import_from("two_camera_setup", _script_dir / "two_camera_setup.py")
build_environment = _setup.build_environment
attach_cameras = _setup.attach_cameras
attach_witness_cameras_with_robot = _setup.attach_witness_cameras_with_robot

PROTOCOL_VERSION = "2.0"
# Files whose content can change an episode's OUTCOME. Their hash is stored
# in run_summary.json; analyze_paper.py only compares runs with equal hashes.
OUTCOME_SOURCE_FILES = ("zeroshot_eval.py", "episode_metrics.py",
                        "scene_params.py", "two_camera_setup.py")

# ══════════════════════════════════════════════════════════════════════════════
#  Rollout constants
# ══════════════════════════════════════════════════════════════════════════════
SIM_DT       = 0.01  # must match SimOptions(dt=...) in build_scene
HOLD_STEPS   = 10    # simulator steps per policy step -> 10 Hz, = dataset FPS
POLICY_DT    = SIM_DT * HOLD_STEPS
ARM_DOFS     = np.arange(5)
GRIPPER_DOF  = np.array([5])
JAW_LINK     = "moving_jaw_so101_v1"     # same link the oracle's IK targets
ROLES        = (("context", "camera1"), ("wrist", "camera2"), ("top", "camera3"))


# ══════════════════════════════════════════════════════════════════════════════
#  Policy wrapper
# ══════════════════════════════════════════════════════════════════════════════

class SmolVLAAgent:
    """
    SmolVLA + the checkpoint's own pre/post-processor pipeline.

    backend="chunk" (default): this class owns the action queue. When it is
    empty, it builds the observation, calls policy.predict_action_chunk()
    with explicitly sampled flow-matching noise, post-processes the first
    n_action_steps actions and queues them. This is the same computation as
    policy.select_action() (which also only uses the observation when its
    queue is empty), but it lets the caller skip rendering on steps where
    the observation would be ignored, and it makes re-plan steps explicit.

    The noise is drawn with policy.model.sample_noise() using the same shape
    sample_actions() would use, so the RNG stream is consumed identically to
    the select_action path. Each forward pass gets a CLONE of the noise:
    LeRobot's Euler integrator updates x_t (= the noise tensor) in place.

    backend="select_action": the previous behaviour (observation built and
    pre-processed every step). Kept as a reference: a pilot run with both
    backends must give identical CSVs (see the protocol).
    """

    def __init__(self, checkpoint: str, device: str, language_instruction: str,
                 n_action_steps: int, backend: str = "chunk"):
        self.instruction = language_instruction
        self.backend = backend
        print(f"[info] Loading SmolVLA checkpoint: {checkpoint}")
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint, device=device)
        cfg = self.policy.config
        if not 1 <= n_action_steps <= cfg.chunk_size:
            raise ValueError(f"n_action_steps ({n_action_steps}) must be in [1, {cfg.chunk_size}]")
        cfg.n_action_steps = n_action_steps
        self.k = n_action_steps
        self.chunk_size = cfg.chunk_size
        self.policy.eval()
        self.device = next(self.policy.parameters()).device
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=checkpoint,
        )
        if backend == "chunk" and not hasattr(self.policy, "predict_action_chunk"):
            raise RuntimeError("This lerobot version has no predict_action_chunk(); "
                               "use --backend select_action.")
        print(f"[info] chunk_size={cfg.chunk_size}  n_action_steps={cfg.n_action_steps}  "
              f"backend={backend}  empty_cameras={getattr(cfg, 'empty_cameras', 'n/a')}")
        self._queue: list[np.ndarray] = []
        self._step = 0

    def reset(self):
        self.policy.reset()
        self._queue = []
        self._step = 0

    def will_replan(self) -> bool:
        """True if the NEXT act() call consults the observation."""
        if self.backend == "chunk":
            return len(self._queue) == 0
        return self._step % self.k == 0

    def needs_observation(self) -> bool:
        return self.backend == "select_action" or len(self._queue) == 0

    def _raw_obs(self, images: dict, joint_state: np.ndarray) -> dict:
        obs = {}
        for key, arr in images.items():
            if arr is None:                      # absent camera: key omitted
                continue
            t = torch.from_numpy(np.ascontiguousarray(arr)).float() / 255.0
            obs[f"observation.images.{key}"] = t.permute(2, 0, 1)
        obs["observation.state"] = torch.from_numpy(joint_state.copy()).float()
        obs["task"] = self.instruction
        return obs

    def _sample_noise(self) -> torch.Tensor:
        cfg = self.policy.config
        return self.policy.model.sample_noise((1, cfg.chunk_size, cfg.max_action_dim), self.device)

    @torch.no_grad()
    def _chunk(self, images: dict, joint_state: np.ndarray, noise: torch.Tensor) -> np.ndarray:
        batch = self.preprocessor(self._raw_obs(images, joint_state))
        chunk = self.policy.predict_action_chunk(batch, noise=noise.clone())
        acts = [self.postprocessor(chunk[:, t, :]).squeeze(0).float().cpu().numpy()
                for t in range(self.k)]
        return np.stack(acts).astype(np.float32)

    @torch.no_grad()
    def act(self, images: dict, joint_state: np.ndarray, probe_images: dict | None = None):
        """Return (action (6,), info). info["replanned"] marks a new chunk."""
        info = {"replanned": False, "infer_ms": None, "probe_div": None, "probe_grip_div": None}

        if self.backend == "select_action":
            replan = self._step % self.k == 0
            t0 = time.perf_counter()
            batch = self.preprocessor(self._raw_obs(images, joint_state))
            a = self.policy.select_action(batch)
            if a.ndim == 3:
                a = a[:, 0, :]
            a = self.postprocessor(a).squeeze(0).float().cpu().numpy()
            if replan:
                info.update(replanned=True, infer_ms=1e3 * (time.perf_counter() - t0))
            self._step += 1
            return a.astype(np.float32), info

        if not self._queue:
            noise = self._sample_noise()
            t0 = time.perf_counter()
            acts = self._chunk(images, joint_state, noise)
            info.update(replanned=True, infer_ms=1e3 * (time.perf_counter() - t0))
            if probe_images is not None:
                # Counterfactual: same state, same noise, unperturbed cameras.
                ref = self._chunk(probe_images, joint_state, noise)
                info["probe_div"] = float(np.linalg.norm(acts[:, :5] - ref[:, :5], axis=1).mean())
                info["probe_grip_div"] = float(np.abs(acts[:, 5] - ref[:, 5]).mean())
            self._queue = list(acts)
        a = self._queue.pop(0)
        self._step += 1
        return a, info


# ══════════════════════════════════════════════════════════════════════════════
#  Scene
# ══════════════════════════════════════════════════════════════════════════════

def build_scene(xml_path: str, perturbation: CameraPerturbation = NOMINAL_PERTURBATION,
                scene_perturbation: ScenePerturbation = NOMINAL_SCENE,
                with_witness: bool = False, with_probe: bool = False):
    """
    Returns (scene, so101, cube, target_zone, sphere_entities,
             training_cams, witness_cams|None, probe_cams|None, table_height).
    Extra cameras never affect physics or the training cameras' images.
    """
    gs.init(backend=gs.gpu, seed=0)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=SIM_DT, substeps=16),
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
    so101 = scene.add_entity(gs.morphs.MJCF(file=xml_path, pos=(0.0, 0.0, table_height)))

    training_cams = attach_cameras(scene, so101, table_height, perturbation=perturbation)
    witness_cams = attach_witness_cameras_with_robot(scene, so101, table_height) if with_witness else None
    probe_cams = (attach_cameras(scene, so101, table_height, perturbation=NOMINAL_PERTURBATION)
                  if with_probe else None)

    scene.build()

    so101.set_dofs_kp(np.array([4000, 4000, 3000, 2000, 2000]), dofs_idx_local=ARM_DOFS)
    so101.set_dofs_kv(np.array([400, 400, 300, 200, 200]), dofs_idx_local=ARM_DOFS)
    so101.set_dofs_kp(np.array([800.0]), dofs_idx_local=GRIPPER_DOF)
    so101.set_dofs_kv(np.array([80.0]), dofs_idx_local=GRIPPER_DOF)
    so101.set_dofs_force_range(lower=np.array([-100.0]), upper=np.array([100.0]),
                               dofs_idx_local=GRIPPER_DOF)
    for link_name in ["gripper", JAW_LINK]:
        so101.get_link(link_name).set_friction(5.0)

    return (scene, so101, cube, target_zone, sphere_entities,
            training_cams, witness_cams, probe_cams, table_height)


def resolve_spawns(n: int, scene_perturbation: ScenePerturbation,
                   matched_pairs: list | None, seed: int):
    """
    Initial (cube_xy, target_xy) for episode indices 0..n-1. Computed up
    front (identically to the old per-episode draws) so that running a
    SUBSET of episodes (--episodes) reproduces the same initial states.
    """
    rng = np.random.default_rng(seed)
    spawns = []
    for ep in range(n):
        if matched_pairs is not None and scene_perturbation.cube_position_preset is None:
            p = matched_pairs[ep]
            spawns.append((np.array(p["cube_xy"], float), np.array(p["target_xy"], float)))
        else:
            spawns.append(scene_perturbation.resolve_spawn(rng))
    return spawns


def reset_episode(scene, so101, cube, target_zone, table_height, cube_xy, target_xy,
                  scene_perturbation: ScenePerturbation = NOMINAL_SCENE,
                  sphere_entities: list | None = None):
    """Reset robot, cube, target zone and spheres, then settle for 20 steps."""
    home = np.zeros(so101.n_dofs)
    so101.set_dofs_position(home)
    so101.set_dofs_velocity(np.zeros(so101.n_dofs))
    so101.control_dofs_position(home)

    cube.set_pos(np.array([cube_xy[0], cube_xy[1], table_height + 0.015]))
    cube.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))
    target_zone.set_pos(np.array([target_xy[0], target_xy[1], table_height + 0.001]))

    if sphere_entities and scene_perturbation.spheres:
        for entity, cfg_s in zip(sphere_entities, scene_perturbation.spheres):
            entity.set_pos(np.array([cfg_s.xy[0], cfg_s.xy[1], table_height + cfg_s.radius]))
            entity.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))

    for _ in range(20):
        so101.set_dofs_velocity(np.zeros(so101.n_dofs))
        so101.control_dofs_position(home)
        scene.step()


# ══════════════════════════════════════════════════════════════════════════════
#  Rollout
# ══════════════════════════════════════════════════════════════════════════════

def _render(cam, is_wrist: bool) -> np.ndarray:
    if is_wrist:
        cam.move_to_attach()        # update attached camera pose from gripper FK
    rgb, _, _, _ = cam.render()
    return np.asarray(rgb)


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def observe(training_cams: dict, perturbation: CameraPerturbation, blank_rng):
    """What the policy receives: {cam_key: image | None(absent)} and a per-role status."""
    images, status = {}, {}
    for role, key in ROLES:
        if perturbation.camera_is_absent(key):
            images[key], status[role] = None, "absent"
        elif perturbation.camera_is_zeroed(key):
            images[key] = perturbation.blank_for((IMG_RES[1], IMG_RES[0], 3), blank_rng)
            status[role] = "blank"
        else:
            images[key], status[role] = _render(training_cams[role], role == "wrist"), "ok"
    return images, status


class TrajLog:
    """Per-step arrays for every episode of the run -> trajectories.npz."""
    FIELDS = ("episode", "t", "replan", "qpos", "action", "jaw", "cube", "gripper_q",
              "d_grasp", "reached", "grasped", "carry", "strict", "probe_div")

    def __init__(self):
        self.rows = {f: [] for f in self.FIELDS}
        self.targets = {}

    def add(self, **kw):
        for f in self.FIELDS:
            self.rows[f].append(kw[f])

    def save(self, path: Path):
        arr = {f: np.asarray(v, dtype=np.float32 if f in ("qpos", "action", "jaw", "cube",
                                                          "gripper_q", "d_grasp", "probe_div")
                             else None)
               for f, v in self.rows.items()}
        eps = sorted(self.targets)
        arr["target_episode"] = np.asarray(eps, dtype=np.int32)
        arr["target_xyz"] = np.asarray([self.targets[e] for e in eps], dtype=np.float32)
        np.savez_compressed(path, **arr)


def run_rollout(agent: SmolVLAAgent, scene, so101, cube, target_zone, training_cams: dict,
                table_height: float, perturbation: CameraPerturbation, *,
                ep: int, max_steps: int, metric_cfg: MetricConfig, blank_rng,
                traj: TrajLog | None = None, recorder: VideoRecorder | None = None,
                witness_cams: dict | None = None, witness_every: int = 4,
                probe_cams: dict | None = None, title: str = "") -> dict:
    agent.reset()
    monitor = EpisodeMonitor(table_height, metric_cfg)
    target_p = _np(target_zone.get_pos())
    if traj is not None:
        traj.targets[ep] = target_p
    recording = recorder is not None and witness_cams is not None
    witness_fps = 1.0 / (SIM_DT * witness_every)
    chips = {"REACH": False, "GRASP": False, "OVER TARGET": False, "SUCCESS": False}

    reason, steps, sim_step = "timeout", max_steps, 0
    n_replans, infer_ms, probe_divs, probe_grip = 0, [], [], []
    vis_rng = np.random.default_rng(10_000 + ep)   # blanks drawn only for video frames
    last_policy_view, last_grid = None, None

    for step in range(max_steps):
        # ── Observation (rendered only when the policy will use it, or when recording)
        images = {}
        status = {}
        policy_looks = agent.needs_observation()
        if policy_looks or recording:
            # Video-only frames draw "noise" blanks from a separate RNG so that
            # recording an episode never changes what the policy receives.
            images, status = observe(training_cams, perturbation,
                                     blank_rng if policy_looks else vis_rng)
        probe_images = None
        if probe_cams is not None and agent.will_replan():
            probe_images = {key: _render(probe_cams[role], role == "wrist") for role, key in ROLES}

        qpos = _np(so101.get_dofs_position())
        action, info = agent.act(images, qpos[:6], probe_images)
        if info["replanned"]:
            n_replans += 1
            if info["infer_ms"] is not None:
                infer_ms.append(info["infer_ms"])
            if info["probe_div"] is not None:
                probe_divs.append(info["probe_div"])
                probe_grip.append(info["probe_grip_div"])

        if recording:
            last_policy_view = policy_strip({role: images[key] for role, key in ROLES}, status)
            recorder.add("policy/combined", last_policy_view, fps=1.0 / POLICY_DT)

        # ── Execute the action for HOLD_STEPS simulator steps
        arm_target, gripper_target = action[:5], np.array([action[5]])
        for _ in range(HOLD_STEPS):
            so101.control_dofs_position(arm_target, dofs_idx_local=ARM_DOFS)
            so101.control_dofs_position(gripper_target, dofs_idx_local=GRIPPER_DOF)
            scene.step()
            sim_step += 1
            if recording and sim_step % witness_every == 0:
                wf = {r: _render(witness_cams[f"witness_{r}"], r == "wrist")
                      for r in ("context", "top", "wrist")}
                for r, f in wf.items():
                    recorder.add(f"witness/{r}", f, fps=witness_fps)
                last_grid = presentation_grid(wf["context"], wf["top"], wf["wrist"],
                                              last_policy_view, info["replanned"])
                recorder.add("presentation",
                             annotate_grid(last_grid, title,
                                           f"ep {ep}   t = {sim_step * SIM_DT:4.1f} s",
                                           chips, replan=info["replanned"]),
                             fps=witness_fps)

        # ── Sub-goals / termination
        jaw = _np(so101.get_link(JAW_LINK).get_pos())
        cube_p = _np(cube.get_pos())
        grip_q = float(_np(so101.get_dofs_position())[5])
        m = monitor.update(jaw, cube_p, target_p, grip_q)
        chips = {"REACH": monitor.lat["reached"], "GRASP": monitor.lat["grasped"],
                 "OVER TARGET": monitor.lat["carry"], "SUCCESS": m["strict"]}
        if traj is not None:
            traj.add(episode=ep, t=step, replan=info["replanned"], qpos=qpos[:6], action=action,
                     jaw=jaw, cube=cube_p, gripper_q=grip_q, d_grasp=m["d_grasp"],
                     reached=m["reached"], grasped=m["grasped"], carry=m["carry"],
                     strict=m["strict"],
                     probe_div=np.nan if info["probe_div"] is None else info["probe_div"])
        stop = monitor.should_stop()
        if stop is not None:
            reason, steps = stop, step + 1
            break

    result = monitor.summary(reason)
    result.update({
        "steps": steps,
        "n_replans": n_replans,
        "infer_ms_mean": round(float(np.mean(infer_ms)), 2) if infer_ms else None,
        "probe_div_mean": round(float(np.mean(probe_divs)), 5) if probe_divs else None,
        "probe_div_first": round(float(probe_divs[0]), 5) if probe_divs else None,
        "probe_grip_div_mean": round(float(np.mean(probe_grip)), 5) if probe_grip else None,
    })

    if recording and last_grid is not None:     # 1.5 s end card with the outcome
        outcome = "success" if result["success_strict"] else f"fail: {result['failure_mode']}"
        end = annotate_grid(last_grid, title, f"ep {ep}   end ({reason})", chips,
                            replan=False, outcome=outcome)
        for _ in range(int(1.5 * witness_fps)):
            recorder.add("presentation", end, fps=witness_fps)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Provenance and summary
# ══════════════════════════════════════════════════════════════════════════════

def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _pkg_version(name: str):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def collect_provenance(args) -> dict:
    files = {f: _sha256_file(_script_dir / f)[:16]
             for f in OUTCOME_SOURCE_FILES if (_script_dir / f).exists()}
    code_hash = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]

    def _git(*cmd):
        try:
            return subprocess.run(["git", *cmd], cwd=_script_dir, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return None

    ckpt = {"path": args.checkpoint}
    p = Path(args.checkpoint)
    if p.is_dir():
        for fname in ("model.safetensors", "config.json"):
            if (p / fname).exists():
                ckpt[f"sha256_{fname}"] = _sha256_file(p / fname)[:16]
    else:
        ckpt["note"] = "hub id or missing local path; pin a revision for reproducibility"

    return {
        "protocol_version": PROTOCOL_VERSION,
        "outcome_code_hash": code_hash,
        "outcome_source_sha256": files,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "checkpoint_fingerprint": ckpt,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "lerobot": _pkg_version("lerobot"), "genesis": _pkg_version("genesis-world"),
                     "numpy": np.__version__},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "hostname": socket.gethostname(),
        "argv": sys.argv,
    }


def build_run_summary(records, args, perturbation, scene_perturbation, csv_path,
                      provenance, metric_cfg, matched_info, spawn_source, timing,
                      video_eps, replay_check, chunk_size):
    n = len(records)
    count = lambda k: int(sum(1 for r in records if r[k]))
    modes = {m: int(sum(1 for r in records if r["failure_mode"] == m)) for m in FAILURE_MODES}
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **provenance,
        "checkpoint": args.checkpoint,
        "checkpoint_id": args.checkpoint_id or Path(args.checkpoint).name,
        "train_seed": args.train_seed,
        "env_seed": args.seed,
        "n_action_steps": args.n_action_steps,
        "chunk_size": chunk_size,
        # "open loop" here = one observation per full chunk. With demos of
        # ~48 steps and chunk_size 50 this is effectively one look per episode.
        "loop_mode": "open_loop" if args.n_action_steps >= chunk_size else "closed_loop",
        "backend": args.backend,
        "max_steps": args.max_steps,
        "success_metric": args.success_metric,
        "metric_config": metric_cfg.to_dict(),
        "matched_eval_set": matched_info,
        "spawn_source": spawn_source,
        "dataset_dir": args.dataset_dir,
        "csv_path": str(csv_path),
        "camera_perturbation": perturbation.label,
        "camera_perturbation_params": perturbation.as_dict(),
        "scene_perturbation": scene_perturbation.label,
        "scene_perturbation_params": scene_perturbation.as_dict(),
        "language_instruction": scene_perturbation.language_instruction,
        "episodes_run": [r["episode"] for r in records],
        "total_episodes": n,
        "successes": count("success"),
        "tsr": count("success") / n,
        "tsr_strict": count("success_strict") / n,
        "tsr_carry": count("success_carry") / n,
        "p_reached_raw": count("reached_raw") / n,
        "p_grasped_raw": count("grasped_raw") / n,
        "pushed_successes": count("pushed_success"),
        "failure_modes": modes,
        "consistency_violations": {
            "grasp_without_reach": count("violation_grasp_without_reach"),
            "success_without_grasp": count("violation_success_without_grasp"),
        },
        # legacy aggregates (old semantics)
        "reached_count": count("reached"),
        "lifted_count": count("lifted"),
        "grasped_count": count("lifted"),
        "placed_count": count("placed"),
        "mean_pss": round(float(np.mean([r["pss"] for r in records])), 6),
        "timing": timing,
        "video_episodes": sorted(video_eps),
        "replay_check": replay_check,
    }


def write_run_summary(summary, out_dir: Path) -> Path:
    path = out_dir / "run_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[ok] Run summary saved → {path}")
    return path


def _replay_check(records, reference_csv: str) -> dict:
    ref = {}
    with open(reference_csv) as f:
        for row in csv.DictReader(f):
            ref[int(row["episode"])] = row
    mismatches = []
    for r in records:
        rr = ref.get(r["episode"])
        if rr is None:
            mismatches.append({"episode": r["episode"], "reason": "not in reference"})
            continue
        for key in ("success_strict", "success_carry", "failure_mode", "steps"):
            if key in rr and str(r[key]) != rr[key]:
                mismatches.append({"episode": r["episode"], "field": key,
                                   "replay": r[key], "reference": rr[key]})
    if mismatches:
        print(f"[warn] Replay differs from {reference_csv} in {len(mismatches)} field(s): "
              f"{mismatches[:5]} — simulation is not bit-reproducible across processes; "
              f"say so in the paper and treat replays as illustrative.")
    else:
        print(f"[ok] Replay reproduces {reference_csv} for all {len(records)} episode(s).")
    return {"reference_csv": reference_csv, "n_mismatches": len(mismatches),
            "mismatches": mismatches[:50]}


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xml", default=None,
                   help="SO-101 MJCF. Default: <script dir>/../so101_arm/so101_new_calib.xml")
    p.add_argument("--checkpoint", default="lerobot/smolvla_base")
    p.add_argument("--checkpoint_id", default=None,
                   help="Label for this checkpoint (e.g. seed0). Required once you have >1.")
    p.add_argument("--train_seed", type=int, default=None)
    p.add_argument("--n", type=int, default=100,
                   help="Episode index range 0..n-1 (and the matched-set prefix used).")
    p.add_argument("--episodes", default=None,
                   help="Run only these indices within 0..n-1, e.g. '3,17,40-42' (replay). "
                        "Initial states and noise seeds are identical to the full run.")
    p.add_argument("--reference_csv", default=None,
                   help="With --episodes: check the replayed outcomes against this CSV.")
    p.add_argument("--out", default=None, help="Explicit output CSV path (wins over --run_root).")
    p.add_argument("--run_root", default=None,
                   help="Campaign tag; output goes to runs/<tag>/eval/k<K>/<slug>/.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=42,
                   help="Env seed: spawn sampling (if no matched set) and per-episode "
                        "flow-matching noise seed = seed*100000 + episode.")
    p.add_argument("--dataset_dir", default="demos/lerobot/",
                   help="Unused (kept for CLI compatibility); stats come from the checkpoint.")
    p.add_argument("--matched_eval_set", default=None)
    p.add_argument("--n_action_steps", type=int, required=True,
                   help="Actions executed per inference. REQUIRED (no silent default): "
                        "50 = open loop for chunk_size 50, 10 = closed loop at 1 Hz re-planning.")
    p.add_argument("--max_steps", type=int, default=60,
                   help="Policy steps per episode (10 Hz). Demos are ~48 steps.")
    p.add_argument("--backend", default="chunk", choices=["chunk", "select_action"])
    p.add_argument("--success_metric", default="strict", choices=["strict", "carry"],
                   help="Which criterion fills the `success` column (both are always logged).")
    p.add_argument("--reach_radius", type=float, default=MetricConfig.reach_radius,
                   help="m from jaw origin to the oracle grasp point. Calibrate on oracle "
                        "rollouts BEFORE the campaign, then freeze.")
    # videos
    p.add_argument("--video_episodes", default="none",
                   help="Episode indices to record: 'none', 'all', '0-4', '0,7,12'.")
    p.add_argument("--no_videos", action="store_true", help="Alias for --video_episodes none.")
    p.add_argument("--witness_every", type=int, default=4,
                   help="Witness frame every N sim steps (4 -> 25 fps, real time).")
    p.add_argument("--video_crf", type=int, default=20)
    # diagnostics
    p.add_argument("--probe_nominal", action="store_true",
                   help="Also compute actions from unperturbed cameras at each re-plan "
                        "(same state + noise) and log the divergence. ~2x inference cost.")
    p.add_argument("--deterministic", action="store_true",
                   help="torch deterministic algorithms (slower). Use for the reproducibility check.")
    # camera perturbation
    p.add_argument("--blackout_mode", default="none", choices=sorted(BLACKOUT_MODES))
    p.add_argument("--blank_fill", default="black", choices=list(BLANK_FILLS))
    p.add_argument("--pos_target", default=None, choices=[None, "context", "wrist", "top"])
    p.add_argument("--pos_offset", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                   metavar=("DX", "DY", "DZ"),
                   help="Metres. World frame for context/top; GRIPPER-LINK frame for wrist.")
    p.add_argument("--tilt_target", default=None, choices=[None, "context", "wrist", "top"])
    p.add_argument("--tilt_deg", type=float, default=0.0)
    # scene perturbation
    p.add_argument("--cube_color", default=DEFAULT_CUBE_COLOR, choices=list(CUBE_COLORS.keys()))
    p.add_argument("--cube_pos", default=None, choices=[None] + list(CUBE_POSITIONS.keys()))
    p.add_argument("--spheres", default="none", choices=list(SPHERE_PRESETS.keys()))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.out is None and args.run_root is None:
        sys.exit("[error] Either --out or --run_root is required.")
    if args.no_videos:
        args.video_episodes = "none"
    if args.reference_csv and not args.episodes:
        sys.exit("[error] --reference_csv only makes sense with --episodes.")
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False

    perturbation = CameraPerturbation(
        blackout_mode=args.blackout_mode, position_target=args.pos_target,
        position_offset=tuple(args.pos_offset), tilt_target=args.tilt_target,
        tilt_deg=args.tilt_deg, blank_fill=args.blank_fill,
    )
    scene_perturbation = ScenePerturbation(
        cube_color=args.cube_color, cube_position_preset=args.cube_pos,
        spheres=SPHERE_PRESETS[args.spheres],
    )
    metric_cfg = MetricConfig(reach_radius=args.reach_radius, policy_dt=POLICY_DT)

    selected = parse_index_spec(args.episodes or "all", args.n)
    video_eps = set(parse_index_spec(args.video_episodes, args.n)) & set(selected)
    replay = args.episodes is not None

    # ── Output paths ────────────────────────────────────────────────────────
    from run_naming import run_root as _run_root, condition_slug, eval_paths, refuse_if_exists
    slug = condition_slug(perturbation, scene_perturbation)
    if args.out is not None:
        out_csv = Path(args.out)
    else:
        paths = eval_paths(_run_root(args.run_root), slug, args.n_action_steps)
        out_csv = (paths["dir"] / "replay" / "zeroshot.csv") if replay else paths["csv"]
        print(f"[run_root] {args.run_root} -> k{args.n_action_steps}/{slug} -> {out_csv}")
    out_dir = out_csv.parent
    video_root = out_dir / "videos"
    refuse_if_exists(out_csv, "Output CSV", args.overwrite)
    refuse_if_exists(out_dir / "run_summary.json", "Run summary", args.overwrite)
    if video_eps:
        refuse_if_exists(video_root / "manifest.json", "Video manifest", args.overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Initial states ──────────────────────────────────────────────────────
    matched_pairs, matched_info = None, None
    if args.matched_eval_set:
        md = json.loads(Path(args.matched_eval_set).read_text())
        matched_pairs = md["cube_target_pairs"]
        if args.n > len(matched_pairs):
            sys.exit(f"[error] --n {args.n} exceeds the matched set's {len(matched_pairs)} episodes.")
        if scene_perturbation.cube_position_preset is None:
            matched_info = {"path": args.matched_eval_set, "content_hash": md.get("content_hash")}
        else:
            print(f"[warn] --matched_eval_set ignored: --cube_pos fixes every spawn.")
    spawn_source = ("preset" if scene_perturbation.cube_position_preset
                    else "matched" if matched_info else "random")
    if spawn_source == "random":
        print("[warn] No matched eval set: episodes are NOT paired with other conditions.")
    spawns = resolve_spawns(args.n, scene_perturbation, matched_pairs, args.seed)

    print(f"[camera perturbation] {perturbation.label}")
    print(f"[scene  perturbation] {scene_perturbation.label}")
    print(f"[episodes] {len(selected)} of 0..{args.n - 1}   videos: {sorted(video_eps) or 'none'}")

    xml_path = (str((_script_dir / ".." / "so101_arm" / "so101_new_calib.xml").resolve())
                if args.xml is None else os.path.abspath(args.xml))
    (scene, so101, cube, target_zone, sphere_entities, training_cams, witness_cams,
     probe_cams, table_height) = build_scene(
        xml_path, perturbation, scene_perturbation,
        with_witness=bool(video_eps), with_probe=args.probe_nominal)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = SmolVLAAgent(args.checkpoint, device, scene_perturbation.language_instruction,
                         args.n_action_steps, backend=args.backend)
    if args.probe_nominal and args.backend != "chunk":
        sys.exit("[error] --probe_nominal requires --backend chunk.")

    provenance = collect_provenance(args)
    title = f"k={args.n_action_steps} | {slug}"
    run_info = {"condition": slug, "n_action_steps": args.n_action_steps,
                "camera_perturbation": perturbation.label,
                "scene_perturbation": scene_perturbation.label,
                "checkpoint_id": args.checkpoint_id or Path(args.checkpoint).name}

    partial_csv = out_csv.with_suffix(".partial.csv")
    partial_csv.unlink(missing_ok=True)
    traj = TrajLog()
    records, ep_times = [], []
    t_run = time.perf_counter()

    print(f"\n{'ep':>4}  {'result':>7}  {'mode':<26} {'steps':>5}  {'replans':>7}")
    print("─" * 60)
    for ep in selected:
        ep_seed = args.seed * 100_000 + ep
        torch.manual_seed(ep_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(ep_seed)
        blank_rng = np.random.default_rng(ep_seed)

        cube_xy, target_xy = spawns[ep]
        reset_episode(scene, so101, cube, target_zone, table_height, cube_xy, target_xy,
                      scene_perturbation, sphere_entities)
        recorder = VideoRecorder(video_root, f"ep{ep:03d}", crf=args.video_crf) if ep in video_eps else None

        t_ep = time.perf_counter()
        try:
            r = run_rollout(agent, scene, so101, cube, target_zone, training_cams, table_height,
                            perturbation, ep=ep, max_steps=args.max_steps, metric_cfg=metric_cfg,
                            blank_rng=blank_rng, traj=traj, recorder=recorder,
                            witness_cams=witness_cams, witness_every=args.witness_every,
                            probe_cams=probe_cams, title=title)
        except BaseException:
            if recorder is not None:
                recorder.abort()
            raise
        ep_times.append(time.perf_counter() - t_ep)

        primary = r["success_strict"] if args.success_metric == "strict" else r["success_carry"]
        row = {
            "episode": ep, "ep_seed": ep_seed,
            "checkpoint_id": args.checkpoint_id or Path(args.checkpoint).name,
            "train_seed": args.train_seed, "env_seed": args.seed,
            "n_action_steps": args.n_action_steps,
            "success": bool(primary),
            **r,
            "cube_x": round(float(cube_xy[0]), 5), "cube_y": round(float(cube_xy[1]), 5),
            "target_x": round(float(target_xy[0]), 5), "target_y": round(float(target_xy[1]), 5),
            "cube_xy": [float(cube_xy[0]), float(cube_xy[1])],
            "target_xy": [float(target_xy[0]), float(target_xy[1])],
            **sphere_geometry(scene_perturbation.spheres, cube_xy, target_xy),
            "sec": round(ep_times[-1], 2),
        }
        records.append(row)

        with open(partial_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            if f.tell() == 0:
                w.writeheader()
            w.writerow(row)

        if recorder is not None:
            files = recorder.finalize("pass" if row["success"] else "fail")
            append_manifest(video_root, {
                "episode": ep, "success": row["success"], "failure_mode": row["failure_mode"],
                "steps": row["steps"], "terminated_reason": row["terminated_reason"],
                "cube_xy": row["cube_xy"], "target_xy": row["target_xy"], "files": files,
            }, run_info)

        res = "PASS ✓" if row["success"] else "FAIL ✗"
        print(f"{ep:4d}  {res:>7}  {row['failure_mode']:<26} {row['steps']:>5}  "
              f"{row['n_replans']:>7}   ({ep_times[-1]:.1f}s)", flush=True)

    # ── Completeness: never write a partial run under the final name ────────
    if len(records) != len(selected):
        sys.exit(f"[fatal] Recorded {len(records)} episodes, expected {len(selected)}. "
                 f"Partial results left in {partial_csv}.")
    partial_csv.replace(out_csv)
    traj.save(out_dir / "trajectories.npz")

    timing = {"wall_sec": round(time.perf_counter() - t_run, 1),
              "sec_per_episode_mean": round(float(np.mean(ep_times)), 3),
              "sec_per_episode_recorded": (round(float(np.mean([t for e, t in zip(selected, ep_times)
                                                                 if e in video_eps])), 3)
                                           if video_eps else None)}
    replay_check = _replay_check(records, args.reference_csv) if args.reference_csv else None
    summary = build_run_summary(records, args, perturbation, scene_perturbation, out_csv,
                                provenance, metric_cfg, matched_info, spawn_source, timing,
                                video_eps, replay_check, agent.chunk_size)
    write_run_summary(summary, out_dir)

    n = len(records)
    print("\n" + "─" * 60)
    print(f"Condition           : k={args.n_action_steps}  {slug}")
    print(f"TSR (strict)        : {summary['tsr_strict']:.1%}")
    print(f"TSR (carry, legacy) : {summary['tsr_carry']:.1%}")
    print(f"P(reached) raw      : {summary['p_reached_raw']:.1%}")
    print(f"P(grasped) raw      : {summary['p_grasped_raw']:.1%}")
    print(f"Pushed successes    : {summary['pushed_successes']}")
    print(f"Failure modes       : {summary['failure_modes']}")
    print(f"Violations          : {summary['consistency_violations']}")
    print(f"Time                : {timing['wall_sec']} s ({timing['sec_per_episode_mean']} s/episode)")
    print("─" * 60)
    print(f"[ok] {out_csv}  (N={n}, success = {args.success_metric})")


if __name__ == "__main__":
    main()