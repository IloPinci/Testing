"""
two_camera_setup.py
-------------------
Scene construction, camera attachment, and success-check helpers.

build_environment() builds the table, cube, target zone, and optional
distractor spheres.  It returns (cube, target_zone, sphere_entities) so
callers can reposition spheres between episodes.

Camera architecture
-------------------
TRAINING cameras  (returned by attach_cameras)
    Rendered at IMG_RES (256×256). Subject to CameraPerturbation axes.
    Written into the LeRobot dataset.

WITNESS cameras   (returned by attach_witness_cameras_with_robot)
    Rendered at WITNESS_RES (1280×720). For visual debugging only;
    never written into the dataset.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import genesis as gs
import torch

from scene_params import (
    CUBE_COLORS, DEFAULT_CUBE_COLOR, IMG_RES, WITNESS_RES,
    CameraPerturbation, NOMINAL_PERTURBATION,
    SphereConfig,
)

_ROLE_TO_CAM_KEY = {"context": "camera1", "wrist": "camera2", "top": "camera3"}


# ══════════════════════════════════════════════════════════════════════════════
#  Environment
# ══════════════════════════════════════════════════════════════════════════════

def build_environment(
    scene,
    table_height: float,
    cube_color: str = DEFAULT_CUBE_COLOR,
    spheres: List[SphereConfig] | None = None,
):
    """
    Add the table, cube, target zone, and optional distractor spheres to
    the Genesis scene.

    Parameters
    ----------
    scene : gs.Scene
        The Genesis scene to add entities to.
    table_height : float
        Height of the table surface in world coordinates.
    cube_color : str
        Key into CUBE_COLORS.  Controls both the rendered colour and (via
        language_instruction_for) the language instruction.
    spheres : list[SphereConfig] or None
        Distractor spheres to add.  Each sphere is placed at
        (xy[0], xy[1], table_height + radius) so it rests on the table.
        Pass [] or None for no distractors.

    Returns
    -------
    cube : gs.Entity
    target_zone : gs.Entity
    sphere_entities : list[gs.Entity]
        Empty list when no spheres are requested.
    """
    if spheres is None:
        spheres = []

    if cube_color not in CUBE_COLORS:
        raise ValueError(
            f"cube_color must be one of {list(CUBE_COLORS)}, got {cube_color!r}"
        )
    cube_rgb = CUBE_COLORS[cube_color]["rgb"]

    table_width       = 1.0
    table_depth       = 1.0
    surface_thickness = 0.05
    leg_thickness     = 0.05

    # ── Tabletop ──────────────────────────────────────────────────────────────
    scene.add_entity(
        gs.morphs.Box(
            size=(table_width, table_depth, surface_thickness),
            pos=(0.0, 0.0, table_height - surface_thickness / 2.0),
            fixed=True,
        ),
        material=gs.materials.Rigid(friction=0.01, coup_friction=0.1, coup_softness=0.001),
    )

    # ── Table legs (four corners) ─────────────────────────────────────────────
    for x in [-table_width / 2 + leg_thickness / 2, table_width / 2 - leg_thickness / 2]:
        for y in [-table_depth / 2 + leg_thickness / 2, table_depth / 2 - leg_thickness / 2]:
            scene.add_entity(
                gs.morphs.Box(
                    size=(leg_thickness, leg_thickness, table_height - surface_thickness),
                    pos=(x, y, (table_height - surface_thickness) / 2.0),
                    fixed=True,
                )
            )

    # ── Cube (colour-parameterised) ───────────────────────────────────────────
    # High friction values prevent the cube from sliding during grasping
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.03, 0.03, 0.03),
            pos=(0.24, 0.0, table_height + 0.015),
        ),
        material=gs.materials.Rigid(
            rho=1000.0,
            friction=3.0,
            coup_friction=1.5,
            coup_softness=0.0001,
        ),
        surface=gs.surfaces.Rough(color=cube_rgb),
    )

    # ── Target zone (visual marker only — no collision) ───────────────────────
    target_zone = scene.add_entity(
        gs.morphs.Cylinder(
            radius=0.05,
            height=0.001,
            pos=(0.15, -0.15, table_height + 0.001),
            fixed=True,
            collision=False,
        ),
        surface=gs.surfaces.Rough(color=(0.0, 1.0, 0.0)),
    )

    # ── Distractor spheres ────────────────────────────────────────────────────
    sphere_entities: list = []
    for cfg in spheres:
        z        = table_height + cfg.radius   # rest on the table surface
        is_fixed = (cfg.mass_kg <= 0.0)
        entity   = scene.add_entity(
            gs.morphs.Sphere(
                radius=cfg.radius,
                pos=(cfg.xy[0], cfg.xy[1], z),
                fixed=is_fixed,
            ),
            material=gs.materials.Rigid(
                # Back-calculate density from mass and volume so Genesis gets
                # the right inertial properties
                rho=cfg.mass_kg / ((4.0 / 3.0) * np.pi * cfg.radius ** 3)
                    if not is_fixed else 1000.0,
                friction=cfg.friction,
            ),
            surface=gs.surfaces.Rough(color=cfg.color),
        )
        sphere_entities.append(entity)
        print(f"[scene] Added sphere '{cfg.label}' "
              f"at ({cfg.xy[0]:.3f}, {cfg.xy[1]:.3f}, {z:.3f})  "
              f"r={cfg.radius}  fixed={is_fixed}")

    return cube, target_zone, sphere_entities


# ══════════════════════════════════════════════════════════════════════════════
#  Nominal camera positions (single source of truth)
#
#  All camera setup functions call these helpers so that training cameras,
#  witness cameras, and the workspace sweep all share identical positions.
# ══════════════════════════════════════════════════════════════════════════════

def _context_nominal(table_height: float):
    """Third-person context camera: slightly to the side, looking down at the workspace."""
    pos    = np.array([0.4, -0.4, table_height + 0.55])
    lookat = np.array([0.2,  0.0, table_height + 0.05])
    return pos, lookat


def _top_nominal(table_height: float):
    """Top-down camera: directly above the workspace centre."""
    pos    = np.array([0.0, 0.0, table_height + 1.0])
    lookat = np.array([0.0, 0.0, table_height + 0.1])
    return pos, lookat


# ══════════════════════════════════════════════════════════════════════════════
#  Training cameras
# ══════════════════════════════════════════════════════════════════════════════

def attach_cameras(
    scene,
    so101,
    table_height: float,
    perturbation: CameraPerturbation = NOMINAL_PERTURBATION,
    context_offset: tuple = (0.0, 0.0, 0.0),
    context_tilt_deg: float = 0.0,
) -> dict:
    """
    Add the three training cameras to the scene and return them as a dict.

    Returns
    -------
    {"context": <cam>, "wrist": <cam>, "top": <cam>}

    Notes
    -----
    The wrist camera is physically attached to the gripper link and
    must be moved to its attach position via cam.move_to_attach() before
    each render call.
    """
    # ── Context camera ────────────────────────────────────────────────────────
    ctx_pos_nom, ctx_look_nom = _context_nominal(table_height)

    # Perturbation offset takes priority over the manual context_offset kwarg
    pert_ctx_offset = perturbation.position_offset_for("context")
    ctx_pos = (ctx_pos_nom + np.array(pert_ctx_offset, dtype=float)
               if any(v != 0.0 for v in pert_ctx_offset)
               else ctx_pos_nom + np.array(context_offset, dtype=float))

    pert_ctx_tilt = perturbation.tilt_deg_for("context")
    ctx_tilt_deg  = pert_ctx_tilt if pert_ctx_tilt != 0.0 else context_tilt_deg
    ctx_lookat    = _apply_tilt(ctx_pos, ctx_pos_nom, ctx_look_nom, ctx_tilt_deg)

    context_cam = scene.add_camera(
        res=IMG_RES, pos=tuple(ctx_pos), lookat=tuple(ctx_lookat), fov=60, GUI=False,
    )

    # ── Top camera ────────────────────────────────────────────────────────────
    top_pos_nom, top_look_nom = _top_nominal(table_height)
    top_offset = np.array(perturbation.position_offset_for("top"), dtype=float)
    top_pos    = top_pos_nom + top_offset
    top_lookat = _apply_tilt(top_pos, top_pos_nom, top_look_nom,
                              perturbation.tilt_deg_for("top"))

    top_cam = scene.add_camera(
        res=IMG_RES, pos=tuple(top_pos), lookat=tuple(top_lookat), fov=65, GUI=False,
    )

    # ── Wrist camera (attached to gripper link) ────────────────────────────────
    # The camera is added at an arbitrary world position; the attach() call below
    # fixes it relative to the gripper_link frame
    wrist_cam = scene.add_camera(
        res=IMG_RES, pos=(0.0, 0.0, 0.0), lookat=(1.0, 0.0, 0.0), fov=50, GUI=False,
    )
    gripper_link      = so101.get_link("gripper")
    # Base translation from gripper origin to camera mounting point (m)
    base_translation  = np.array([0.01, -0.145, -0.062])
    wrist_offset      = np.array(perturbation.position_offset_for("wrist"), dtype=float)
    final_translation = base_translation + wrist_offset

    # Build the 4×4 attach transform: 90° rotation around X to point the lens forward
    wrist_tilt_deg = perturbation.tilt_deg_for("wrist")
    theta = np.radians(90)
    c, s  = np.cos(theta), np.sin(theta)
    rotation = np.array([
        [1.0, 0.0,  0.0, 0.0],
        [0.0,   c,   -s, 0.0],
        [0.0,   s,    c, 0.0],
        [0.0, 0.0,  0.0, 1.0],
    ])
    if wrist_tilt_deg != 0.0:
        rotation = _rotate_4x4_z(rotation, wrist_tilt_deg)
    offset_T        = rotation.copy()
    offset_T[:3, 3] = final_translation
    wrist_cam.attach(gripper_link, offset_T)

    if perturbation.label != "nominal":
        print(f"[camera perturbation] {perturbation.label}")

    return {"context": context_cam, "wrist": wrist_cam, "top": top_cam}


# ══════════════════════════════════════════════════════════════════════════════
#  Witness cameras
# ══════════════════════════════════════════════════════════════════════════════

def attach_witness_cameras(scene, table_height: float) -> dict:
    """
    Add context and top witness cameras (no robot handle needed).

    Does not include a wrist camera; use attach_witness_cameras_with_robot()
    if a wrist witness is required.
    """
    ctx_pos, ctx_lookat = _context_nominal(table_height)
    top_pos, top_lookat = _top_nominal(table_height)

    w_context = scene.add_camera(
        res=WITNESS_RES, pos=tuple(ctx_pos), lookat=tuple(ctx_lookat), fov=60, GUI=False,
    )
    w_top = scene.add_camera(
        res=WITNESS_RES, pos=tuple(top_pos), lookat=tuple(top_lookat), fov=65, GUI=False,
    )
    print(f"[witness cameras] context + top at {WITNESS_RES[1]}×{WITNESS_RES[0]}")
    return {"witness_context": w_context, "witness_top": w_top}


def attach_witness_cameras_with_robot(scene, so101, table_height: float) -> dict:
    """Add full three-camera witness rig including an attached wrist camera."""
    ctx_pos, ctx_lookat = _context_nominal(table_height)
    top_pos, top_lookat = _top_nominal(table_height)

    w_context = scene.add_camera(
        res=WITNESS_RES, pos=tuple(ctx_pos), lookat=tuple(ctx_lookat), fov=60, GUI=False,
    )
    w_top = scene.add_camera(
        res=WITNESS_RES, pos=tuple(top_pos), lookat=tuple(top_lookat), fov=65, GUI=False,
    )
    w_wrist = scene.add_camera(
        res=WITNESS_RES, pos=(0.0, 0.0, 0.0), lookat=(1.0, 0.0, 0.0), fov=50, GUI=False,
    )
    # Same attach transform as the training wrist camera
    gripper_link = so101.get_link("gripper")
    theta = np.radians(90)
    c, s  = np.cos(theta), np.sin(theta)
    rotation = np.array([
        [1.0, 0.0,  0.0, 0.0],
        [0.0,   c,   -s, 0.0],
        [0.0,   s,    c, 0.0],
        [0.0, 0.0,  0.0, 1.0],
    ])
    offset_T        = rotation.copy()
    offset_T[:3, 3] = np.array([0.01, -0.145, -0.062])
    w_wrist.attach(gripper_link, offset_T)

    print(f"[witness cameras] context + wrist + top at {WITNESS_RES[1]}×{WITNESS_RES[0]}")
    return {"witness_context": w_context, "witness_wrist": w_wrist, "witness_top": w_top}


# ══════════════════════════════════════════════════════════════════════════════
#  Geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def _apply_tilt(cam_pos, nominal_cam_pos, nominal_lookat, tilt_deg):
    """Rotate the lookat direction around world Z by tilt_deg degrees."""
    if tilt_deg == 0.0:
        return nominal_lookat.copy()
    look_vec = nominal_lookat - nominal_cam_pos
    theta    = np.radians(tilt_deg)
    c, s     = np.cos(theta), np.sin(theta)
    rot_z    = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return cam_pos + rot_z @ look_vec


def _rotate_4x4_z(mat: np.ndarray, deg: float) -> np.ndarray:
    """Left-multiply a 4×4 homogeneous matrix by a rotation around Z."""
    theta = np.radians(deg)
    c, s  = np.cos(theta), np.sin(theta)
    rz = np.array([
        [ c, -s, 0.0, 0.0],
        [ s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    return rz @ mat


# ══════════════════════════════════════════════════════════════════════════════
#  Success / sub-goal checks
# ══════════════════════════════════════════════════════════════════════════════

def is_success(cube, target_zone, table_height, radius=0.05, min_height=0.005):
    """
    Return True if the cube is resting on the target zone.

    The cube must be within `radius` of the target centre in XY,
    above the table by at least `min_height`, and not airborne
    (i.e. not more than 5 cm above the table — the gripper has released it).
    """
    cube_pos    = cube.get_pos()
    target_pos  = target_zone.get_pos()
    xy_dist      = torch.norm(cube_pos[:2] - target_pos[:2])
    on_target    = xy_dist < radius
    above_table  = cube_pos[2] > (table_height + min_height)
    not_airborne = cube_pos[2] < (table_height + 0.05)
    return bool(on_target and above_table and not_airborne)


def check_sub_goals(so101, cube, target_zone, table_height,
                    radius=0.05, min_height=0.005, _latch: dict | None = None):
    """
    Evaluate intermediate sub-goals and compute a Partial Success Score (PSS).

    Sub-goals (cumulative):
      near_block  — gripper is within 5 cm of the cube (reached)
      lifted      — cube is elevated and co-moving with the gripper (grasped)
      placed      — cube is on the target zone (full success)

    PSS formula: 0.2 * reached + 0.4 * lifted + 1.0 * placed

    _latch : dict or None
        If provided, "lifted" is latched True once achieved and stays True
        even if the cube is later dropped.  This prevents a failed carry
        from zeroing out the sub-goal progress that was already earned.
    """
    cube_pos     = cube.get_pos()
    gripper_pos  = so101.get_link("gripper").get_pos()
    gripper_dist = torch.norm(gripper_pos - cube_pos).item()
    near_block   = gripper_dist < 0.05
    is_elevated  = cube_pos[2].item() > (table_height + 0.02)

    if _latch is not None:
        if is_elevated and near_block:
            _latch["lifted"] = True
        lifted = _latch["lifted"]
    else:
        lifted = bool(is_elevated and near_block)

    placed  = is_success(cube, target_zone, table_height, radius, min_height)
    partial = 0.2 * near_block + 0.4 * lifted + 1.0 * placed
    return {"near_block": near_block, "lifted": lifted, "placed": placed, "Sum": partial}


# ══════════════════════════════════════════════════════════════════════════════
#  Episode reset
# ══════════════════════════════════════════════════════════════════════════════

def reset_episode(
    scene,
    so101,
    cube,
    table_height: float,
    home_dofs=None,
    sphere_entities: list | None = None,
    sphere_configs:  list | None = None,
):
    """
    Reset the robot and cube to their home positions and step the scene to settle.

    Parameters
    ----------
    sphere_entities : list[gs.Entity] or None
        Genesis entity handles returned by build_environment().
    sphere_configs  : list[SphereConfig] or None
        Matching SphereConfig objects (same order as sphere_entities).
        When both are provided, each sphere is repositioned to its
        configured XY location so it rests on the table surface.
    """
    if home_dofs is None:
        home_dofs = np.zeros(so101.n_dofs)

    so101.set_dofs_position(home_dofs)
    so101.set_dofs_velocity(np.zeros(so101.n_dofs))
    so101.control_dofs_position(home_dofs)
    cube.set_pos(torch.tensor([0.24, 0.0, table_height + 0.015]))
    cube.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))

    if sphere_entities and sphere_configs:
        for entity, cfg in zip(sphere_entities, sphere_configs):
            z = table_height + cfg.radius
            entity.set_pos(np.array([cfg.xy[0], cfg.xy[1], z]))
            entity.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))

    for _ in range(10):
        scene.step()

    return home_dofs


# ══════════════════════════════════════════════════════════════════════════════
#  Main (demonstration collection CLI)
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Collect demonstrations with a CLI-configurable seed and scene
    perturbation, instead of a hardcoded seed and a hand-edited variable.

    Examples:
        # Nominal, seed 0 (default dataset-generation seed for the paper)
        python two_camera_setup.py --seed 0 --n_episodes 200 \\
            --output_dir demos/nominal/lerobot/

        # A second independent dataset for a second training seed (S4)
        python two_camera_setup.py --seed 1 --n_episodes 200 \\
            --output_dir demos/seed1/lerobot/ \\
            --repo_id local/genesis_pickplace_seed1
    """
    import argparse
    from oracle_direct import collect_demonstrations
    from scene_params import ScenePerturbation, SPHERE_PRESETS, NOMINAL_SCENE

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None,
                        help="Dataset-generation seed (ST5). REQUIRED for any "
                             "run you intend to report or reproduce — omitting "
                             "it produces an unseeded, non-reproducible dataset.")
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--output_dir", default=None,
                        help="Explicit dataset output dir. Either this OR "
                             "--run_root is required. --out wins if both given.")
    parser.add_argument("--run_root", default=None,
                        help="Short campaign tag, e.g. 'icra-final'. Derives "
                             "the dataset path as runs/<run_root>/data/"
                             "<dataset_slug>/lerobot/ (see run_naming.py), "
                             "isolated from any prior demos/ tree.")
    parser.add_argument("--dataset_id", default=None,
                        help="Short label for this dataset when using --run_root "
                             "(e.g. 'nominal', 'seed1'). Defaults to the scene "
                             "perturbation's own slug if omitted.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow writing into an already-populated dataset "
                             "directory. Without this, the script refuses.")
    parser.add_argument("--repo_id", default=None,
                        help="Defaults to local/genesis_pickplace_<scene_label>")
    parser.add_argument("--cube_color", default="red",
                        choices=["red", "blue", "green", "yellow"])
    parser.add_argument("--cube_pos", default=None,
                        help="Fixed spawn preset name, or omit for randomised spawn")
    parser.add_argument("--spheres", default="none",
                        help="Key into SPHERE_PRESETS")
    args = parser.parse_args()

    if args.output_dir is None and args.run_root is None:
        print("[error] Either --output_dir or --run_root is required.")
        sys.exit(1)

    if args.seed is None:
        print("[warn] --seed not set: this dataset will be UNSEEDED and "
              "not reproducible. Pass --seed explicitly for any run that "
              "will be reported (M8/ST5).")

    scene_pert = ScenePerturbation(
        cube_color=args.cube_color,
        cube_position_preset=args.cube_pos,
        spheres=SPHERE_PRESETS[args.spheres],
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = "/absolute/path/to/so101_new_calib.xml"

    # gs.init's seed controls Genesis-internal stochastic elements (if any);
    # kept equal to the dataset-generation seed for a single reproducible
    # number, falling back to a fixed default only when --seed is omitted.
    gs.init(backend=gs.gpu, seed=(args.seed if args.seed is not None else 42))

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=16),
        rigid_options=gs.options.RigidOptions(
            constraint_solver=gs.constraint_solver.Newton,
            iterations=100, tolerance=1e-9,
            constraint_timeconst=0.006,
            enable_self_collision=True, box_box_detection=True,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False, world_frame_size=0.5, show_cameras=False,
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
        cube_color=scene_pert.cube_color,
        spheres=scene_pert.spheres,
    )

    so101 = scene.add_entity(
        gs.morphs.MJCF(file=xml_path, pos=(0.0, 0.0, table_height))
    )

    training_cams = attach_cameras(scene, so101, table_height)
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
    # Higher friction on the gripper links improves grasp stability
    for link_name in ["gripper", "moving_jaw_so101_v1"]:
        so101.get_link(link_name).set_friction(5.0)

    def success_fn(c, tz, th):
        return is_success(c, tz, th, radius=0.05, min_height=0.005)

    def sub_goals_fn(robot, c, tz, th, latch=None):
        return check_sub_goals(robot, c, tz, th, radius=0.05, min_height=0.005,
                                _latch=latch)

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        from run_naming import run_root as _run_root, dataset_dir
        ds_slug    = args.dataset_id or scene_pert.label
        root       = _run_root(args.run_root)
        output_dir = str(dataset_dir(root, ds_slug))
        print(f"[run_root] {args.run_root} -> dataset slug '{ds_slug}' -> {output_dir}")

    from run_naming import refuse_if_exists
    refuse_if_exists(Path(output_dir), "Dataset output dir", args.overwrite)

    repo_id = args.repo_id or f"local/genesis_pickplace_{scene_pert.label}"

    collect_demonstrations(
        so101=so101, scene=scene,
        cameras=training_cams, witness_cameras=witness_cams,
        cube=cube, target_zone=target_zone,
        sphere_entities=sphere_entities,
        table_height=table_height,
        is_success_fn=success_fn, check_sub_goals_fn=sub_goals_fn,
        n_episodes=args.n_episodes,
        output_dir=output_dir,
        fps=10,
        repo_id=repo_id,
        scene_perturbation=scene_pert,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()