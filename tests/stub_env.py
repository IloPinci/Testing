"""
CPU-only stand-ins for genesis + lerobot so the evaluation PIPELINE
(bookkeeping, metrics, CSV/summary/trajectory/video output, replay, backend
equivalence) can be tested without a GPU or the simulator. The physics is a
toy kinematic model; nothing here says anything about SmolVLA's behaviour.

The fake policy mirrors the parts of lerobot's SmolVLA that matter for the
evaluation code:
  * select_action() keeps a deque of n_action_steps actions and only builds a
    new chunk when it is empty;
  * sample_actions() integrates IN PLACE on the noise tensor it is given
    (lerobot does `x_t = noise; x_t += dt * v_t`).
"""
from __future__ import annotations

import sys
import types
from collections import deque

import numpy as np
import torch

GRASP_OFFSET = np.array([0.016, 0.0, 0.07])
TABLE = 0.8
RENDER_COUNT = {"n": 0}


# ── fake genesis ─────────────────────────────────────────────────────────────
def install_fake_genesis():
    gs = types.ModuleType("genesis")
    gs.gpu = "gpu"
    gs.init = lambda **kw: None
    sys.modules["genesis"] = gs
    return gs


# ── toy world ────────────────────────────────────────────────────────────────
class World:
    def __init__(self):
        self.q = np.zeros(6); self.q[5] = 0.8
        self.target_q = self.q.copy()
        self.cube = np.array([0.24, 0.0, TABLE + 0.015])
        self.target = np.array([0.15, -0.15, TABLE + 0.001])
        self.attached = False

    def jaw(self):
        # joints 0..2 ARE the jaw xyz offsets from a home point (toy kinematics)
        return np.array([0.20, 0.0, TABLE + 0.25]) + self.q[:3]

    def step(self):
        self.q += 0.35 * (self.target_q - self.q)
        grasp_pt = self.cube + GRASP_OFFSET
        if not self.attached and self.q[5] < 0.3 and np.linalg.norm(self.jaw() - grasp_pt) < 0.02:
            self.attached = True
        if self.attached and self.q[5] > 0.5:
            self.attached = False
        if self.attached:
            self.cube = self.jaw() - GRASP_OFFSET
        else:
            self.cube[2] = max(TABLE + 0.015, self.cube[2] - 0.02)   # falls back to the table


class T:
    def __init__(self, f): self.f = f
    def get_pos(self): return torch.tensor(self.f(), dtype=torch.float32)


class Robot:
    n_dofs = 6
    def __init__(self, w): self.w = w
    def set_dofs_position(self, q): self.w.q = np.array(q, float); self.w.target_q = self.w.q.copy()
    def set_dofs_velocity(self, v): pass
    def control_dofs_position(self, q, dofs_idx_local=None):
        if dofs_idx_local is None: self.w.target_q = np.array(q, float)
        else: self.w.target_q[np.asarray(dofs_idx_local)] = np.asarray(q, float)
    def get_dofs_position(self): return torch.tensor(self.w.q, dtype=torch.float32)
    def get_link(self, name): return T(self.w.jaw)


class Cube:
    def __init__(self, w): self.w = w
    def get_pos(self): return torch.tensor(self.w.cube, dtype=torch.float32)
    def set_pos(self, p): self.w.cube = np.array(p, float); self.w.attached = False
    def set_quat(self, q): pass


class Target:
    def __init__(self, w): self.w = w
    def get_pos(self): return torch.tensor(self.w.target, dtype=torch.float32)
    def set_pos(self, p): self.w.target = np.array(p, float)


class Scene:
    def __init__(self, w): self.w = w
    def step(self): self.w.step()


class Cam:
    """Image brightness encodes the cube position as seen by this camera, plus
    a camera-specific offset (a perturbed camera sees a shifted value)."""
    def __init__(self, w, role, offset=0):
        self.w, self.role, self.offset = w, role, offset
    def move_to_attach(self): pass
    def render(self, *a, **k):
        RENDER_COUNT["n"] += 1
        v = int(np.clip(128 + 400 * (self.w.cube[0] - 0.24) + self.offset, 1, 254))
        return np.full((256, 256, 3), v, np.uint8), None, None, None


def make_build_scene(world_box):
    def build_scene(xml_path, perturbation, scene_perturbation, with_witness=False, with_probe=False):
        w = World(); world_box["w"] = w
        # graded toy response: wrist image brightness shifts with the perturbation size
        off = 0.0
        if perturbation.tilt_target == "wrist":
            off += 1.6 * perturbation.tilt_deg
        if perturbation.position_target == "wrist":
            off += 400 * perturbation.position_offset[0]
        training = {"context": Cam(w, "context"), "wrist": Cam(w, "wrist", off), "top": Cam(w, "top")}
        witness = ({"witness_context": Cam(w, "c"), "witness_wrist": Cam(w, "w"),
                    "witness_top": Cam(w, "t")} if with_witness else None)
        probe = ({"context": Cam(w, "context"), "wrist": Cam(w, "wrist"), "top": Cam(w, "top")}
                 if with_probe else None)
        return (Scene(w), Robot(w), Cube(w), Target(w), [], training, witness, probe, TABLE)
    return build_scene


# ── fake lerobot ─────────────────────────────────────────────────────────────
class Cfg:
    chunk_size = 50
    n_action_steps = 50
    max_action_dim = 32
    empty_cameras = 0


class FakeModel:
    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)


def _plan(state, cube, target, bias, attached=False):
    """Scripted 'policy': the rest of a pick-and-place plan from the current
    state (phase-aware, so closed-loop re-planning makes progress)."""
    home = np.array([0.20, 0.0, TABLE + 0.25])
    def to_q(xyz): return xyz - home
    rest_z = TABLE + 0.015
    g = np.r_[cube[:2], rest_z] + GRASP_OFFSET + np.array([bias, 0, 0])
    place = np.r_[target[:2], rest_z] + GRASP_OFFSET
    wps = [(to_q(g + [0, 0, 0.06]), 0.8, 8), (to_q(g), 0.8, 6), (to_q(g), 0.1, 4),
           (to_q(g + [0, 0, 0.12]), 0.1, 6), (to_q(place + [0, 0, 0.12]), 0.1, 10),
           (to_q(place), 0.1, 6), (to_q(place), 0.8, 10)]
    jaw = home + state[:3]
    if attached:
        start = 3 if cube[2] < rest_z + 0.05 and np.linalg.norm(cube[:2] - target[:2]) > 0.05 else 4
        if np.linalg.norm(jaw - (place + [0, 0, 0.12])) < 0.02 or np.linalg.norm(jaw - place) < 0.03:
            start = 5
    elif np.linalg.norm(cube[:2] - target[:2]) < 0.05:
        start = 6                                     # already delivered: keep the gripper open
    elif np.linalg.norm(jaw - g) < 0.015:
        start = 2
    elif np.linalg.norm(jaw[:2] - g[:2]) < 0.015 and g[2] - 0.01 < jaw[2] < g[2] + 0.07:
        start = 1                                     # above the cube: descend
    else:
        start = 0
    wps = wps[start:]
    acts, cur = [], state.copy()
    for q3, grip, n in wps:
        for i in range(n):
            a = np.zeros(6); a[:3] = cur[:3] + (q3 - cur[:3]) * (i + 1) / n; a[5] = grip
            acts.append(a)
        cur = acts[-1]
    while len(acts) < 50:
        acts.append(acts[-1].copy())
    return np.array(acts[:50])


class FakePolicy:
    world_box: dict = {}

    def __init__(self):
        self.config = Cfg()
        self.model = FakeModel()
        self._p = torch.nn.Parameter(torch.zeros(1))
        self.reset()

    @classmethod
    def from_pretrained(cls, ckpt, device="cpu"): return cls()
    def parameters(self): return iter([self._p])
    def eval(self): return self
    def reset(self): self._queue = deque(maxlen=self.config.n_action_steps)

    def _get_action_chunk(self, batch, noise=None):
        if noise is None:
            noise = self.model.sample_noise((1, self.config.chunk_size, self.config.max_action_dim), "cpu")
        x_t = noise
        x_t += 0.5                                    # in-place, like lerobot's Euler step
        w = self.world_box["w"]
        cams = [k for k in batch if k.startswith("observation.images.")]
        missing = 3 - len(cams)
        blank = sum(float(batch[k].mean()) < 0.01 for k in cams)
        wrist = batch.get("observation.images.camera2")
        shift = 0.0 if wrist is None else (float(wrist.mean()) * 255 - (128 + 400 * (w.cube[0] - 0.24))) / 400
        bias = 0.011 * (missing + blank) + shift + 0.008 * float(x_t[0, 0, 0] - 0.5)
        state = batch["observation.state"][0].numpy()
        plan = _plan(state, w.cube.copy(), w.target.copy(), bias, attached=w.attached)
        return torch.tensor(plan, dtype=torch.float32)[None]

    def predict_action_chunk(self, batch, noise=None):
        return self._get_action_chunk(batch, noise)

    def select_action(self, batch, noise=None):
        if len(self._queue) == 0:
            a = self._get_action_chunk(batch, noise)
            self._queue.extend(a.transpose(0, 1)[: self.config.n_action_steps])
        return self._queue.popleft()


def fake_processors(policy_cfg=None, pretrained_path=None):
    def pre(obs):
        out = {}
        for k, v in obs.items():
            out[k] = v[None] if torch.is_tensor(v) else v
        return out
    return pre, (lambda a: a)


def install_fake_lerobot():
    for name in ["lerobot", "lerobot.policies", "lerobot.policies.smolvla",
                 "lerobot.policies.smolvla.modeling_smolvla", "lerobot.policies.factory"]:
        sys.modules[name] = types.ModuleType(name)
    sys.modules["lerobot.policies.smolvla.modeling_smolvla"].SmolVLAPolicy = FakePolicy
    sys.modules["lerobot.policies.factory"].make_pre_post_processors = fake_processors