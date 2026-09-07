"""
inspect_observation_schema.py
------------------------------
S2 (BLOCKER): before trusting any "absent camera" result, confirm exactly
what SmolVLA receives in each condition, and whether "absent" produces a
structurally valid observation or an out-of-distribution input shape the
model was never trained to handle.

This does NOT run a rollout. It builds one observation dict per condition
from a single fixed dummy state and inspects it — keys, shapes, dtypes,
and (critically) whether SmolVLAPolicy.select_action raises, silently
pads, or silently reorders anything.

Usage:
    python inspect_observation_schema.py --checkpoint <ckpt> --dataset_dir <dir>

Output: results/schema_check/schema_report.json + printed summary.
Read the summary before trusting ANY absent_* result in the paper.
"""
import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import torch

from scene_params import BLACKOUT_MODES, is_camera_absent

CONDITIONS_TO_CHECK = [
    "none",             # nominal — all 3 tensors present, unmodified
    "context_only",     # other 2 present but zeroed (blackout, not absent)
    "no_top",            # top zeroed
    "absent_top",        # top tensor OMITTED
    "absent_wrist_top",  # two tensors OMITTED
    "absent_all",         # all tensors OMITTED
]

# Per-camera DISTINCT solid colours, not a shared all-zero image. Three
# reasons this matters:
#   1. An all-zero image is a no-op under every blackout mode (zeroing an
#      already-zero image changes nothing), so "none" / "context_only" /
#      "no_top" previously fed the model byte-for-byte identical tensors —
#      making it impossible to notice if select_action() wasn't even being
#      re-invoked per condition (see the policy.reset() note below).
#   2. Distinct colours let you SEE whether removing a camera key changes
#      which learned camera-position-embedding a given image ends up
#      matched with, i.e. whether the model attends to images by NAME
#      (observation.images.cameraN) or by POSITIONAL SLOT. A crash-free
#      forward pass alone doesn't rule out the latter, which would be a
#      silent, much worse failure mode than a crash.
#   3. It costs nothing — same shape/dtype as before.
DUMMY_IMG_BY_ROLE = {
    "camera1": np.full((256, 256, 3), (200,  30,  30), dtype=np.uint8),  # context: red
    "camera2": np.full((256, 256, 3), ( 30, 200,  30), dtype=np.uint8),  # wrist:   green
    "camera3": np.full((256, 256, 3), ( 30,  30, 200), dtype=np.uint8),  # top:     blue
}
DUMMY_STATE = np.zeros(6, dtype=np.float32)


def describe_tensor(t):
    if t is None:
        return None
    return {"shape": list(t.shape), "dtype": str(t.dtype), "device": str(t.device)}


def build_obs(mode: str, tokenizer, device):
    """Mirror exactly the obs-dict construction in SmolVLAAgent.act()."""
    from scene_params import CameraPerturbation
    pert = CameraPerturbation(blackout_mode=mode)

    def img_to_tensor(arr):
        t = torch.from_numpy(arr.copy()).float() / 255.0
        return t.permute(2, 0, 1).unsqueeze(0).to(device)

    absent = {k for k in ["camera1", "camera2", "camera3"] if pert.camera_is_absent(k)}

    ctx_rgb   = None if "camera1" in absent else pert.apply_blackout(DUMMY_IMG_BY_ROLE["camera1"], "camera1")
    wrist_rgb = None if "camera2" in absent else pert.apply_blackout(DUMMY_IMG_BY_ROLE["camera2"], "camera2")
    top_rgb   = None if "camera3" in absent else pert.apply_blackout(DUMMY_IMG_BY_ROLE["camera3"], "camera3")

    obs = {}
    if ctx_rgb is not None:   obs["observation.images.camera1"] = img_to_tensor(ctx_rgb)
    if wrist_rgb is not None: obs["observation.images.camera2"] = img_to_tensor(wrist_rgb)
    if top_rgb is not None:   obs["observation.images.camera3"] = img_to_tensor(top_rgb)

    obs["observation.state"] = torch.from_numpy(DUMMY_STATE).float().unsqueeze(0).to(device)

    enc = tokenizer(["dummy instruction"], return_tensors="pt", padding=True,
                     truncation=True, max_length=64)
    obs["observation.language.tokens"]         = enc["input_ids"].to(device)
    obs["observation.language.attention_mask"] = enc["attention_mask"].bool().to(device)

    return obs, absent, pert.label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="lerobot/smolvla_base")
    parser.add_argument("--dataset_dir", default="demos/lerobot/")
    parser.add_argument("--out", default="results/schema_check/schema_report.json")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] Loading {args.checkpoint} on {device} for a structural probe "
          f"(no rollout, single forward pass per condition)...")
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint, device=device)
    policy.eval()
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

    report = {}
    for mode in CONDITIONS_TO_CHECK:
        entry = {"blackout_mode": mode}
        try:
            # CRITICAL: reset the policy's internal action queue before
            # every condition. SmolVLA (like every chunked LeRobot policy —
            # ACT, Diffusion Policy, etc.) doesn't run a fresh forward pass
            # on every call to select_action(): it predicts a CHUNK of
            # future actions once, caches them, and pops one cached action
            # per call until the queue is empty. Looping over conditions
            # without resetting means later conditions can silently just
            # POP a cached action left over from an EARLIER condition's
            # forward pass — never actually invoking the model on that
            # condition's observation at all. policy.reset() clears that
            # queue so every condition below is a genuine, independent
            # forward pass on ITS OWN obs dict.
            policy.reset()

            obs, absent, label = build_obs(mode, tokenizer, device)
            entry["label"] = label
            entry["absent_camera_keys"] = sorted(absent)
            entry["observation_keys"] = sorted(obs.keys())
            entry["tensor_shapes"] = {
                k: describe_tensor(v) for k, v in obs.items() if torch.is_tensor(v)
            }
            with torch.no_grad():
                action = policy.select_action(obs)
            entry["forward_pass"] = "OK"
            entry["action_shape"] = list(action.shape)
            entry["action_values_finite"] = bool(torch.isfinite(action).all().item())
            entry["action_sample"] = action.flatten()[:6].cpu().tolist()
        except Exception as e:
            entry["forward_pass"] = "RAISED"
            entry["error_type"] = type(e).__name__
            entry["error_message"] = str(e)
            entry["traceback"] = traceback.format_exc()

        report[mode] = entry
        status = entry["forward_pass"]
        print(f"\n[{mode:20s}] forward_pass={status}")
        if status == "OK":
            print(f"    keys        : {entry['observation_keys']}")
            print(f"    action shape: {entry['action_shape']}  finite={entry['action_values_finite']}")
        else:
            print(f"    {entry['error_type']}: {entry['error_message']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n[ok] Full report written to {args.out}")

    print("\n" + "=" * 70)
    print("WHAT TO LOOK FOR")
    print("=" * 70)
    print("""
  1. Does policy.select_action() RAISE for any absent_* mode?
     -> If yes: your 0% TSR for that condition may just be a crash /
        exception being swallowed somewhere upstream in run_rollout(),
        not a genuine "model failed to complete the task." Check
        run_rollout()'s call to agent.act() for a bare except.

  2. If it does NOT raise, does SmolVLAPolicy silently substitute
     something for the missing key (a learned mask embedding, a zero
     tensor, a cached previous frame)? Check the model source
     (modeling_smolvla.py) for how it handles a missing
     observation.images.cameraN key internally — grep for how it
     iterates over expected camera keys during training vs inference.
     If SmolVLA was trained with a fixed hard-coded list of camera keys
     and never saw a variable-length list, this answers S2 directly:
     "camera absent" is out-of-distribution INPUT STRUCTURE, not
     out-of-distribution information content, and the paper's framing
     needs to say that explicitly rather than claiming it's evidence
     about "reliance on visual modalities."

  3. Compare action_sample across none / no_top / absent_top. Each camera
     now gets a DISTINCT solid colour (not a shared zero image) and
     policy.reset() runs before every condition, so these are genuine,
     independent forward passes -- not cached pops from one earlier
     condition's action chunk (that was silently the case in an earlier
     version of this script; if you're diffing against an old
     schema_report.json, discard it, it isn't comparable). If no_top and
     absent_top now produce wildly different actions despite near-
     identical available information (both have context+wrist available,
     absent_top just doesn't have a 3rd tensor at all), that's your direct
     evidence for S2's distinction and can be reported as its own small
     result.
""")


if __name__ == "__main__":
    main()