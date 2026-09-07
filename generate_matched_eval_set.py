"""
generate_matched_eval_set.py
------------------------------
ST4 (HIGH): generate ONE fixed set of (cube_xy, target_xy) initial
conditions, saved to disk, and reused by every perturbation condition and
every checkpoint/seed. This turns cross-condition comparisons from
independently-sampled to paired, which is a large variance reduction for
free (no extra rollouts) and gives you a much cleaner causal story:
"rollout 37 under nominal vs rollout 37 under camera-shift started from
the identical physical state."

Uses the SAME rejection-sampling rule as ScenePerturbation.resolve_spawn
(min_sep=0.12, same nominal centers) so the matched set is drawn from
exactly the distribution the random (unmatched) baseline used — this is
NOT a new distribution, just a frozen, reusable draw from the old one.

Run this ONCE before the final evaluation campaign (ST7: this file
becomes part of the frozen protocol — do not regenerate it between
conditions, or you lose the pairing).

Usage:
    python generate_matched_eval_set.py --n 100 --seed 1234 \
        --run_root icra-final
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scene_params import NOMINAL_SCENE


def generate(n: int, seed: int):
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n):
        cube_xy, target_xy = NOMINAL_SCENE.resolve_spawn(rng)
        pairs.append({
            "cube_xy":   [float(cube_xy[0]),   float(cube_xy[1])],
            "target_xy": [float(target_xy[0]), float(target_xy[1])],
        })
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234,
                        help="Seed for generating the matched set itself. "
                             "This is a THIRD seed distinct from --seed "
                             "(env/rollout noise) and --train_seed used "
                             "elsewhere in the pipeline.")
    parser.add_argument("--out", default=None,
                        help="Explicit output path. Either this OR --run_root "
                             "is required.")
    parser.add_argument("--run_root", default=None,
                        help="Short campaign tag, e.g. 'icra-final'. Derives "
                             "the path as runs/<run_root>/matched_eval_set.json "
                             "(see run_naming.py) so it lives alongside every "
                             "eval run from the same campaign.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow regenerating an already-existing matched "
                             "eval set. Think twice: every condition and every "
                             "checkpoint that already used the old file loses "
                             "its pairing the moment you replace it, and "
                             "nothing will tell you that happened afterward.")
    args = parser.parse_args()

    if args.out is None and args.run_root is None:
        print("[error] Either --out or --run_root is required.")
        return

    if args.out is not None:
        out_path = Path(args.out)
    else:
        from run_naming import run_root as _run_root, matched_eval_set_path
        out_path = matched_eval_set_path(_run_root(args.run_root))
        print(f"[run_root] {args.run_root} -> {out_path}")

    pairs = generate(args.n, args.seed)

    payload = {
        "n": args.n,
        "generation_seed": args.seed,
        "cube_target_pairs": pairs,
        # A hash of the pairs themselves, so any downstream script can
        # assert it's reading the exact frozen set the paper reports on,
        # not a regenerated (and therefore different) one.
        "content_hash": hashlib.sha256(
            json.dumps(pairs, sort_keys=True).encode()
        ).hexdigest()[:16],
    }

    from run_naming import refuse_if_exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        refuse_if_exists(out_path, "Matched eval set", args.overwrite)
    except SystemExit:
        print(f"[error] {out_path} already exists. This is almost certainly "
              f"the frozen set every condition's --matched_eval_set already "
              f"points at — regenerating it silently invalidates every paired "
              f"comparison already computed against it. Pass --overwrite only "
              f"if you are certain nothing downstream depends on the old file, "
              f"or choose a different --run_root / --out.")
        return
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[ok] Wrote {args.n} matched initial conditions → {out_path}")
    print(f"     content_hash = {payload['content_hash']}")
    print(f"     Pass --matched_eval_set {out_path} to zeroshot_eval.py "
          f"for every condition in the final campaign.")


if __name__ == "__main__":
    main()