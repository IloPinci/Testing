"""
stats_analysis.py
------------------
Implements ST1, ST2, ST3 from the review worksheet:

ST1 — Wilson CI stays for binary proportions (TSR, P(reached), P(lifted),
      P(placed)). Mean PSS gets a percentile bootstrap CI instead, since
      it's a continuous mean, not a binomial proportion.

ST2 — Report P(reached), P(lifted), P(placed) per condition as the
      primary diagnostic, each with its own Wilson CI. Also runs an
      internal-consistency check: every "placed" episode must also have
      "reached" and "lifted" True, since placement should logically
      require passing through the earlier sub-goals. Any violation is
      printed explicitly — it means the sub-goal logic (or its latching)
      has a bug and the manuscript's pipeline-stage story is not yet
      trustworthy.

ST3 — For any two conditions (typically perturbed vs. nominal), compute
      delta_TSR = TSR_perturbed - TSR_nominal with a bootstrap CI on the
      delta itself (not just two separate CIs eyeballed against each
      other), plus the same for P(reached/lifted/placed) and mean PSS.
      If --matched (episode-index-aligned, i.e. built from the same
      matched_eval_set — see ST4) is set, the bootstrap resamples
      MATCHED PAIRS (paired bootstrap), which is more powerful than an
      unpaired bootstrap when the two runs share initial conditions.

Usage:
    # Single condition: decomposed probabilities + bootstrap PSS CI
    python stats_analysis.py describe --csv runs/icra-final/eval/nominal/zeroshot.csv

    # Two conditions: effect size with CI
    python stats_analysis.py compare \\
        --csv_a runs/icra-final/eval/nominal/zeroshot.csv \\
        --csv_b runs/icra-final/eval/wpos+3.0cm/zeroshot.csv \\
        --matched   # only if both used the same --matched_eval_set
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


# ══════════════════════════════════════════════════════════════════════════
#  Loading
# ══════════════════════════════════════════════════════════════════════════

def load_csv(path):
    records = []
    with open(path) as f:
        for row in csv.DictReader(f):
            records.append({
                "episode": int(row["episode"]),
                "success": row["success"].strip().lower() == "true",
                "pss":     float(row["pss"]),
                "reached": row["reached"].strip().lower() == "true",
                "lifted":  row["lifted"].strip().lower()  == "true",
                "placed":  row["placed"].strip().lower()  == "true",
            })
    return records


# ══════════════════════════════════════════════════════════════════════════
#  ST1 — Wilson CI (binary) and bootstrap CI (continuous mean)
# ══════════════════════════════════════════════════════════════════════════

def wilson_ci(successes: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom  = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half   = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_mean_ci(values, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for a mean. ST1: use this for PSS, not Wilson."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


# ══════════════════════════════════════════════════════════════════════════
#  ST2 — decomposed subgoal probabilities + internal consistency check
# ══════════════════════════════════════════════════════════════════════════

def decompose(records):
    n = len(records)
    reached = sum(r["reached"] for r in records)
    lifted  = sum(r["lifted"]  for r in records)
    placed  = sum(r["placed"]  for r in records)
    pss_vals = [r["pss"] for r in records]

    result = {
        "n": n,
        "p_reached": {"count": reached, "rate": reached / n, "ci95": wilson_ci(reached, n)},
        "p_lifted":  {"count": lifted,  "rate": lifted  / n, "ci95": wilson_ci(lifted,  n)},
        "p_placed":  {"count": placed,  "rate": placed  / n, "ci95": wilson_ci(placed,  n)},
        "tsr":       {"count": sum(r["success"] for r in records),
                      "rate": sum(r["success"] for r in records) / n,
                      "ci95": wilson_ci(sum(r["success"] for r in records), n)},
        "mean_pss":  {"value": float(np.mean(pss_vals)),
                      "ci95_bootstrap": bootstrap_mean_ci(pss_vals)},
    }

    # Internal consistency: placed => lifted => reached, by construction of
    # the pipeline (you cannot place a cube you never lifted, and you
    # cannot lift a cube you never reached). Any violation means the
    # sub-goal latching logic has a bug.
    violations = {"placed_without_lifted": 0, "placed_without_reached": 0,
                  "lifted_without_reached": 0}
    bad_episodes = []
    for r in records:
        v = []
        if r["placed"] and not r["lifted"]:
            violations["placed_without_lifted"] += 1
            v.append("placed_without_lifted")
        if r["placed"] and not r["reached"]:
            violations["placed_without_reached"] += 1
            v.append("placed_without_reached")
        if r["lifted"] and not r["reached"]:
            violations["lifted_without_reached"] += 1
            v.append("lifted_without_reached")
        if v:
            bad_episodes.append({"episode": r["episode"], "violations": v})

    result["consistency_check"] = {
        "violations": violations,
        "total_violating_episodes": len(bad_episodes),
        "bad_episodes": bad_episodes[:20],  # cap for readability
    }
    return result


def print_describe(result, label=""):
    print(f"\n{'='*60}\nDecomposed sub-goal probabilities{': ' + label if label else ''}  (N={result['n']})\n{'='*60}")
    for key, name in [("p_reached", "P(reached)"), ("p_lifted", "P(lifted)"),
                       ("p_placed", "P(placed)"), ("tsr", "TSR")]:
        d = result[key]
        lo, hi = d["ci95"]
        print(f"  {name:<12} = {d['rate']:.1%}   95% Wilson CI [{lo:.3f}, {hi:.3f}]   (count {d['count']}/{result['n']})")
    m = result["mean_pss"]
    lo, hi = m["ci95_bootstrap"]
    print(f"  {'Mean PSS':<12} = {m['value']:.3f}  95% bootstrap CI [{lo:.3f}, {hi:.3f}]")

    cc = result["consistency_check"]
    print(f"\n  Internal consistency check:")
    if cc["total_violating_episodes"] == 0:
        print(f"    OK — no episode violates reached <= lifted <= placed ordering.")
    else:
        print(f"    [WARNING] {cc['total_violating_episodes']} episode(s) violate expected "
              f"sub-goal ordering: {cc['violations']}")
        print(f"    This means the sub-goal / latch logic has a bug — do not report "
              f"the pipeline-stage breakdown until this is fixed and re-verified.")
        for be in cc["bad_episodes"][:5]:
            print(f"      episode {be['episode']}: {be['violations']}")


# ══════════════════════════════════════════════════════════════════════════
#  ST3 — effect sizes (delta) with bootstrap CI, paired or unpaired
# ══════════════════════════════════════════════════════════════════════════

def _binary_array(records, key):
    return np.array([1.0 if r[key] else 0.0 for r in records])


def delta_ci_unpaired(a, b, n_boot=10000, alpha=0.05, seed=0):
    """Bootstrap CI for mean(b) - mean(a), independent resampling of each."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    na, nb = len(a), len(b)
    idx_a = rng.integers(0, na, size=(n_boot, na))
    idx_b = rng.integers(0, nb, size=(n_boot, nb))
    deltas = b[idx_b].mean(axis=1) - a[idx_a].mean(axis=1)
    point = float(b.mean() - a.mean())
    lo = float(np.percentile(deltas, 100 * alpha / 2))
    hi = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    return point, lo, hi


def delta_ci_paired(a, b, n_boot=10000, alpha=0.05, seed=0):
    """
    Bootstrap CI for mean(b - a) using paired differences. Requires a and b
    to be the SAME LENGTH and index-aligned (i.e. both built from the same
    matched_eval_set, episode i in a corresponds to episode i in b).
    More powerful than the unpaired version when conditions share initial
    conditions (ST4).
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert len(a) == len(b), "paired delta requires equal-length, index-aligned arrays"
    diffs = b - a
    n = len(diffs)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diffs[idx].mean(axis=1)
    point = float(diffs.mean())
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return point, lo, hi


def compare(records_a, records_b, paired=False):
    fn = delta_ci_paired if paired else delta_ci_unpaired
    if paired and len(records_a) != len(records_b):
        raise ValueError(f"--matched requires equal N: a has {len(records_a)}, "
                          f"b has {len(records_b)}")

    out = {"paired": paired, "n_a": len(records_a), "n_b": len(records_b)}
    for key in ["success", "reached", "lifted", "placed"]:
        a = _binary_array(records_a, key)
        b = _binary_array(records_b, key)
        point, lo, hi = fn(a, b)
        out[f"delta_{key}"] = {"point": point, "ci95": [lo, hi],
                                "significant": not (lo <= 0 <= hi)}

    pss_a = [r["pss"] for r in records_a]
    pss_b = [r["pss"] for r in records_b]
    point, lo, hi = fn(pss_a, pss_b)
    out["delta_pss"] = {"point": point, "ci95": [lo, hi], "significant": not (lo <= 0 <= hi)}
    return out


def print_compare(out, label_a="A", label_b="B"):
    mode = "PAIRED (matched initial conditions)" if out["paired"] else "UNPAIRED (independent samples)"
    print(f"\n{'='*70}\nEffect size: {label_b} - {label_a}   [{mode}]\n{'='*70}")
    print(f"  N_a={out['n_a']}  N_b={out['n_b']}")
    for key, name in [("delta_success", "Delta TSR"), ("delta_reached", "Delta P(reached)"),
                       ("delta_lifted", "Delta P(lifted)"), ("delta_placed", "Delta P(placed)"),
                       ("delta_pss", "Delta mean PSS")]:
        d = out[key]
        lo, hi = d["ci95"]
        sig = "  <-- 95% CI excludes 0" if d["significant"] else ""
        print(f"  {name:<20} = {d['point']:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]{sig}")


# ══════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_describe = sub.add_parser("describe", help="ST1+ST2: decomposed probabilities for one condition")
    p_describe.add_argument("--csv", required=True)
    p_describe.add_argument("--label", default=None)
    p_describe.add_argument("--out", default=None, help="Optional path to write JSON")

    p_compare = sub.add_parser("compare", help="ST3: effect size between two conditions")
    p_compare.add_argument("--csv_a", required=True, help="Usually the nominal/baseline condition")
    p_compare.add_argument("--csv_b", required=True, help="Usually the perturbed condition")
    p_compare.add_argument("--label_a", default="nominal")
    p_compare.add_argument("--label_b", default="perturbed")
    p_compare.add_argument("--matched", action="store_true",
                           help="Set only if both CSVs were generated with the "
                                "SAME --matched_eval_set file (ST4) — enables "
                                "the more powerful paired bootstrap.")
    p_compare.add_argument("--out", default=None)

    args = parser.parse_args()

    if args.cmd == "describe":
        records = load_csv(args.csv)
        result = decompose(records)
        print_describe(result, args.label or args.csv)
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=2))
            print(f"\n[ok] {args.out}")

    elif args.cmd == "compare":
        records_a = load_csv(args.csv_a)
        records_b = load_csv(args.csv_b)
        out = compare(records_a, records_b, paired=args.matched)
        print_compare(out, args.label_a, args.label_b)
        if args.out:
            Path(args.out).write_text(json.dumps(out, indent=2))
            print(f"\n[ok] {args.out}")


if __name__ == "__main__":
    main()