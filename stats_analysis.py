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
from math import comb
from pathlib import Path

import numpy as np


# ══════════════════════════════════════════════════════════════════════════
#  Loading
# ══════════════════════════════════════════════════════════════════════════

_BOOL = {"true": True, "false": False}


def _parse(v):
    if v is None:
        return None
    t = v.strip()
    if t == "":
        return None
    lt = t.lower()
    if lt in _BOOL:
        return _BOOL[lt]
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        return t


def load_csv(path, metric: str = "success"):
    """
    Load one zeroshot.csv. Every column is parsed (bool/int/float/str/None),
    so protocol-v2 columns (success_strict, failure_mode, t_grasp, ...) are
    available; v1 CSVs still load. `metric` selects which column is exposed
    as r["y"] (the outcome analysed by compare()). Rows are sorted by episode
    so paired comparisons are index-aligned by episode id, not file order.
    """
    records = []
    with open(path) as f:
        for row in csv.DictReader(f):
            r = {k: _parse(v) for k, v in row.items()}
            if metric not in r:
                raise KeyError(f"{path} has no column {metric!r} "
                               f"(protocol-v1 CSV? use metric='success').")
            r["y"] = bool(r[metric])
            records.append(r)
    records.sort(key=lambda r: r["episode"])
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

def _prop(values):
    c, n = int(sum(bool(v) for v in values)), len(values)
    return {"count": c, "rate": c / n, "ci95": wilson_ci(c, n)}

def decompose(records):
    n = len(records)
    reached = sum(r["reached"] for r in records)
    lifted  = sum(r["lifted"]  for r in records)
    placed  = sum(r["placed"]  for r in records)
    pss_vals = [r["pss"] for r in records]

    result = {
        "n": n,
        
        **({"p_success_strict": _prop([r["success_strict"] for r in records]),
            "p_success_carry": _prop([r["success_carry"] for r in records]),
            "p_reached_raw": _prop([r["reached_raw"] for r in records]),
            "p_grasped_raw": _prop([r["grasped_raw"] for r in records]),
            "failure_modes": {m: sum(1 for r in records if r["failure_mode"] == m)
                              for m in dict.fromkeys(r["failure_mode"] for r in records)}}
           if "success_strict" in records[0] else {}),

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
    raw = "reached_raw" in records[0] and "grasped_raw" in records[0]
    if raw:
        # Protocol v2: raw latches, so this check CAN fail and is informative.
        records_cc = [{"episode": r["episode"], "placed": r.get("success_strict"),
                       "lifted": r["grasped_raw"], "reached": r["reached_raw"]} for r in records]
    else:
        # Protocol v1 CSVs forced placed=>lifted=>reached before writing,
        # so this branch can never report a violation.
        records_cc = records
    violations = {"placed_without_lifted": 0, "placed_without_reached": 0,
                  "lifted_without_reached": 0}
    bad_episodes = []
    for r in records_cc:
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
        "uses_raw_latches": raw,
        "violations": violations,
        "total_violating_episodes": len(bad_episodes),
        "bad_episodes": bad_episodes[:20],  # cap for readability
    }
    return result



def print_describe(result, label=""):
    print(f"\n{'='*60}\nDecomposed sub-goal probabilities{': ' + label if label else ''}  (N={result['n']})\n{'='*60}")
    rows = [("p_reached", "P(reached)"), ("p_lifted", "P(lifted)"),
            ("p_placed", "P(placed)"), ("tsr", "TSR")]
    rows += [(k, n) for k, n in [("p_reached_raw", "P(reach,raw)"), ("p_grasped_raw", "P(grasp,raw)"),
                                 ("p_success_carry", "TSR carry"), ("p_success_strict", "TSR strict")]
             if k in result]
    for key, name in rows:
        d = result[key]
        lo, hi = m["ci95_bootstrap"]
    print(f"  {'Mean PSS':<12} = {m['value']:.3f}  95% bootstrap CI [{lo:.3f}, {hi:.3f}]")

    if "failure_modes" in result:
        print("  Failure modes: " + ", ".join(f"{k}={v}" for k, v in result["failure_modes"].items()))
    cc = result["consistency_check"]
    print(f"\n  Internal consistency check:")
    if not cc["uses_raw_latches"]:
        print("    (v1 CSV: sub-goals were forced monotone before writing — this check "
              "is uninformative by construction)")
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


def mcnemar_exact(a, b):
    """
    Exact (binomial) McNemar test for paired binary outcomes a_i, b_i.
    n01 = #(a=1, b=0), n10 = #(a=0, b=1). Two-sided p-value. This is the
    standard test for "same episodes, two conditions" and is what reviewers
    expect next to a paired-bootstrap CI.
    """
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    assert len(a) == len(b), "McNemar needs index-aligned pairs"
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    m = n01 + n10
    if m == 0:
        return {"n01": 0, "n10": 0, "p": 1.0}
    k = min(n01, n10)
    p = sum(comb(m, i) for i in range(k + 1)) / 2 ** m
    return {"n01": n01, "n10": n10, "p": min(1.0, 2 * p)}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input). Controls the
    family-wise error rate across all perturbation-vs-nominal tests."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def did_paired(nom_a, pert_a, nom_b, pert_b, n_boot=10000, alpha=0.05, seed=0):
    """
    Paired difference-in-differences: how much MORE does the perturbation
    hurt under setting B than under setting A (e.g. B = closed loop,
    A = open loop)?  DiD = (pert_b - nom_b) - (pert_a - nom_a), computed per
    episode index (all four arrays aligned on the matched eval set) and
    bootstrapped over episodes. Negative = B is less robust than A.
    """
    arrs = [np.asarray(x, dtype=float) for x in (nom_a, pert_a, nom_b, pert_b)]
    n = len(arrs[0])
    assert all(len(x) == n for x in arrs), "DiD needs four index-aligned arrays"
    d = (arrs[3] - arrs[2]) - (arrs[1] - arrs[0])
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return float(d.mean()), float(np.percentile(boot, 100 * alpha / 2)), \
        float(np.percentile(boot, 100 * (1 - alpha / 2)))


def cluster_bootstrap_mean(groups, n_boot=10000, alpha=0.05, seed=0):
    """
    Two-stage (hierarchical) bootstrap for episodes nested in training seeds:
    resample checkpoints with replacement, then episodes within each. Use
    this, not a pooled Wilson CI, once you have >1 training seed — pooling
    treats 3x100 episodes as 300 independent draws and ignores between-seed
    variance (usually the larger source of variability in policy learning).
    groups: list of 1-D arrays (one per checkpoint).
    """
    groups = [np.asarray(g, dtype=float) for g in groups]
    rng = np.random.default_rng(seed)
    G = len(groups)
    means = np.empty(n_boot)
    for i in range(n_boot):
        picks = rng.integers(0, G, size=G)
        means[i] = np.mean([groups[j][rng.integers(0, len(groups[j]), len(groups[j]))].mean()
                            for j in picks])
    point = float(np.mean([g.mean() for g in groups]))
    return point, float(np.percentile(means, 100 * alpha / 2)), \
        float(np.percentile(means, 100 * (1 - alpha / 2)))


def power_mcnemar(p_a, p_b, n, rho=0.5, alpha=0.05, n_sim=4000, seed=0):
    """
    Simulated power of the exact McNemar test to detect p_a -> p_b with n
    paired episodes. rho in [0,1] sets how strongly outcomes are coupled
    across the two conditions (0 = independent, 1 = maximal overlap); the
    paired design is more powerful the larger rho. Estimate rho from a pilot
    (e.g. nominal k50 vs nominal k10) rather than guessing.
    """
    p11 = p_a * p_b + rho * (min(p_a, p_b) - p_a * p_b)
    probs = np.array([p11, p_a - p11, p_b - p11, 1 - p_a - p_b + p11])   # 11, 10, 01, 00
    probs = np.clip(probs, 0, 1); probs /= probs.sum()
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sim):
        c11, c10, c01, c00 = rng.multinomial(n, probs)
        a = np.r_[np.ones(c11 + c10), np.zeros(c01 + c00)].astype(bool)
        b = np.r_[np.ones(c11), np.zeros(c10), np.ones(c01), np.zeros(c00)].astype(bool)
        hits += mcnemar_exact(a, b)["p"] < alpha
    return hits / n_sim


def compare(records_a, records_b, paired=False, metric="success"):
    fn = delta_ci_paired if paired else delta_ci_unpaired
    if paired and len(records_a) != len(records_b):
        raise ValueError(f"--matched requires equal N: a has {len(records_a)}, "
                          f"b has {len(records_b)}")
    if paired and [r["episode"] for r in records_a] != [r["episode"] for r in records_b]:
        raise ValueError("paired comparison needs the same episode ids in both runs")

    out = {"paired": paired, "n_a": len(records_a), "n_b": len(records_b), "metric": metric}
    for key in [metric, "reached", "lifted", "placed"]:
        a = _binary_array(records_a, key)
        b = _binary_array(records_b, key)
        point, lo, hi = fn(a, b)
        name = "success" if key == metric else key
        out[f"delta_{name}"] = {"point": point, "ci95": [lo, hi],
                                "significant": not (lo <= 0 <= hi)}
    if paired:
        out["delta_success"]["mcnemar"] = mcnemar_exact(
            _binary_array(records_a, metric), _binary_array(records_b, metric))

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
        if "mcnemar" in d:
            mc = d["mcnemar"]
            print(f"  {'':<20}   exact McNemar: n01={mc['n01']} n10={mc['n10']} p={mc['p']:.3g}")


# ══════════════════════════════════════════════════════════════════════════

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
    p_compare.add_argument("--metric", default="success",
                           help="Outcome column, e.g. success, success_strict, success_carry.")
    p_compare.add_argument("--out", default=None)

    p_power = sub.add_parser("power", help="Simulated McNemar power for a paired design")
    p_power.add_argument("--p_a", type=float, required=True, help="e.g. nominal TSR")
    p_power.add_argument("--p_b", type=float, required=True, help="TSR you want to detect")
    p_power.add_argument("--rho", type=float, default=0.5)
    p_power.add_argument("--alpha", type=float, default=0.05,
                         help="Per-test alpha (divide by #tests for a Bonferroni-safe estimate)")
    p_power.add_argument("--n", type=int, nargs="+", default=[50, 100, 150, 200, 300])

    args = parser.parse_args()
    
    if args.cmd == "describe":
        records = load_csv(args.csv)
        result = decompose(records)
        print_describe(result, args.label or args.csv)
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=2))
            print(f"\n[ok] {args.out}")

    elif args.cmd == "compare":
        records_a = load_csv(args.csv_a, metric=args.metric)
        records_b = load_csv(args.csv_b, metric=args.metric)
        out = compare(records_a, records_b, paired=args.matched, metric=args.metric)
        print_compare(out, args.label_a, args.label_b)
        if args.out:
            Path(args.out).write_text(json.dumps(out, indent=2))
            print(f"\n[ok] {args.out}")

    elif args.cmd == "power":
        print(f"Power of exact McNemar, p_a={args.p_a} -> p_b={args.p_b}, rho={args.rho}, alpha={args.alpha}")
        for n in args.n:
            print(f"  N={n:<5} power={power_mcnemar(args.p_a, args.p_b, n, args.rho, args.alpha):.2f}")


if __name__ == "__main__":
    main()