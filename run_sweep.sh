#!/usr/bin/env bash
# run_sweep.sh — sequential driver for the Step 4 zeroshot_eval.py sweep.
#
# Usage (run from repo root, e.g. /workspace):
#   export COMMON="--run_root icra-final --checkpoint $CKPT \
#       --checkpoint_id baseline --dataset_dir $DATA --n 100 \
#       --matched_eval_set $MATCH"
#   nohup bash icra/run_sweep.sh > runs/icra-final/sweep_driver.out 2>&1 &
#   tail -f runs/icra-final/sweep_driver.out
#
# Run this under nohup/tmux/screen. With 16 conditions at ~45 min each this
# is a ~12 hour job and an SSH disconnect must not kill it.
#
# Behavior:
#   - Runs each condition SEQUENTIALLY. On one GPU there's no benefit to
#     parallelizing these, and running two Genesis/CUDA processes at once
#     is a good way to get silently corrupted renders, not just contention.
#   - NEVER aborts the whole sweep because one condition failed. Every
#     condition's outcome (OK / ALREADY_DONE / FAILED / SMOKE_FAILED) is
#     appended to runs/<tag>/sweep_log.csv and the driver moves on.
#   - For absent_* conditions specifically: runs a throwaway --n 3 smoke
#     test FIRST, to a scratch path that never touches the real per-
#     condition output location. If the smoke test crashes, the real
#     --n 100 run for that condition is skipped entirely (not attempted)
#     — so a structurally-incompatible condition costs ~1 minute of GPU
#     time instead of 45.
#   - Safe to re-run after an interruption (Ctrl-C, SSH drop, OOM). Any
#     condition whose output already exists gets a fast, expected refusal
#     from zeroshot_eval.py's own overwrite guard; this script detects
#     that specific message and logs it as ALREADY_DONE, not FAILED, and
#     moves on without wasting time re-attempting it.
#
# This script deliberately never re-derives condition_slug()/output paths
# itself — that logic lives in run_naming.py and duplicating it here would
# be exactly the kind of drift-prone re-implementation already fixed
# elsewhere in this codebase. It only watches zeroshot_eval.py's own
# stdout/exit code to decide what happened.

set -uo pipefail   # NOT -e: a failed condition must not stop the sweep

if [ -z "${COMMON:-}" ]; then
    echo "[fatal] \$COMMON is not set. Export it first, e.g.:"
    echo '  export COMMON="--run_root icra-final --checkpoint $CKPT --checkpoint_id baseline --dataset_dir $DATA --n 100 --matched_eval_set $MATCH"'
    exit 1
fi

RUN_ROOT_TAG=$(echo "$COMMON" | grep -oP '(?<=--run_root )\S+' || true)
if [ -z "$RUN_ROOT_TAG" ]; then
    echo "[fatal] Could not find --run_root in \$COMMON."
    echo "        This driver requires --run_root (not --out) in \$COMMON, since"
    echo "        it needs a stable, tag-derived place to write its own sweep_log.csv"
    echo "        and per-condition logs alongside your actual eval outputs."
    exit 1
fi

LOG_DIR="runs/${RUN_ROOT_TAG}/sweep_logs"
SWEEP_LOG="runs/${RUN_ROOT_TAG}/sweep_log.csv"
mkdir -p "$LOG_DIR"
if [ ! -f "$SWEEP_LOG" ]; then
    echo "timestamp,condition,extra_args,status,elapsed_sec,log_file" > "$SWEEP_LOG"
fi

# ── Condition list: "slug|extra args" ────────────────────────────────────
# The slug here is only a log-filename / sweep_log.csv label for THIS
# script — it is not guaranteed to be byte-identical to condition_slug()'s
# own output naming in run_naming.py. Don't use it to guess CSV paths for
# analyze_all_results.py; that script derives paths from CONDITION_SPECS
# itself and doesn't need this file at all.
CONDITIONS=(
    "wpos+1.5cm|--pos_target wrist --pos_offset 0.015 0 0"
    "wpos+3.0cm|--pos_target wrist --pos_offset 0.03 0 0"
    "wtilt-10|--tilt_target wrist --tilt_deg -10"
    "wtilt+10|--tilt_target wrist --tilt_deg 10"
    "cpos-nearL|--cube_pos near_left"
    "cpos-farL|--cube_pos far_left"
    "ctx-only|--blackout_mode context_only"
    "wrist-only|--blackout_mode wrist_only"
    "top-only|--blackout_mode top_only"
    "no-top|--blackout_mode no_top"
    "no-wrist|--blackout_mode no_wrist"
    "no-ctx|--blackout_mode no_context"
    "abs-top|--blackout_mode absent_top"
    "abs-w+t|--blackout_mode absent_wrist_top"
    "sph-2flank|--spheres two_flanking"
    "sph-colormatch|--spheres color_match"
)
# Slugs that get a --n 3 smoke test before the real run. Bash 3.2 (macOS
# default) has no associative arrays, hence the plain space-separated
# lookup below instead of a proper set/dict.
SMOKE_FIRST=" abs-top abs-w+t "

run_one() {
    local slug="$1"; shift
    local extra_args="$*"
    local log_file="${LOG_DIR}/${slug}.log"
    local t0 t1 elapsed status rc

    echo ""
    echo "============================================================"
    echo "RUNNING: ${slug}   (args: ${extra_args})"
    echo "============================================================"

    t0=$(date +%s)
    # shellcheck disable=SC2086
    python icra/zeroshot_eval.py $COMMON $extra_args > "$log_file" 2>&1
    rc=$?
    t1=$(date +%s)
    elapsed=$((t1 - t0))

    if [ $rc -eq 0 ]; then
        status="OK"
        echo "  OK  (${elapsed}s)  -> $log_file"
    elif grep -q "already exists" "$log_file"; then
        status="ALREADY_DONE"
        echo "  = already had data, skipped by zeroshot_eval.py's own overwrite guard  (${elapsed}s)"
    else
        status="FAILED"
        echo "  FAILED (exit $rc, ${elapsed}s) -- see $log_file"
        echo "  last 25 lines:"
        tail -25 "$log_file" | sed 's/^/    /'
    fi

    printf '%s,%s,"%s",%s,%s,%s\n' \
        "$(date -Iseconds)" "$slug" "$extra_args" "$status" "$elapsed" "$log_file" >> "$SWEEP_LOG"
}

smoke_test_absent() {
    local slug="$1"; shift
    local extra_args="$*"
    local smoke_out="${LOG_DIR}/${slug}_smoke.csv"
    local smoke_log="${LOG_DIR}/${slug}_smoke.log"
    local rc

    echo "  ... smoke-testing ${slug} with --n 3 first (scratch output, never touches the real run's path)"
    # shellcheck disable=SC2086
    python icra/zeroshot_eval.py $COMMON $extra_args \
        --n 3 --out "$smoke_out" --overwrite > "$smoke_log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "  SMOKE TEST FAILED for ${slug} -- skipping the full --n run entirely."
        echo "  See $smoke_log for the crash. This condition likely needs the S2"
        echo "  investigation (inspect_observation_schema.py) revisited before it can"
        echo "  run at all -- do not just retry it as-is."
        printf '%s,%s,"%s",%s,%s,%s\n' \
            "$(date -Iseconds)" "$slug" "$extra_args" "SMOKE_FAILED" "0" "$smoke_log" >> "$SWEEP_LOG"
        return 1
    fi
    echo "  smoke test passed"
    return 0
}

for entry in "${CONDITIONS[@]}"; do
    slug="${entry%%|*}"
    extra_args="${entry#*|}"

    if [[ "$SMOKE_FIRST" == *" ${slug} "* ]]; then
        if smoke_test_absent "$slug" "$extra_args"; then
            run_one "$slug" $extra_args
        fi
    else
        run_one "$slug" $extra_args
    fi
done

echo ""
echo "============================================================"
echo "Sweep finished. Summary (runs/${RUN_ROOT_TAG}/sweep_log.csv):"
echo "============================================================"
column -s, -t "$SWEEP_LOG" 2>/dev/null || cat "$SWEEP_LOG"

FAILED_COUNT=$(grep -c ',FAILED,\|,SMOKE_FAILED,' "$SWEEP_LOG" || true)
if [ "${FAILED_COUNT:-0}" -gt 0 ]; then
    echo ""
    echo "[warn] ${FAILED_COUNT} condition(s) failed or were skipped due to a smoke-test failure."
    echo "       Do NOT run analyze_all_results.py yet -- check sweep_log.csv and the"
    echo "       individual logs in runs/${RUN_ROOT_TAG}/sweep_logs/ before treating"
    echo "       this campaign as complete."
fi