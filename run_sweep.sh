#!/usr/bin/env bash
# run_sweep.sh — sequential driver for the evaluation campaign (protocol v2).
#
# Usage (from the directory that contains icra/):
#   export CKPT=... MATCH=runs/paper-v2/matched_eval_set.json
#   export COMMON="--run_root paper-v2 --checkpoint $CKPT --checkpoint_id seed0 \
#       --train_seed 0 --n 100 --matched_eval_set $MATCH"
#   export HORIZONS="50 10"            # execution horizons (open loop, closed loop)
#   export CONDITION_SET=core           # core | dose | controls | all
#   export VIDEO_ARGS="--video_episodes 0-4"   # same 5 initial states in every condition
#   nohup bash icra/run_sweep.sh > runs/paper-v2/sweep_driver.out 2>&1 &
#
# What changed vs. v1 and why
#   * Every condition runs at EVERY horizon in $HORIZONS inside ONE campaign
#     (same code, checkpoint, matched set) -> outputs runs/<tag>/eval/k<K>/<slug>/.
#     OL vs CL used to be two separately produced roots, which made it impossible
#     to rule out code/protocol differences between them.
#   * --n_action_steps is set here, explicitly, per run. $COMMON must not contain it.
#   * Conditions are grouped into sets (core / dose-response / controls) so a
#     paper campaign is a declared list, not whatever happened to be run.
#   * Videos: the SAME episode indices are recorded in every condition and
#     horizon, so make_comparison_video.py can put them side by side.
#   * Everything else (never abort the sweep on one failure, ALREADY_DONE
#     detection, absent_* smoke tests) is unchanged.

set -uo pipefail   # NOT -e: a failed condition must not stop the sweep

if [ -z "${COMMON:-}" ]; then
    echo "[fatal] \$COMMON is not set (see header)."; exit 1
fi
if echo "$COMMON" | grep -q -- "--n_action_steps"; then
    echo "[fatal] Remove --n_action_steps from \$COMMON; set \$HORIZONS instead."; exit 1
fi
if ! echo "$COMMON" | grep -q -- "--matched_eval_set"; then
    echo "[warn] \$COMMON has no --matched_eval_set: runs will NOT be paired across conditions."
fi
RUN_ROOT_TAG=$(echo "$COMMON" | grep -oP '(?<=--run_root )\S+' || true)
if [ -z "$RUN_ROOT_TAG" ]; then
    echo "[fatal] \$COMMON must contain --run_root <tag>."; exit 1
fi

HORIZONS="${HORIZONS:-50 10}"
CONDITION_SET="${CONDITION_SET:-core}"
VIDEO_ARGS="${VIDEO_ARGS:---video_episodes 0-4}"
EVAL="${EVAL_SCRIPT:-icra/zeroshot_eval.py}"

LOG_DIR="runs/${RUN_ROOT_TAG}/sweep_logs"
SWEEP_LOG="runs/${RUN_ROOT_TAG}/sweep_log.csv"
mkdir -p "$LOG_DIR"
[ -f "$SWEEP_LOG" ] || echo "timestamp,condition,horizon,extra_args,status,elapsed_sec,log_file" > "$SWEEP_LOG"

# ── Condition sets: "label|extra args" ──────────────────────────────────────
CORE=(
    "nominal|--blackout_mode none"
    # camera pose (wrist)
    "wpos-3.0cm|--pos_target wrist --pos_offset -0.03 0 0"
    "wpos+1.5cm|--pos_target wrist --pos_offset 0.015 0 0"
    "wpos+3.0cm|--pos_target wrist --pos_offset 0.03 0 0"
    "wtilt-10|--tilt_target wrist --tilt_deg -10"
    "wtilt+10|--tilt_target wrist --tilt_deg 10"
    # object position (out of the +-4 cm training range)
    "cpos-nearL|--cube_pos near_left"
    "cpos-farL|--cube_pos far_left"
    # camera availability: full 2^3 factorial (all-black completes it)
    "ctx-only|--blackout_mode context_only"
    "wrist-only|--blackout_mode wrist_only"
    "top-only|--blackout_mode top_only"
    "no-top|--blackout_mode no_top"
    "no-wrist|--blackout_mode no_wrist"
    "no-ctx|--blackout_mode no_context"
    "all-black|--blackout_mode all_black"
    # token dropped instead of blanked
    "abs-top|--blackout_mode absent_top"
    "abs-w+t|--blackout_mode absent_wrist_top"
    "abs-ctx|--blackout_mode absent_context"      # remaining cameras shift up one slot
    "abs-wrist|--blackout_mode absent_wrist"      # top camera moves into the wrist slot
    # all cameras present, two views exchanged between input slots
    "swap-wt|--camera_slots swap_wrist_top"
    "swap-cw|--camera_slots swap_context_wrist"
    "swap-ct|--camera_slots swap_context_top"
    # distractors, with visual-only twins (physical obstruction vs distraction)
    "sph-2flank|--spheres two_flanking"
    "sph-2flank-ghost|--spheres two_flanking_ghost"
    "sph-colormatch|--spheres color_match"
    "sph-colormatch-ghost|--spheres color_match_ghost"
)
# Dose-response: several magnitudes per axis -> fit TSR vs magnitude curves.
DOSE=(
    "wpos-1.5cm|--pos_target wrist --pos_offset -0.015 0 0"
    "wpos-0.75cm|--pos_target wrist --pos_offset -0.0075 0 0"
    "wpos+0.75cm|--pos_target wrist --pos_offset 0.0075 0 0"
    "wposy-1.5cm|--pos_target wrist --pos_offset 0 -0.015 0"
    "wposy+1.5cm|--pos_target wrist --pos_offset 0 0.015 0"
    "wposz-1.5cm|--pos_target wrist --pos_offset 0 0 -0.015"
    "wposz+1.5cm|--pos_target wrist --pos_offset 0 0 0.015"
    "wtilt-2.5|--tilt_target wrist --tilt_deg -2.5"
    "wtilt+2.5|--tilt_target wrist --tilt_deg 2.5"
    "wtilt-5|--tilt_target wrist --tilt_deg -5"
    "wtilt+5|--tilt_target wrist --tilt_deg 5"
    "wtilt-15|--tilt_target wrist --tilt_deg -15"
    "wtilt+15|--tilt_target wrist --tilt_deg 15"
    "ctxpos+1.5cm|--pos_target context --pos_offset 0.015 0 0"
    "ctxpos+3.0cm|--pos_target context --pos_offset 0.03 0 0"
    "ctxtilt+5|--tilt_target context --tilt_deg 5"
    "ctxtilt+10|--tilt_target context --tilt_deg 10"
    "toppos+3.0cm|--pos_target top --pos_offset 0.03 0 0"
    "toptilt+10|--tilt_target top --tilt_deg 10"
)
# Controls: is a blanked camera harmful because information is missing, or
# because a black frame is out of distribution? Plus colour / language.
CONTROLS=(
    "no-wrist-gray|--blackout_mode no_wrist --blank_fill gray"
    "no-wrist-noise|--blackout_mode no_wrist --blank_fill noise"
    "no-top-gray|--blackout_mode no_top --blank_fill gray"
    "no-ctx-gray|--blackout_mode no_context --blank_fill gray"
    "cube-blue|--cube_color blue"
    "cube-yellow|--cube_color yellow"
)

case "$CONDITION_SET" in
    core)     CONDITIONS=("${CORE[@]}") ;;
    dose)     CONDITIONS=("${DOSE[@]}") ;;
    controls) CONDITIONS=("${CONTROLS[@]}") ;;
    all)      CONDITIONS=("${CORE[@]}" "${DOSE[@]}" "${CONTROLS[@]}") ;;
    *) echo "[fatal] CONDITION_SET must be core|dose|controls|all"; exit 1 ;;
esac
SMOKE_FIRST=" abs-top abs-w+t abs-ctx abs-wrist "

run_one() {
    local label="$1" k="$2"; shift 2
    local extra_args="$*"
    local log_file="${LOG_DIR}/k${k}_${label}.log"
    local t0 t1 elapsed status rc
    echo ""
    echo "== k=${k}  ${label}   (args: ${extra_args})"
    t0=$(date +%s)
    # shellcheck disable=SC2086
    python "$EVAL" $COMMON --n_action_steps "$k" $VIDEO_ARGS $extra_args > "$log_file" 2>&1
    rc=$?
    t1=$(date +%s); elapsed=$((t1 - t0))
    if [ $rc -eq 0 ]; then
        status="OK"; echo "   OK  (${elapsed}s)"
    elif grep -q "already exists" "$log_file"; then
        status="ALREADY_DONE"; echo "   = already done, skipped"
    else
        status="FAILED"; echo "   FAILED (exit $rc) -- see $log_file"; tail -15 "$log_file" | sed 's/^/     /'
    fi
    printf '%s,%s,%s,"%s",%s,%s,%s\n' "$(date -Iseconds)" "$label" "$k" "$extra_args" \
        "$status" "$elapsed" "$log_file" >> "$SWEEP_LOG"
}

smoke_test() {
    local label="$1" k="$2"; shift 2
    local extra_args="$*"
    local smoke_log="${LOG_DIR}/k${k}_${label}_smoke.log"
    # shellcheck disable=SC2086
    python "$EVAL" $COMMON --n_action_steps "$k" $extra_args --n 3 --video_episodes none \
        --out "${LOG_DIR}/k${k}_${label}_smoke.csv" --overwrite > "$smoke_log" 2>&1
    if [ $? -ne 0 ]; then
        echo "   SMOKE TEST FAILED for ${label} (k=${k}) -- skipping. See $smoke_log"
        printf '%s,%s,%s,"%s",%s,%s,%s\n' "$(date -Iseconds)" "$label" "$k" "$extra_args" \
            "SMOKE_FAILED" "0" "$smoke_log" >> "$SWEEP_LOG"
        return 1
    fi
    return 0
}

for entry in "${CONDITIONS[@]}"; do
    label="${entry%%|*}"
    extra_args="${entry#*|}"
    for k in $HORIZONS; do
        if [[ "$SMOKE_FIRST" == *" ${label} "* ]]; then
            smoke_test "$label" "$k" $extra_args && run_one "$label" "$k" $extra_args
        else
            run_one "$label" "$k" $extra_args
        fi
    done
done

echo ""
echo "Sweep finished. runs/${RUN_ROOT_TAG}/sweep_log.csv:"
column -s, -t "$SWEEP_LOG" 2>/dev/null || cat "$SWEEP_LOG"
FAILED_COUNT=$(grep -c ',FAILED,\|,SMOKE_FAILED,' "$SWEEP_LOG" || true)
if [ "${FAILED_COUNT:-0}" -gt 0 ]; then
    echo "[warn] ${FAILED_COUNT} run(s) failed -- inspect logs before analysis."
else
    echo "Next: python icra/analyze_paper.py --run_root runs/${RUN_ROOT_TAG}"
fi