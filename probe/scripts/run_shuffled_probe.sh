#!/bin/bash
# Launches the F5 shuffled-features probe sweep for a given task using
# 03_run_probe.py with --variant A_shuffled.
#
# Usage:
#   run_shuffled_probe.sh <task> [gpu_csv_list]
#     <task>          push | strike
#     [gpu_csv_list]  optional comma-separated GPU indices to use; default "1,2,3"
#                     (avoids GPU 0 because F5 strike extraction occupies GPU 0).
#
# Sharding: one target per GPU. With 3 GPUs (default), targets prioritised
# ee_acceleration → ee_velocity → contact_force_log1p_mag are launched in
# parallel; contact_flag is appended sequentially via wait-then-pickup.
#
# Pre-check (HARD): variant_A_shuffled cache .npz count MUST equal the
# variant_A reference cache count. Anything less means extraction incomplete
# and we ABORT to preserve the cleanliness of ΔR² = R²(unshuf) − R²(shuf).
#
# Output:
#   /home/solee/physrepa_tasks/probe/results/<task>/variant_A_shuffled/<target>.csv
#   per-fold rows, schema matches main variant_A probe (used by 10_bootstrap_cis.py).
#
# Logs:
#   /home/solee/physrepa_tasks/probe/results/logs/shuffled_probe_<task>_<target>.log
set -e
TASK="${1:?usage: $0 <task> [gpu_csv_list]}"
GPU_CSV="${2:-1,2,3}"
PROBE=/home/solee/physrepa_tasks/probe
LOG_DIR=$PROBE/results/logs
mkdir -p "$LOG_DIR"

# Codex-recommended target priority: dynamics first (acc/vel), force, then flag.
TARGETS=(ee_acceleration ee_velocity contact_force_log1p_mag contact_flag)

# Pre-check: shuffled count == unshuffled (Variant A) count.
ROOT=/mnt/md1/solee/physprobe_features
SHUF_DIR=$ROOT/$TASK/variant_A_shuffled
ORIG_DIR=$ROOT/$TASK/variant_A
N_SHUF=$(ls -1 "$SHUF_DIR"/episode_*.npz 2>/dev/null | wc -l)
N_ORIG=$(ls -1 "$ORIG_DIR"/episode_*.npz 2>/dev/null | wc -l)
if [ "$N_SHUF" -lt 1 ]; then
    echo "[shuf_probe] $TASK: no shuffled .npz files at $SHUF_DIR — ABORT."
    exit 1
fi
if [ "$N_SHUF" -ne "$N_ORIG" ]; then
    echo "[shuf_probe] $TASK: variant_A_shuffled has $N_SHUF files but variant_A has $N_ORIG — extraction INCOMPLETE. ABORT."
    exit 1
fi
echo "[shuf_probe] $TASK: $N_SHUF shuffled .npz files (matches variant_A). Launching probe sweep on GPUs $GPU_CSV."

# Parse GPU list
IFS=',' read -ra GPUS <<<"$GPU_CSV"
N_GPUS=${#GPUS[@]}
N_TARGETS=${#TARGETS[@]}
echo "[shuf_probe] $N_GPUS GPUs available; $N_TARGETS targets queued."

# Round-robin schedule: targets[i] -> GPU[i % N_GPUS].
# If N_TARGETS > N_GPUS, the trailing target(s) wait until a GPU is free.
declare -a JOB_PIDS
declare -a JOB_GPUS
for i in $(seq 0 $((N_TARGETS - 1))); do
    TARGET="${TARGETS[$i]}"
    GPU_IDX=$((i % N_GPUS))
    GPU="${GPUS[$GPU_IDX]}"
    LOG=$LOG_DIR/shuffled_probe_${TASK}_${TARGET}.log

    if [ "$i" -ge "$N_GPUS" ]; then
        # Wait for the earlier job on this GPU to finish.
        WAIT_PID=${JOB_PIDS[$GPU_IDX]}
        echo "[shuf_probe] $TASK / $TARGET waits for PID $WAIT_PID (GPU $GPU)"
        wait $WAIT_PID || echo "[shuf_probe]   warning: PID $WAIT_PID exited non-zero"
    fi

    echo "[shuf_probe] $TASK / $TARGET on GPU $GPU → $LOG"
    CUDA_VISIBLE_DEVICES=$GPU /isaac-sim/python.sh $PROBE/scripts/03_run_probe.py \
        --task "$TASK" \
        --variant A_shuffled \
        --gpu 0 \
        --targets "$TARGET" \
        --layers all \
        > "$LOG" 2>&1 &
    JOB_PIDS[$GPU_IDX]=$!
    JOB_GPUS[$GPU_IDX]=$GPU
    echo "[shuf_probe]   PID ${JOB_PIDS[$GPU_IDX]} on GPU $GPU"
done

echo "[shuf_probe] $TASK: all targets queued. Waiting for remaining jobs..."
for pid in "${JOB_PIDS[@]}"; do
    wait "$pid" || echo "[shuf_probe]   warning: PID $pid exited non-zero"
done
echo "[shuf_probe] $TASK: ALL DONE."
