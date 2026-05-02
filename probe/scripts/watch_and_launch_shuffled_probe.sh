#!/bin/bash
# Watcher: polls extraction completion for a given task. When the variant_A
# and variant_A_shuffled cache counts match (extraction DONE), auto-launches
# run_shuffled_probe.sh <task>.
#
# Usage:
#   watch_and_launch_shuffled_probe.sh <task> [gpu_csv]
#
# Polls every 60 s, exits when the launcher returns or when the launcher
# refuses (still incomplete).
set -e
TASK="${1:?usage: $0 <task> [gpu_csv]}"
GPU_CSV="${2:-1,2,3}"
PROBE=/home/solee/physrepa_tasks/probe
ROOT=/mnt/md1/solee/physprobe_features
LOG=$PROBE/results/logs/watch_${TASK}.log

while true; do
    N_SHUF=$(ls -1 "$ROOT/$TASK/variant_A_shuffled"/episode_*.npz 2>/dev/null | wc -l)
    N_ORIG=$(ls -1 "$ROOT/$TASK/variant_A"/episode_*.npz 2>/dev/null | wc -l)
    TS=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
    echo "$TS [watch_${TASK}] shuffled=$N_SHUF orig=$N_ORIG" | tee -a "$LOG"
    if [ "$N_SHUF" -ge 1 ] && [ "$N_SHUF" -eq "$N_ORIG" ]; then
        echo "$TS [watch_${TASK}] cache equality reached — launching shuffled probe" | tee -a "$LOG"
        bash $PROBE/scripts/run_shuffled_probe.sh "$TASK" "$GPU_CSV" 2>&1 | tee -a "$LOG"
        echo "$TS [watch_${TASK}] DONE." | tee -a "$LOG"
        exit 0
    fi
    sleep 60
done
