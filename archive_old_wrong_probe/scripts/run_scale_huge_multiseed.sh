#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/solee/physrepa_tasks"
FEATURE_ROOT="/mnt/md1/solee/features/physprobe_vith_tokenpatch"

run_probe() {
  local gpu="$1"
  local seed="$2"
  env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    /isaac-sim/python.sh "$ROOT/probe_physprobe.py" \
    --task push \
    --model huge \
    --targets ee_direction_3d ee_speed \
    --solver trainable \
    --cv-splits 5 \
    --norm zscore \
    --device cuda:0 \
    --feature-type token_patch \
    --feature-root "$FEATURE_ROOT" \
    --run-tag "scale_huge_seed${seed}" \
    --probe-seed "$seed"
}

run_probe 0 123 &
PID0=$!
run_probe 1 2024 &
PID1=$!

wait "$PID0"
wait "$PID1"
