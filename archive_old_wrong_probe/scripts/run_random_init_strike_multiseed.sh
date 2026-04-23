#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/solee/physrepa_tasks"
FEATURE_ROOT="/mnt/md1/solee/features/physprobe_vitl_tokenpatch_randominit_seed0"

run_probe() {
  local gpu="$1"
  local seed="$2"
  local tag="rev1_randominit_strike_seed${seed}"
  env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    /isaac-sim/python.sh "$ROOT/probe_physprobe.py" \
    --task strike \
    --model large \
    --targets ee_direction_3d object_direction_3d ee_speed \
    --solver trainable \
    --cv-splits 5 \
    --norm zscore \
    --device cuda:0 \
    --feature-type token_patch \
    --feature-root "$FEATURE_ROOT" \
    --run-tag "$tag" \
    --probe-seed "$seed"
}

run_probe 0 42 &
PID0=$!
run_probe 1 123 &
PID1=$!
run_probe 2 2024 &
PID2=$!

wait "$PID0"
wait "$PID1"
wait "$PID2"
