#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/solee/physrepa_tasks"
DATA_BASE="/mnt/md1/solee/data/isaac_physrepa_native_force_recollect"
LOG_DIR="$ROOT/artifacts/logs/native_force_multiseed"
MASTER_LOG="$LOG_DIR/master.log"
mkdir -p "$LOG_DIR"

timestamp() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

run_job() {
  local gpu="$1"
  local model="$2"
  local feature_root="$3"
  local seed="$4"
  local run_tag="$5"
  local limit="$6"
  local log_file="$LOG_DIR/${run_tag}.log"
  local result_csv="$ROOT/artifacts/results/probe_events_strike_contact_force_native_${model}_${run_tag}.csv"

  if [[ -f "$result_csv" ]]; then
    echo "[$(timestamp)] SKIP $run_tag (result exists)" | tee -a "$MASTER_LOG" >> "$log_file"
    return 0
  fi

  echo "[$(timestamp)] START $run_tag gpu=$gpu model=$model feature_root=$feature_root limit=$limit" | tee -a "$MASTER_LOG" >> "$log_file"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
    /isaac-sim/python.sh "$ROOT/probe_events.py" \
      --task strike \
      --model "$model" \
      --feature-root "$feature_root" \
      --data-base "$DATA_BASE" \
      --label-mode native \
      --device cuda:0 \
      --run-tag "$run_tag" \
      --probe-seed "$seed" \
      --episode-limit "$limit" \
      >> "$log_file" 2>&1
  echo "[$(timestamp)] DONE $run_tag" | tee -a "$MASTER_LOG" >> "$log_file"
}

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <gpu> <model> <feature_root> <episode_limit>" >&2
  exit 1
fi

GPU="$1"
MODEL="$2"
FEATURE_ROOT="$3"
LIMIT="$4"

case "$MODEL" in
  large)
    PREFIX="phase3_events_native_force_multiseed_large_seed"
    ;;
  videomae_large)
    PREFIX="phase3_events_native_force_multiseed_videomae_seed"
    ;;
  dinov2_large)
    PREFIX="phase3_events_native_force_multiseed_dino_seed"
    ;;
  *)
    echo "Unsupported model: $MODEL" >&2
    exit 1
    ;;
esac

for SEED in 42 123 2024; do
  run_job "$GPU" "$MODEL" "$FEATURE_ROOT" "$SEED" "${PREFIX}${SEED}" "$LIMIT"
done
