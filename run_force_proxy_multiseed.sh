#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/solee/physrepa_tasks"
LOG_DIR="$ROOT/artifacts/logs/force_proxy_multiseed"
MASTER_LOG="$LOG_DIR/master.log"
mkdir -p "$LOG_DIR"

timestamp() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

result_csv_for() {
  local model="$1"
  local run_tag="$2"
  echo "$ROOT/artifacts/results/probe_events_strike_contact_force_proxy_${model}_${run_tag}.csv"
}

wait_for_existing_run() {
  local run_tag="$1"
  while pgrep -f "$run_tag" >/dev/null 2>&1; do
    sleep 30
  done
}

run_job() {
  local gpu="$1"
  local model="$2"
  local feature_root="$3"
  local seed="$4"
  local run_tag="$5"
  local limit="$6"
  local log_file="$LOG_DIR/${run_tag}.log"
  local result_csv
  result_csv="$(result_csv_for "$model" "$run_tag")"

  if [[ -f "$result_csv" ]]; then
    echo "[$(timestamp)] SKIP $run_tag (result exists: $result_csv)" | tee -a "$MASTER_LOG" >> "$log_file"
    return 0
  fi

  wait_for_existing_run "$run_tag"

  if [[ -f "$result_csv" ]]; then
    echo "[$(timestamp)] SKIP $run_tag (result appeared while waiting)" | tee -a "$MASTER_LOG" >> "$log_file"
    return 0
  fi

  echo "[$(timestamp)] START $run_tag gpu=$gpu model=$model feature_root=$feature_root limit=$limit" | tee -a "$MASTER_LOG" >> "$log_file"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
    /isaac-sim/python.sh "$ROOT/probe_events.py" \
      --task strike \
      --model "$model" \
      --feature-root "$feature_root" \
      --device cuda:0 \
      --run-tag "$run_tag" \
      --probe-seed "$seed" \
      --episode-limit "$limit" \
      >> "$log_file" 2>&1
  echo "[$(timestamp)] DONE $run_tag" | tee -a "$MASTER_LOG" >> "$log_file"
}

worker_gpu0() {
  run_job 0 large /mnt/md1/solee/features/physprobe_vitl 42 phase3_events_force_multiseed_large_seed42_ep1000 1000
  run_job 0 dinov2_large /mnt/md1/solee/features/physprobe_dinov2_large_tokenpatch 42 phase3_events_force_multiseed_dino_seed42_ep1000 1000
  run_job 0 videomae_large /mnt/md1/solee/features/physprobe_videomae_large_tokenpatch 123 phase3_events_force_multiseed_videomae_seed123_ep1000 1000
  run_job 0 large /mnt/md1/solee/features/physprobe_vitl 2024 phase3_events_force_multiseed_large_seed2024_ep1000 1000
  run_job 0 dinov2_large /mnt/md1/solee/features/physprobe_dinov2_large_tokenpatch 2024 phase3_events_force_multiseed_dino_seed2024_ep1000 1000
}

worker_gpu1() {
  run_job 1 videomae_large /mnt/md1/solee/features/physprobe_videomae_large_tokenpatch 42 phase3_events_force_multiseed_videomae_seed42_ep1000 1000
  run_job 1 large /mnt/md1/solee/features/physprobe_vitl 123 phase3_events_force_multiseed_large_seed123_ep1000 1000
  run_job 1 dinov2_large /mnt/md1/solee/features/physprobe_dinov2_large_tokenpatch 123 phase3_events_force_multiseed_dino_seed123_ep1000 1000
  run_job 1 videomae_large /mnt/md1/solee/features/physprobe_videomae_large_tokenpatch 2024 phase3_events_force_multiseed_videomae_seed2024_ep1000 1000
}

worker_gpu0 &
PID0=$!
worker_gpu1 &
PID1=$!

wait "$PID0"
wait "$PID1"
