#!/bin/bash
# Copyright (c) 2024, PhysREPA Project.
# SPDX-License-Identifier: BSD-3-Clause
#
# Train PhysREPA RL manipulation tasks using RSL-RL PPO.
#
# Usage:
#   ./train_rl.sh PhysREPA-PickPlace-Franka-v0
#   ./train_rl.sh PhysREPA-Push-Franka-v0
#   ./train_rl.sh PhysREPA-Stack-Franka-v0
#
# Or train all three sequentially:
#   ./train_rl.sh all
#
# Custom args:
#   ./train_rl.sh PhysREPA-PickPlace-Franka-v0 4096 3000

set -euo pipefail

TASK=${1:-"PhysREPA-PickPlace-Franka-v0"}
NUM_ENVS=${2:-4096}
MAX_ITERATIONS=${3:-3000}

# Ensure physrepa_tasks is importable
export PYTHONPATH="/home/solee:${PYTHONPATH:-}"

ISAAC_LAB_DIR="/home/solee/IsaacLab"
TRAIN_SCRIPT="${ISAAC_LAB_DIR}/scripts/reinforcement_learning/rsl_rl/train.py"

run_training() {
    local task_name=$1
    local iters=$2
    echo ""
    echo "============================================"
    echo "  Training: ${task_name}"
    echo "  Envs: ${NUM_ENVS} | Iterations: ${iters}"
    echo "============================================"
    echo ""
    cd "${ISAAC_LAB_DIR}"
    # Use -c to import physrepa_tasks before running the train script.
    # This ensures gymnasium task registrations happen before the script
    # tries to look up the task by name.
    ./isaaclab.sh -p -c "
import physrepa_tasks.rl_envs  # register PhysREPA tasks
import runpy, sys
sys.argv = ['train.py', '--task=${task_name}', '--num_envs=${NUM_ENVS}', '--headless', '--max_iterations=${iters}']
sys.path.insert(0, '${ISAAC_LAB_DIR}/scripts/reinforcement_learning/rsl_rl')
runpy.run_path('${TRAIN_SCRIPT}', run_name='__main__')
"
}

if [ "$TASK" = "all" ]; then
    echo "=== Training all PhysREPA RL tasks ==="
    run_training "PhysREPA-PickPlace-Franka-v0" 3000
    run_training "PhysREPA-Push-Franka-v0" 3000
    run_training "PhysREPA-Stack-Franka-v0" 5000
else
    run_training "$TASK" "$MAX_ITERATIONS"
fi

echo ""
echo "=== Training complete ==="
