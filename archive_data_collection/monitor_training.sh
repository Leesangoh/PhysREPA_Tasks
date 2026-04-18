#!/bin/bash
# Autonomous training monitor — checks every 2 hours, resumes finished tasks
# Writes status to /tmp/training_monitor.log

LOG="/tmp/training_monitor.log"
ISAAC_DIR="/home/solee/IsaacLab"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"; }

resume_task() {
    local task=$1
    local task_name=$2
    local device=$3
    local max_iter=$4
    
    # Find latest run dir
    local run_dir=$(ls -td ${ISAAC_DIR}/logs/rsl_rl/physrepa_${task}/2026* 2>/dev/null | head -1)
    if [ -z "$run_dir" ]; then
        log "ERROR: No run dir found for ${task}"
        return 1
    fi
    local run_name=$(basename "$run_dir")
    
    log "Resuming ${task} from ${run_dir} with ${max_iter} more iterations"
    
    tmux kill-session -t train_${task} 2>/dev/null
    tmux new-session -d -s train_${task}
    tmux send-keys -t train_${task} "cd ${ISAAC_DIR} && PYTHONPATH=/home/solee:\$PYTHONPATH ./isaaclab.sh -p -c '
import physrepa_tasks.rl_envs
import runpy, sys
sys.argv = [\"train.py\", \"--task=${task_name}\", \"--num_envs=4096\", \"--headless\", \"--max_iterations=${max_iter}\", \"--device=${device}\", \"--resume\", \"--load_run=${run_name}\"]
sys.path.insert(0, \"${ISAAC_DIR}/scripts/reinforcement_learning/rsl_rl\")
runpy.run_path(\"${ISAAC_DIR}/scripts/reinforcement_learning/rsl_rl/train.py\", run_name=\"__main__\")
' 2>&1 | tee /tmp/train_${task}.log" Enter
}

check_and_resume() {
    for task_info in "pick_place:PhysREPA-PickPlace-Franka-v0:cuda:0:15000" \
                     "push:PhysREPA-Push-Franka-v0:cuda:1:15000" \
                     "stack:PhysREPA-Stack-Franka-v0:cuda:2:20000"; do
        IFS=: read -r task task_name device max_iter <<< "$task_info"
        
        # Check if tmux session exists and process is running
        if ! tmux has-session -t "train_${task}" 2>/dev/null; then
            log "${task}: tmux session dead — resuming"
            resume_task "$task" "$task_name" "$device" "$max_iter"
            continue
        fi
        
        # Check if training process is still alive
        if ! ps aux | grep -v grep | grep "task=${task_name}" > /dev/null 2>&1; then
            # Process might have finished or crashed
            local last_reward=$(grep "Mean reward" /tmp/train_${task}.log 2>/dev/null | tail -1 | awk '{print $NF}')
            local last_error=$(grep "position_error" /tmp/train_${task}.log 2>/dev/null | tail -1 | awk '{print $NF}')
            local has_error=$(grep -c "RuntimeError\|Exception\|Error executing" /tmp/train_${task}.log 2>/dev/null)
            
            log "${task}: process not running. Last reward=${last_reward}, error=${last_error}, crashes=${has_error}"
            
            if [ "$has_error" -gt 0 ]; then
                log "${task}: had errors — resuming anyway"
            fi
            resume_task "$task" "$task_name" "$device" "$max_iter"
        else
            local reward=$(grep "Mean reward" /tmp/train_${task}.log 2>/dev/null | tail -1 | awk '{print $NF}')
            local pos_err=$(grep "position_error" /tmp/train_${task}.log 2>/dev/null | tail -1 | awk '{print $NF}')
            local eta=$(grep "ETA" /tmp/train_${task}.log 2>/dev/null | tail -1 | awk '{print $NF}')
            log "${task}: running. reward=${reward}, pos_error=${pos_err}, ETA=${eta}"
        fi
    done
}

# Main loop — run for 16 hours (8 checks × 2 hour intervals)
log "=== Training monitor started ==="
log "Will monitor for 16 hours with 2-hour intervals"

for i in $(seq 1 8); do
    log "--- Check ${i}/8 ---"
    check_and_resume
    
    if [ $i -lt 8 ]; then
        log "Sleeping 2 hours..."
        sleep 7200
    fi
done

log "=== Training monitor finished (16 hours) ==="

# Final status
log "=== FINAL STATUS ==="
for task in pick_place push stack; do
    reward=$(grep "Mean reward" /tmp/train_${task}.log 2>/dev/null | tail -1 | awk '{print $NF}')
    pos_err=$(grep "position_error" /tmp/train_${task}.log 2>/dev/null | tail -1 | awk '{print $NF}')
    log "${task}: reward=${reward}, pos_error=${pos_err}"
    
    # Find best checkpoint
    run_dir=$(ls -td ${ISAAC_DIR}/logs/rsl_rl/physrepa_${task}/2026* 2>/dev/null | head -1)
    latest=$(ls -t ${run_dir}/model_*.pt 2>/dev/null | head -1)
    log "${task}: best checkpoint=${latest}"
done
