#!/bin/bash
# Collect 250 episodes for all 10 tasks with full domain randomization.
# Runs 1 task per GPU (2 at a time) to avoid VRAM contention / Vulkan crashes.
# Tasks are processed in waves of 2; already-completed tasks are skipped.
# language_annotation.json is saved automatically alongside HDF5 files.
#
# Output: data/{task}/train_diverse/
#   ├── language_annotation.json   ← move-level boundaries (DO NOT DELETE)
#   ├── scene_info.json
#   ├── seed.txt
#   └── data/episode{N}.hdf5

TASK_CONFIG="train_diverse"
LOG_DIR="logs/collect_${TASK_CONFIG}"
mkdir -p "${LOG_DIR}"

[ -f ./script/.update_path.sh ] && ./script/.update_path.sh > /dev/null 2>&1 || true

TASKS=(
    battery_try
    observe_and_pickup
    cover_blocks
    press_button
    place_block_mat
    put_back_block
    rearrange_blocks
    swap_blocks
    swap_T
    blocks_ranking_try
)

TASKS_PER_WAVE=2   # 1 per GPU — prevents VRAM contention
NUM_GPUS=2

echo "======================================================"
echo "Collecting ${#TASKS[@]} tasks × 250 episodes"
echo "Config: ${TASK_CONFIG} (full randomization)"
echo "Mode: ${TASKS_PER_WAVE} tasks at a time (1 per GPU)"
echo "Logs: ${LOG_DIR}/"
echo "======================================================"
echo ""

ALL_OK=true
pending=()
for task in "${TASKS[@]}"; do
    completed=$(ls data/${task}/${TASK_CONFIG}/data/episode*.hdf5 2>/dev/null | wc -l)
    if [ "${completed}" -ge 250 ]; then
        echo "SKIP: ${task} already has ${completed} episodes"
    else
        pending+=("${task}")
    fi
done

echo "${#pending[@]} tasks to collect: ${pending[*]}"
echo ""

# Worker-pool: keep NUM_GPUS slots busy at all times.
# gpu_pid[g] = PID of task currently running on GPU g (0 = free)
# gpu_task[g] = task name on GPU g
declare -a gpu_pid=()
declare -a gpu_task=()
for (( g=0; g<NUM_GPUS; g++ )); do gpu_pid[$g]=0; gpu_task[$g]=""; done

next=0   # index into pending[]

launch_on_gpu() {
    local g=$1
    local task="${pending[$next]}"
    next=$((next + 1))
    local log_file="${LOG_DIR}/${task}.log"
    echo "Starting: ${task}  (GPU ${g}) → ${log_file}"
    CUDA_VISIBLE_DEVICES=${g} \
    PYTHONWARNINGS=ignore::UserWarning \
    /home/luhr/miniconda3/envs/RMBench/bin/python script/collect_data.py "${task}" "${TASK_CONFIG}" \
        >> "${log_file}" 2>&1 &
    gpu_pid[$g]=$!
    gpu_task[$g]="${task}"
}

# Fill all GPU slots initially
for (( g=0; g<NUM_GPUS && next<${#pending[@]}; g++ )); do
    launch_on_gpu $g
done

# Keep polling until all tasks are dispatched and finished
while true; do
    all_idle=true
    for (( g=0; g<NUM_GPUS; g++ )); do
        pid="${gpu_pid[$g]}"
        [ "$pid" -eq 0 ] && continue
        all_idle=false

        if ! kill -0 "$pid" 2>/dev/null; then
            # Process finished — collect exit code
            wait "$pid"
            exit_code=$?
            task="${gpu_task[$g]}"
            if [ $exit_code -eq 0 ]; then
                n=$(ls data/${task}/${TASK_CONFIG}/data/*.hdf5 2>/dev/null | wc -l)
                echo "[DONE]   ${task} (${n} episodes)"
                rm -rf "data/${task}/${TASK_CONFIG}/.cache"
            else
                echo "[FAILED] ${task} (exit ${exit_code}) — check ${LOG_DIR}/${task}.log"
                ALL_OK=false
            fi
            gpu_pid[$g]=0
            gpu_task[$g]=""

            # Immediately assign next pending task to this GPU
            if [ $next -lt ${#pending[@]} ]; then
                launch_on_gpu $g
                all_idle=false
            fi
        fi
    done

    # Check if everything is done
    active=0
    for (( g=0; g<NUM_GPUS; g++ )); do [ "${gpu_pid[$g]}" -ne 0 ] && active=$((active+1)); done
    [ $active -eq 0 ] && [ $next -ge ${#pending[@]} ] && break

    sleep 10
done

echo "======================================================"
if $ALL_OK; then
    echo "All tasks completed. Data saved to:"
    for task in "${TASKS[@]}"; do
        n=$(ls data/${task}/${TASK_CONFIG}/data/*.hdf5 2>/dev/null | wc -l)
        echo "  data/${task}/${TASK_CONFIG}/  (${n} episodes)"
    done
    echo ""
    echo "IMPORTANT: Keep language_annotation.json files — needed for move-level annotation!"
else
    echo "Some tasks failed. Check logs in ${LOG_DIR}/"
    exit 1
fi
echo "======================================================"
