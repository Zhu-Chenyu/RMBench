#!/bin/bash
# Collect 200 episodes for all 10 tasks in parallel.
# Tasks are spread across 2 GPUs. At most 5 tasks run concurrently per GPU,
# which keeps CPU usage at roughly half the available cores.

TASK_CONFIG="diverse"
LOG_DIR="logs/collect_${TASK_CONFIG}"
mkdir -p "${LOG_DIR}"

./script/.update_path.sh > /dev/null 2>&1

TASKS=(
    cover_blocks
    swap_blocks
    rearrange_blocks
    place_block_mat
    press_button
    observe_and_pickup
    put_back_block
    battery_try
    blocks_ranking_try
    swap_T
)

NUM_GPUS=2
NUM_TASKS=${#TASKS[@]}

echo "Collecting data for ${NUM_TASKS} tasks with config '${TASK_CONFIG}' using ${NUM_GPUS} GPUs."
echo "Logs will be written to ${LOG_DIR}/"
echo ""

PIDS=()
for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"
    gpu_id=$((i % NUM_GPUS))
    log_file="${LOG_DIR}/${task}.log"
    echo "Starting: ${task}  (GPU ${gpu_id})  -> ${log_file}"
    CUDA_VISIBLE_DEVICES=${gpu_id} \
    PYTHONWARNINGS=ignore::UserWarning \
    python script/collect_data.py "${task}" "${TASK_CONFIG}" \
        > "${log_file}" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All ${NUM_TASKS} collection jobs launched (PIDs: ${PIDS[*]})."
echo "Waiting for all jobs to finish..."

ALL_OK=true
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    task="${TASKS[$i]}"
    wait "${pid}"
    exit_code=$?
    if [ "${exit_code}" -ne 0 ]; then
        echo "[FAILED] ${task} (exit code ${exit_code})"
        ALL_OK=false
    else
        echo "[DONE]   ${task}"
        rm -rf "data/${task}/${TASK_CONFIG}/.cache"
    fi
done

if $ALL_OK; then
    echo ""
    echo "All tasks completed successfully."
else
    echo ""
    echo "Some tasks failed. Check logs in ${LOG_DIR}/ for details."
    exit 1
fi
