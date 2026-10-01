#!/bin/bash
set -euo pipefail

policy_name=GR00T
task_name=${1}
task_config=${2}
ckpt_setting=${3}
seed=${4}
gpu_id=${5}
model_path=${6}
execute_horizon=${7:-8}
test_num=${8:-100}
model_host=${9:-}
model_port=${10:-5555}
memory_url=${MEMORY_URL:-}
memory_log_dir=${MEMORY_LOG_DIR:-}
memory_max_chunks=${MEMORY_MAX_CHUNKS:-0}
memory_timeout_s=${MEMORY_TIMEOUT_S:-20}
memory_language_max_chars=${MEMORY_LANGUAGE_MAX_CHARS:-0}
memory_long_budget=${MEMORY_LONG_BUDGET:-1500}
memory_short_budget=${MEMORY_SHORT_BUDGET:-2300}
use_memory_as_language=${USE_MEMORY_AS_LANGUAGE:-False}

export CUDA_VISIBLE_DEVICES=${gpu_id}
export GR00T_REPO=${GR00T_REPO:-/home/dfv1344/Isaac-GR00T}
export PYTHONPATH=${GR00T_REPO}:${PYTHONPATH:-}

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mGR00T model path: ${model_path}\033[0m"
if [[ -n "${model_host}" ]]; then
    echo -e "\033[33mGR00T server: ${model_host}:${model_port}\033[0m"
fi
if [[ "${use_memory_as_language}" == "True" || "${use_memory_as_language}" == "true" || -n "${memory_url}" ]]; then
    echo -e "\033[33mMemory URL: ${memory_url}\033[0m"
    echo -e "\033[33mUse memory as language: ${use_memory_as_language}\033[0m"
    echo -e "\033[33mMemory log dir: ${memory_log_dir}\033[0m"
    echo -e "\033[33mMemory action-input max chars: ${memory_language_max_chars}\033[0m"
fi

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --ckpt_setting "${ckpt_setting}" \
    --seed "${seed}" \
    --policy_name "${policy_name}" \
    --model_path "${model_path}" \
    --model_host "${model_host}" \
    --model_port "${model_port}" \
    --device "cuda" \
    --execute_horizon "${execute_horizon}" \
    --test_num "${test_num}" \
    --language_mode "instruction" \
    --memory_url "${memory_url}" \
    --memory_log_dir "${memory_log_dir}" \
    --memory_max_chunks "${memory_max_chunks}" \
    --memory_timeout_s "${memory_timeout_s}" \
    --memory_language_max_chars "${memory_language_max_chars}" \
    --memory_long_budget "${memory_long_budget}" \
    --memory_short_budget "${memory_short_budget}" \
    --use_memory_as_language "${use_memory_as_language}"
