#!/bin/bash
# Usage: bash eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id>
# Example: bash eval.sh clean_table demo_clean_table base 0 0

policy_name=GR00T_Mem
task_name=${1}
task_config=${2}
ckpt_setting=${3}
seed=${4}
gpu_id=${5}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

# RMBench conda env has sapien + zmq + gr00t source path. The GR00T model
# itself runs in a SEPARATE inference server (started independently from
# Isaac-GR00T/.venv) — see deploy_policy.py docstring.
export PATH="/home/luhr/miniconda3/envs/RMBench/bin:${PATH}"
export GR00T_REPO="/home/luhr/chenyu/final_project/code/Isaac-GR00T"
export PYTHONPATH="${GR00T_REPO}:${PYTHONPATH:-}"

cd ../..  # move to RMBench root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name}
