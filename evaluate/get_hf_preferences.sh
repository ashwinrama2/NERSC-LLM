#!/usr/bin/env bash

#Configs
export MODEL="google/gemma-2-27b-it"
export NGPUS=4
export VLLM_ATTENTION_BACKEND="FLASHINFER"
export PYTHONUSERBASE="$SCRATCH/tmp/vllm_python"
export HF_HOME="$SCRATCH/huggingface"
export VLLM_NCCL_SO_PATH="/opt/udiImage/modules/nccl-2.18/lib/libnccl.so.2"

SHIFTER="shifter --image=vllm/vllm-openai:v0.5.3 --module=gpu,nccl-2.18"

#Setup Ray Head
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )
node_1_ip=${nodes_array[0]}
RAY_HEAD_PORT=6379
ip_head=$node_1_ip:$RAY_HEAD_PORT
RAY_NODE_ADDRESS=$ip_head
export RAY_NODE_ADDRESS
echo "[ ] Starting ray head"
srun --nodes=1 --ntasks=1 -w $node_1_ip $SHIFTER \
    ray start --head --node-ip-address=$node_1_ip --port=$RAY_HEAD_PORT --block &
sleep 30

#Run python
echo "[ ] Running vllm"
script_dir=$(dirname "$(realpath "$0")")
local_code_folder="$(dirname "$script_dir")/evaluate"
overwrite=0

$SHIFTER python3 get_hf_preferences.py --curr_dir $local_code_folder --overwrite $overwrite
