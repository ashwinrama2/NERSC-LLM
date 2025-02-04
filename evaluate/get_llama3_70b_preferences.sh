#!/usr/bin/env bash

#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=asrama@lbl.gov

#SBATCH --job-name gen_test
#SBATCH --output judge.%j.out
#SBATCH --error judge.%j.err

#SBATCH -C gpu
#SBATCH --time=00:30:00
#SBATCH -q debug
#SBATCH -A nstaff
#SBATCH --image=vllm/vllm-openai:v0.4.2
#SBATCH --module=gpu,nccl-2.18

### This script works for any number of nodes, Ray will find and manage all resources
#SBATCH --nodes=2

### Give all resources to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=128

## Setup
echo "[slurm] - Run with shifter"
shifter --image=vllm/vllm-openai:v0.4.2 --module=gpu,nccl-2.18 python3 --version
shifter --image=vllm/vllm-openai:v0.4.2 --module=gpu,nccl-2.18 ray --version
export HF_HOME=$SCRATCH/huggingface/

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )
node_1_ip=${nodes_array[0]} 
RAY_HEAD_PORT=6379
ip_head=$node_1_ip:$RAY_HEAD_PORT
RAY_NODE_ADDRESS=$ip_head
export RAY_NODE_ADDRESS
echo "[slurm] - IP Head: $RAY_NODE_ADDRESS"

## Start ray Head
echo "[slurm] - Starting ray HEAD"
srun --nodes=1 --ntasks=1 -w $node_1_ip shifter --image=vllm/vllm-openai:v0.4.2 --module=gpu,nccl-2.18 ray start --head --node-ip-address=$node_1_ip --port=$RAY_HEAD_PORT --block &
sleep 30 ##needed?

## Start ray worker
worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
echo "[slurm] - Starting $worker_num ray worker"
for ((  i=1; i<=$worker_num; i++ ))
do
  node_i=${nodes_array[$i]}
  echo "    - $i at $node_i"
  srun --nodes=1 --ntasks=1 -w $node_i shifter --image=vllm/vllm-openai:v0.4.2 --module=gpu,nccl-2.18 ray start --address $RAY_NODE_ADDRESS --block &
done

## Execute ray code
echo "[slurm] Executing ray code..."
sleep 10

script_dir=$(dirname "$(realpath "$0")")
local_code_folder="$(dirname "$script_dir")/evaluate"
overwrite=0
python_instance="shifter --module=gpu,nccl-2.18 --image=asnaylor/lmntfy:v0.3 python3"

$python_instance $local_code_folder/get_llama3_70b_preferences.py --curr_dir $local_code_folder --overwrite $overwrite

echo "[slurm] Exiting slurm script..."
exit
