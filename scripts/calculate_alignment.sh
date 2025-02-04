#!/bin/bash

# local folders
script_dir=$(dirname "$(realpath "$0")")
local_code_folder="$(dirname "$script_dir")/evaluate"
# main folders
python_instance="shifter --module=gpu,nccl-2.18 --image=asnaylor/lmntfy:v0.3 python3"

# runs the worker
# Using python_instance to run the chatbot script in code_folder
$python_instance $local_code_folder/calculate_alignment.py --curr_dir $local_code_folder
