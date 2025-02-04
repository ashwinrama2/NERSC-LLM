#!/bin/bash

# base folder
chatbot_root="/global/cfs/cdirs/nstaff/chatbot"
# local folders
script_dir=$(dirname "$(realpath "$0")")
local_code_folder="$(dirname "$script_dir")/evaluate"
# main folders
models_folder="$chatbot_root/models"
# data folders
python_instance="shifter --module=gpu,nccl-2.18 --image=asnaylor/lmntfy:v0.3 python3"

overwrite=0

# Generate judge names using the Python script and read them into an array
readarray -t judge_names < <($python_instance $local_code_folder/find_judge_models.py)

# Print out all the judge names
judge_names_str=$(printf ", %s" "${judge_names[@]}")
printf "Checking judges:%s\n" "${judge_names_str:1}"
printf "\n"

# Loop over each judge name and run the Python script
for judge_name in "${judge_names[@]}"
do
    $python_instance $local_code_folder/get_judge_preferences.py --models_folder $models_folder --judge_name $judge_name --curr_dir $local_code_folder --overwrite $overwrite
done
