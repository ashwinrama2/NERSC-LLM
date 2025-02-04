#!/bin/bash

# base folder
chatbot_root="/global/cfs/cdirs/nstaff/chatbot"
# local folders
script_dir=$(dirname "$(realpath "$0")")
local_code_folder="$(dirname "$script_dir")/evaluate"
# main folders
code_folder="$chatbot_root/production_code"
models_folder="$chatbot_root/models"
data_folder="$code_folder/data"
# data folders
documentation_folder="$data_folder/nersc_doc/docs"
database_folder="$data_folder/database"
python_instance="shifter --module=gpu,nccl-2.18 --image=asnaylor/lmntfy:v0.3 python3"

overwrite_chunks=0
overwrite_answers=0

# Run Python script to generate common chunks
$python_instance $local_code_folder/generate_test_chunks.py --docs_folder $documentation_folder --database_folder $database_folder --models_folder $models_folder --curr_dir $local_code_folder --overwrite $overwrite_chunks

# Generate model names using the Python script and read them into an array
readarray -t model_names < <($python_instance $local_code_folder/find_chatbot_models.py)

# Print out all the model names
model_names_str=$(printf ", %s" "${model_names[@]}")
#printf "Checking models:%s\n" "${model_names_str:1}"
#printf "\n"

# Loop over each model name and run the Python script
for model_name in "${model_names[@]}"
do
    $python_instance $local_code_folder/generate_test_answers.py --models_folder $models_folder --curr_dir $local_code_folder --model_name $model_name --overwrite $overwrite_answers
done
