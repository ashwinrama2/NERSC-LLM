import sys;
import os;
import os.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')));
import argparse
import asyncio
import json
import inspect
import lmntfy.models.llm as module
from pathlib import Path
from itertools import permutations
from random import shuffle
from metrics import calculate_elo 
from find_chatbot_models import get_chatbot_model_names
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM

import ray
ray.init()

NGPUS=8



def parse_args():
    # Read the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--curr_dir", default=".", type=Path, help="path to the folder containing test questions")
    parser.add_argument("--overwrite", default=0, type=int, help="whether or not to re-compare and overwrite pairwise contest data")
    args = parser.parse_args()
    return args

async def main():
    # process command line arguments
    args = parse_args()
    judge_name = "Llama3_70b"
    curr_dir = args.curr_dir
    overwrite = args.overwrite


    models_list = get_chatbot_model_names()
    models_dict = {model_name:read_json_to_dict(f'{curr_dir}/test_answers/{model_name}_answers.json') for model_name in models_list} #store the answer dictionary with its corresponding model's name

    #load the test questions into a dictionary
    test_questions_dict = read_json_to_dict(f'{curr_dir}/test_questions.json')

    #load the test chunks into a dictionary
    test_chunks_dict = read_json_to_dict(f'{curr_dir}/test_chunks.json')

    os.makedirs(f'{curr_dir}/judge_data/contests', exist_ok=True)
    os.makedirs(f'{curr_dir}/judge_data/elo_scores', exist_ok=True)
    
    #loads contest data into a list of dictionaries
    if overwrite or not os.path.isfile(f'{curr_dir}/judge_data/contests/{judge_name}_contest_data.json'):
        contest_data = []
        write_list_to_json(contest_data, f'{curr_dir}/judge_data/contests/{judge_name}_contest_data.json')
    else:
        contest_data = read_json_to_list(f'{curr_dir}/judge_data/contests/{judge_name}_contest_data.json')

    #initializes the judge model
    print(f"Initializing LLM Judge '{judge_name}'.\n")
    model_name = "MaziyarPanahi/Llama-3-70B-Instruct-32k-v0.1" 
    llm = LLM(model=model_name, max_seq_len_to_capture=32000, tensor_parallel_size=NGPUS)
    sampling_params = SamplingParams(temperature=0, max_tokens=None, stop=["</reasoning>", "</answer>"], include_stop_str_in_output=False)

    print(f"Initiating Pairwise Comparison for '{judge_name}.'\n")
    #mod_perms = list([(a, b, idx) for a, b in permutations(models_list, 2) for idx in test_questions_dict.keys()])
    #shuffle(mod_perms)

    human_contests = read_json_to_list("human_data/human_contest_data.json")
    i = 0

    for contest in human_contests:
        i += 1
        a = contest["first"]
        b = contest["second"]
        idx = contest["question"]
        #a, b, idx = mod_perms[i]
        if {"first":a, "second":b, "question":idx, "winner":a} in contest_data or {"first":a, "second":b, "question":idx, "winner":b} in contest_data:
            pass

        else:
            eval_prompt = load_markdown(f'{curr_dir}/preference_prompt_judge.md')
            question = test_questions_dict[idx]
            documentation = "\n".join([to_xml(chunk) for chunk in test_chunks_dict[idx]])
            answer1 = models_dict[a].get(idx, "No answer.")
            answer2 = models_dict[b].get(idx, "No answer.")
            eval_prompt = eval_prompt.format(QUESTION=question, DOCUMENTATION=documentation, ANSWER1=answer1, ANSWER2=answer2)
           
            try:
                message = [{'role': 'user', 'content':eval_prompt}]
                tokenizer = AutoTokenizer.from_pretrained("MaziyarPanahi/Llama-3-70B-Instruct-32k-v0.1")
                reasoning_prompt = str(tokenizer.apply_chat_template(message, tokenize=False) + "\n<reasoning>")
                outputs = llm.generate([reasoning_prompt], sampling_params)
                
                for output in outputs:
                    reasoning = output.outputs[0].text
                #print("\nREASONING:", reasoning)

                answer_prompt = reasoning_prompt + reasoning + "\nBased on this reasoning, the better answer is: <answer>"
                outputs = llm.generate([answer_prompt], sampling_params)

                for output in outputs:
                    preference = output.outputs[0].text
                #print("PREFERENCE:", preference)
            
            except Exception as e:
                print(f"Error during llm.generate(): {e}")
                print("Skipping Process.\n")
                return

            data = {"first":a, "second":b, "question":idx, "winner": a if "1" in preference and "2" not in preference else (b if "2" in preference and "1" not in preference else None)}
            print(f"{data}\t{i+1} of {len(human_contests)} pairs compared for '{judge_name}'.")
            contest_data.append(data)
            write_list_to_json(contest_data, f'{curr_dir}/judge_data/contests/{judge_name}_contest_data.json')
    
    elo_scores = calculate_elo(models_list, contest_data)
    write_list_to_json(elo_scores, f'{curr_dir}/judge_data/elo_scores/{judge_name}_elo_scores.json')
    print(f"Completed Pairwise Comparison for '{judge_name}'.\n")


def read_json_to_dict(filename):
    with open(filename, 'r') as file:
        try:
            content = file.read()
            if not content.strip():  # If the file is empty, load an empty dictionary
                return {}
            else:  # Otherwise, load the JSON into a dictionary
                content = json.loads(content)
                return {int(key): value for key, value in content.items()}
        except json.JSONDecodeError:
            # Handle the case where the file is not in a valid JSON format
            raise ValueError("File contains invalid JSON")


def read_json_to_list(filename):
    def convert_types(item):
        try:
            item["first"] = str(item["first"])
            item["second"] = str(item["second"])
            item["question"] = int(item["question"])
            item["winner"] = str(item["winner"])
        except (ValueError, KeyError):
            raise ValueError("Invalid dictionary format or type")
        return item
    
    with open(filename, 'r') as file:
        try:
            content = file.read()
            if not content.strip():  # If the file is empty, load an empty list
                return []
            else:  # Otherwise, load the JSON into a list of dictionaries
                content = json.loads(content)
                if isinstance(content, list) and all(isinstance(item, dict) for item in content):
                    # Convert types and validate the structure
                    return [convert_types(item) for item in content]
                else:
                    raise ValueError("JSON is not a list of dictionaries")
        except json.JSONDecodeError:
            # Handle the case where the file is not in a valid JSON format
            raise ValueError("File contains invalid JSON")


def write_dict_to_json(dictionary, filename):
    with open(filename, 'w') as file:
        json.dump(dictionary, file, indent=4)


def write_list_to_json(lst, filename):
    with open(filename, 'w') as file:
        json.dump(lst, file, indent=4)


def load_markdown(file_path):
    with open(file_path, 'r') as file:
        return file.read()


def to_xml(dct):
        """turns a dictionary representation of a chunk into an XML representation suitable for usage in a prompt"""
        return ('<resource>\n'
                f'<url>{dct["url"]}</url>\n'
                '<text>\n'
                f'{dct["content"]}\n'
                '</text>\n'
                '</resource>')


if __name__ == "__main__":
    asyncio.run(main())
