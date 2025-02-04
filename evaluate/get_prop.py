import openai # CBORG API Proxy Server is OpenAI-compatible through the openai module
import os
import sys;
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


def get_prefs(judge):
    #loads contest data into a list of dictionaries
    if overwrite or not os.path.isfile(f'judge_data/contests/{judge}_contest_data.json'):
        contest_data = []
        write_list_to_json(contest_data, f'judge_data/contests/{judge}_contest_data.json')
    else:
        contest_data = read_json_to_list(f'judge_data/contests/{judge}_contest_data.json')


    print(f"Initiating Pairwise Comparison for '{judge}'.\n")
    human_contests = read_json_to_list("human_data/human_contest_data.json")
    i = 0

    for contest in human_contests:
        i += 1
        a = contest["first"]
        b = contest["second"]
        idx = contest["question"]
        
        if {"first":a, "second":b, "question":idx, "winner":a} in contest_data or {"first":a, "second":b, "question":idx, "winner":b} in contest_data:
            pass

        else:
            eval_prompt = load_markdown(f'preference_prompt_judge.md')
            question = test_questions_dict[idx]
            documentation = "\n".join([to_xml(chunk) for chunk in test_chunks_dict[idx]])
            answer1 = models_dict[a].get(idx, "No answer.")
            answer2 = models_dict[b].get(idx, "No answer.")
            eval_prompt = str(eval_prompt.format(QUESTION=question, DOCUMENTATION=documentation, ANSWER1=answer1, ANSWER2=answer2))
            try:
                message = [{'role': 'user', 'content':eval_prompt}]
                reasoning = client.chat.completions.create(model=judges[judge], messages=message, temperature=0.0).choices[-1].message.content
                #print(reasoning, "\n")
                answer_prompt = reasoning + "\nBased on this reasoning, which answer does the author think is better? Answer #1 or Answer #2? Write only the integer of the preferred answer."
                message = [{'role': 'user', 'content':answer_prompt}]
                preference = client.chat.completions.create(model=judges[judge], messages=message, temperature=0.0).choices[-1].message.content
                #print(preference, "\n")

            except Exception as e:
                print(f"Error calling judge {judge}:\n{e}\n")
                print("Skipping Process.\n")
                return

            data = {"first":a, "second":b, "question":idx, "winner": a if "1" in preference and "2" not in preference else (b if "2" in preference and "1" not in preference else None)}
            print(f"{data}\t{i} of {len(human_contests)} pairs compared for '{judge}'.")
            contest_data.append(data)
            write_list_to_json(contest_data, f'judge_data/contests/{judge}_contest_data.json')
    
    elo_scores = calculate_elo(models_list, contest_data)
    write_list_to_json(elo_scores, f'judge_data/elo_scores/{judge}_elo_scores.json')
    print(f"Completed Pairwise Comparison for '{judge}'.\n")



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


client = openai.OpenAI(
api_key='sk-ZaEsqhNq8a1YVLAgCfIeIg', # Please do not store your API key in the code
base_url="https://api.cborg.lbl.gov" # https://api-local.cborg.lbl.gov
)

overwrite = 0

judges = {
    "Llama3.1_LBNL":"lbl/llama-3",          
    #"Cohere":"lbl/command-r-plus",
    #"GPT3_5":"openai/gpt-3.5-turbo",
    #"GPT4o":"openai/gpt-4o",
    #"Claude_Haiku":"anthropic/claude-haiku",
    #"Claude_Sonnet":"anthropic/claude-sonnet",
    #"Claude_Opus":"anthropic/claude-opus"
}

models_list = get_chatbot_model_names()
models_dict = {model_name:read_json_to_dict(f'test_answers/{model_name}_answers.json') for model_name in models_list}

#load the test questions into a dictionary
test_questions_dict = read_json_to_dict(f'test_questions.json')

#load the test chunks into a dictionary
test_chunks_dict = read_json_to_dict(f'test_chunks.json')

os.makedirs(f'judge_data/contests', exist_ok=True)
os.makedirs(f'judge_data/elo_scores', exist_ok=True)

for judge in judges:
    get_prefs(judge)
