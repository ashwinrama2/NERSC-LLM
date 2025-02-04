import sys;
import os; 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))); 
import lmntfy;
import argparse
import asyncio
from pathlib import Path
import json
import os
from random import sample
import lmntfy.models.llm as module
from metrics import calculate_elo
from find_chatbot_models import get_chatbot_model_names


def parse_args():
    # Read the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--curr_dir", default=".", type=Path, help="path to the folder containing test questions")
    parser.add_argument("--overwrite", default=0, type=int, help="whether or not to overwrite the current human preferences")
    args = parser.parse_args()
    return args


async def main():
    # process command line arguments
    args = parse_args()
    curr_dir = args.curr_dir
    overwrite = args.overwrite

    models_list = get_chatbot_model_names()
    models_dict = {model_name:read_json_to_dict(f'{curr_dir}/test_answers/{model_name}_answers.json') for model_name in models_list} #store the answer dictionary with its corresponding model's name

    #load the test questions into a dictionary
    test_questions_dict = read_json_to_dict(f'{curr_dir}/test_questions.json')
    
    #loads contest data into a list of dictionaries
    contest_data_file = f'{curr_dir}/human_data/human_contest_data.json'
    progress_file = f'{curr_dir}/human_data/progress.txt'

    os.makedirs(f'{curr_dir}/human_data', exist_ok=True)

    if os.path.isfile(progress_file):
        start = int(open(progress_file, 'r').readline().strip())
    else:
        start = 1
        with open(progress_file, 'w') as file:
            file.write(str(start))

    if overwrite or not os.path.isfile(contest_data_file):
        contest_data = []
        write_list_to_json(contest_data, contest_data_file)
    else:
        contest_data = read_json_to_list(contest_data_file)

    print("Initiating Ranked Comparison\n")
    for idx in range(start, len(test_questions_dict)+1):
        m1, m2, m3, m4, m5, m6, m7, m8 = sample(models_list, 8)
        mod = [(m1, m2, m3, m4), (m5, m6, m7, m8)]

        for i in range(2):
            a, b, c, d = mod[i]
            question = remove_after_last_period(test_questions_dict[idx])
            answer1 = remove_after_last_period(models_dict[a].get(idx, "No answer."))
            answer2 = remove_after_last_period(models_dict[b].get(idx, "No answer."))
            answer3 = remove_after_last_period(models_dict[c].get(idx, "No answer."))
            answer4 = remove_after_last_period(models_dict[d].get(idx, "No answer."))

            eval_prompt = load_markdown(f'{curr_dir}/preference_prompt_human.md')
            eval_prompt = eval_prompt.format(NUM=idx, QUESTION=question, ANSWER1=answer1, ANSWER2=answer2, ANSWER3=answer3, ANSWER4=answer4)
            os.system("clear")
            print(eval_prompt)

            prefs = input("Preference: ")
            decision = return_pairwise_contests(prefs, [a, b, c, d], idx)
            contest_data += decision
            write_list_to_json(contest_data, contest_data_file)
            with open(progress_file, 'w') as file:
                file.write(str(idx))

    elo_scores = calculate_elo(models_list, contest_data)
    write_list_to_json(elo_scores, f'{curr_dir}/human_data/human_elo_scores.json')


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


def return_pairwise_contests(ranks_str, models_list, idx):
    ranks = list(map(int, ranks_str.split(',')))
    contests = []

    for i in range(len(models_list)):
        for j in range(len(models_list)):
            if i != j:
                contest = {
                    "first": models_list[i],
                    "second": models_list[j],
                    "question":idx,
                    "winner": models_list[i] if ranks[i] < ranks[j] else models_list[j]
                }
                contests.append(contest)

    return contests


def remove_after_last_period(text):
    reference_index = text.rfind("References:")
    if reference_index == -1:
        return text
    last_period_index = text[:reference_index].rfind(".")
    if last_period_index == -1:
        return text
    return text[:last_period_index + 1]


if __name__ == "__main__":
    asyncio.run(main())
