import sys;
import os;
import os.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))); 
import lmntfy
import argparse
import asyncio
import json
from pathlib import Path
from lmntfy.question_answering import QuestionAnswerer
from typing import Dict, List
from lmntfy.database import Chunk
import inspect
import lmntfy.models.llm as module


def parse_args():
    # Read the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_folder", default="../models", type=Path, help="path to the folder containing all the models")
    parser.add_argument("--curr_dir", default=".", type=Path, help="path to the folder containing test questions")
    parser.add_argument("--model_name", default=None, type=str, help="the base model of the chatbot that generates the test answers")
    parser.add_argument("--overwrite", default=0, type=int, help="whether or not to re-compare and overwrite test answers")
    args = parser.parse_args()
    return args


async def main():
    # process command line arguments
    args = parse_args()
    models_folder = args.models_folder
    curr_dir = args.curr_dir
    model_name = args.model_name
    overwrite = args.overwrite

    # Create a dictionary of model's name mapped to the model object for all LLMs
    all_models_dict = {name:obj for name, obj in inspect.getmembers(module) if inspect.isclass(obj) and issubclass(obj, module.LanguageModel) and obj is not module.LanguageModel and name not in ["Default", "Judge"]}

    #check if the passed in model_name is a valid model
    if model_name not in all_models_dict:
        print("ERROR: Model '" + model_name + "' is not a recognized model. Skipping process.\n")
        return
    else:
        LLM_obj = all_models_dict[model_name]

    #load the test questions
    test_questions_dict = read_json_to_dict(f'{curr_dir}/test_questions.json')

    os.makedirs(f'{curr_dir}/test_answers', exist_ok=True)
    target_answer_file = f'{curr_dir}/test_answers/{model_name}_answers.json'
    
    #check if target_answer_file already exists:
    if overwrite or not os.path.isfile(target_answer_file): #file does not exist or overwrite
        #proceed from scratch
        test_answers_dict = {}
        unanswered_Qs = [key for key in test_questions_dict if key not in test_answers_dict]
        write_dict_to_json(test_answers_dict, target_answer_file) 
        print(f"Creating answer file for '{model_name}' and answering all questions now.")

    else: #file exists
        #load the test_answers into a dict
        test_answers_dict = read_json_to_dict(target_answer_file)
        
        #delete all answers from test_answers_dict if they are not in test_questions_dict
        test_answers_dict = {key: value for key, value in test_answers_dict.items() if key in test_questions_dict}
        
        #write updated test_answers_dict to target_answer_file
        write_dict_to_json(test_answers_dict, target_answer_file) 

        #find all questions IDs that are in test_questions_dict, but not in test_answers_dict. These need to be answered.
        unanswered_Qs = [key for key in test_questions_dict if key not in test_answers_dict]
        
        if len(unanswered_Qs) == 0: 
            print(f"Answer file for '{model_name}' exists and is fully updated. Skipping Process.\n")
            return

        else:
            print(f"Answer file for '{model_name}' exists, but is not up to date with test_questions.json. Answering missing questions now.")
    

    print(f"...Initializing LLM for '{model_name}'...")
    try:
        llm = LLM_obj(models_folder, device='cuda')
    except Exception as e:
        print(f"Error initializing LLM for '{model_name}'. Skipping process.")
        print(f"{e}\n")
        return
    question_answerer = lmntfy.QuestionAnswerer(llm, None)
    
    
    print(f"...Generating Test Answers for '{model_name}'...")
    await generate_answers(question_answerer, test_questions_dict, unanswered_Qs, test_answers_dict, target_answer_file, curr_dir, model_name)
    print(f"...Finished Generating Test Answers for '{model_name}'...\n")
    return


async def generate_answers(question_answerer:QuestionAnswerer, questions_dict:Dict[int, str], indices_needed:List[int], answers_dict:Dict[int, str], target_answer_file:str, curr_dir:str, model_name:str):
    #run on a handful of test question for quick evaluation purposes"""
    answers_dict = {**answers_dict}

    #ask questions concurently
    for idx in indices_needed:
        try:
            message = [{'role': 'user', 'content': questions_dict[idx]}]
            chunks = read_json_to_dict(f'{curr_dir}/test_chunks.json')[idx]
            chunks = [Chunk.from_dict(chunk) for chunk in chunks]
            answers_dict[idx] = await question_answerer._answer_messages(message, chunks)
            write_dict_to_json(dict(sorted(answers_dict.items())), target_answer_file) #save answers dictionary to json     
        except Exception as e:
            print(f"Error asking Question {idx} to LLM '{model_name}'. Skipping question.")
            print(f"{e}\n")
    return


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


def write_dict_to_json(dictionary, filename):
    with open(filename, 'w') as file:
        json.dump(dictionary, file, indent=4)


if __name__ == "__main__":
    asyncio.run(main())
