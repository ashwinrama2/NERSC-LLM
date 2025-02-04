import sys;
import os; 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))); 
import lmntfy;
import argparse
import asyncio
import json
from pathlib import Path
from lmntfy.question_answering import QuestionAnswerer
from typing import Dict, List
import os.path


def parse_args():
    # Read the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_folder", default="./data/nersc_doc/docs", type=Path, help="path to the NERSC documentation folder")
    parser.add_argument("--database_folder", default="./data/database", type=Path, help="path to the database saving folder") 
    parser.add_argument("--models_folder", default="../models", type=Path, help="path to the folder containing all the models")
    parser.add_argument("--curr_dir", default=".", type=Path, help="path to the folder containing test questions")
    parser.add_argument("--overwrite", default=0, type=int, help="whether or not to re-compare and overwrite test answers")
    args = parser.parse_args()
    return args

async def main():
    # process command line arguments
    args = parse_args()
    docs_folder = args.docs_folder
    database_folder = args.database_folder
    models_folder = args.models_folder
    curr_dir = args.curr_dir
    overwrite = args.overwrite

    #load the test questions
    test_questions_dict = read_json_to_dict(f'{curr_dir}/test_questions.json')

    target_chunks_file = f'{curr_dir}/test_chunks.json'
    
    #check if target_chunks_file already exists:
    if overwrite or not os.path.isfile(target_chunks_file): #file does not exist or overwrite
        #start from scratch
        test_chunks_dict = {}
        unchunked_Qs = [key for key in test_questions_dict if key not in test_chunks_dict]
        write_dict_to_json(test_chunks_dict, target_chunks_file) 
        print(f"\nCreating chunks file and generating chunks for all questions now.")

    else:
        #load the test_chunks into a dict
        test_chunks_dict = read_json_to_dict(target_chunks_file)
        
        #delete all chunks from test_chunks_dict if they are not in test_questions_dict
        test_chunks_dict = {key: value for key, value in test_chunks_dict.items() if key in test_questions_dict}
        
        #write updated test_chunks_dict to target_chunks_file
        write_dict_to_json(test_chunks_dict, target_chunks_file) 

        #find all questions IDs that are in test_questions_dict, but not in test_chunks_dict. These need to be chunked.
        unchunked_Qs = [key for key in test_questions_dict if key not in test_chunks_dict]
        
        if len(unchunked_Qs) == 0: 
            print(f"Chunks file exists and is fully updated. Skipping Process.\n")
            return

        else:
            print(f"Chunks file exists, but is not up to date with test_questions.json. Generating chunks for missing questions now.")
    

    print(f"Initializing Default LLM for Chunks\n")
    search_engine = lmntfy.database.search.Default(models_folder, device='cuda')
    llm = lmntfy.models.llm.Default(models_folder, device='cuda')
    database = lmntfy.database.Database(docs_folder, database_folder, search_engine, llm, update_database=False)
    question_answerer = lmntfy.QuestionAnswerer(llm, database)
    
    
    print(f"\n...Generating Chunks...")
    await generate_chunks(question_answerer, test_questions_dict, unchunked_Qs, test_chunks_dict, target_chunks_file)
    print(f"...Finished Generating Chunks...\n")
    return


async def generate_chunks(question_answerer:QuestionAnswerer, questions_dict:Dict[int, str], indices_needed:List[int], chunks_dict:Dict[int, str], target_chunks_file:str):
    #run on a handful of test question for quick evaluation purposes"""
    chunks_dict = {**chunks_dict}

    #ask questions concurently
    for idx in indices_needed:
        try:
            message = [{'role': 'user', 'content': questions_dict[idx]}]
            keywords = await question_answerer._extract_question(message)
            chunks = question_answerer.database.get_closest_chunks(keywords, 8)[0:8]
            chunks_dict[idx] = [chunk.to_dict() for chunk in chunks]
            write_dict_to_json(dict(sorted(chunks_dict.items())), target_chunks_file) #save chunks dictionary as json 
        
        except Exception as e:
            print(f"Error generating chunk for Question {idx}. Skipping question.\n")
    
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
