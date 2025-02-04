import argparse
import asyncio
from pathlib import Path
import json
from find_judge_models import get_judge_model_names
import os
import math


def parse_args():
    # Read the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--curr_dir", default=".", type=Path, help="path to the folder containing test questions")
    args = parser.parse_args()
    return args


def main():
    # process command line arguments
    args = parse_args()
    curr_dir = args.curr_dir


    judge_list = get_judge_model_names()
    judge_prefs_dict = {
    judge_name: read_json_to_list(f'{curr_dir}/judge_data/contests/{judge_name}_contest_data.json')
    for judge_name in judge_list
    if os.path.exists(f'{curr_dir}/judge_data/contests/{judge_name}_contest_data.json')
}
    judge_alignment = {judge_name:[] for judge_name in judge_prefs_dict}

    human_prefs = read_json_to_list(f'{curr_dir}/human_data/human_contest_data.json')

    for judge1 in judge_prefs_dict:
        judge1_prefs = judge_prefs_dict[judge1]

        for judge2 in judge_prefs_dict:
            judge2_prefs = judge_prefs_dict[judge2]
            first = judge2_prefs["first"]
            second = judge2_prefs["second"]
            question = judge2_prefs["question"]
            winner = judge2_prefs["winner"]

            first_wins = {"first":first, "second":second, "question":question, "winner":first}
            second_wins = {"first":first, "second":second, "question":question, "winner":second}
            human_pref = {"first":first, "second":second, "question":question, "winner":winner} 

            #assumes that judge_prefs is a superset of human_prefs. (i.e., judge_prefs has all prefs, human_prefs has sampled prefs).
            if (first_wins in judge1_prefs) or (second_wins in judge1_prefs): #check if the contest is in judge_prefs (may include winner).
                if first_wins in judge1_prefs and first_wins == human_pref: #check if the contest is also the winner
                    judge_alignment[judge1] += [1]
                elif second_wins in judge1_prefs and second_wins == human_pref: #check if the contest is also the winner
                    judge_alignment[judge1] += [1]
                else:  # if the contest is common, but not the winner
                    judge_alignment[judge1] += [0]
            else:
                pass

    for judge in judge_alignment:
        if len(judge_prefs_dict[judge]) > 0:
            alignment, stdev = calc_stats(judge_alignment[judge])
            discrepancy = round(count_discrepancies(judge_prefs_dict[judge]), 4)
            nones = round(count_nones(judge_prefs_dict[judge]), 4)
            print(f"{judge}:\tAlignment - {round(alignment, 4)} ± {round(stdev, 4)},\tDiscrepancy - {discrepancy},\tNones - {nones}, Matches - {len(judge_alignment[judge])}")


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
            if not content.strip() or content == []:  # If the file is empty, load an empty list
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


def count_discrepancies(preferences):
    discrepancies = 0
    seen = {}

    for pref in preferences:
        first = pref["first"]
        second = pref["second"]
        question = pref["question"]
        winner = pref["winner"]
        
        # Check if the swapped pair with the same question is already seen
        if (second, first, question) in seen:
            if seen[(second, first, question)] != winner:
                discrepancies += 1
        else:
            seen[(first, second, question)] = winner

    return discrepancies/len(preferences)


def calc_stats(sequence):
    mean = sum(sequence) / len(sequence)
    squared_diffs = [(x - mean) ** 2 for x in sequence]
    variance = sum(squared_diffs) / len(sequence)
    std = math.sqrt(variance)
    return mean, std


def count_nones(preferences):
    nones = 0

    for pref in preferences:
        if pref["winner"] in ["None", None]:
            nones += 1

    return nones/len(preferences)

if __name__ == "__main__":
    main()

