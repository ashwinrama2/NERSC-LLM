from find_judge_models import get_judge_model_names
from find_chatbot_models import get_chatbot_model_names
import math
import json


def calculate_elo(models, preferences, k=32):
    # Initialize the ELO ratings and variances
    mean = 1000
    elo_ratings = {model: mean for model in models}
    match_counts = {model: 0 for model in models}
    squared_diffs = {model: 0 for model in models}

    # Function to calculate expected score
    def expected_score(rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    # Update the ELO ratings based on preferences
    for match in preferences:
        if match['winner'] in ['None', None]:
            pass
        else:
            winner = match['winner']
            loser = match['first'] if match['second'] == winner else match['second']

            winner_elo = elo_ratings[winner]
            loser_elo = elo_ratings[loser]

            expected_winner = expected_score(winner_elo, loser_elo)
            expected_loser = expected_score(loser_elo, winner_elo)

            # Update ratings
            elo_ratings[winner] = winner_elo + k * (1 - expected_winner)
            elo_ratings[loser] = loser_elo + k * (0 - expected_loser)

            # Update match counts
            match_counts[winner] += 1
            match_counts[loser] += 1

            # Update squared differences
            squared_diffs[winner] += (1 - expected_winner) ** 2
            squared_diffs[loser] += (0 - expected_loser) ** 2

    # Calculate standard deviation for each player
    std_devs = {}
    for model in models:
        if match_counts[model] > 0:
            variance = (k ** 2 / match_counts[model]) * squared_diffs[model]
            std_dev = math.sqrt(variance)
        else:
            std_dev = 0  # If no matches were played, standard deviation is 0
        std_devs[model] = round(std_dev, 2)

    sorted_elo_ratings = {k: (round(v), std_devs[k]) for k, v in sorted(elo_ratings.items(), key=lambda item: item[1], reverse=True)}
    return sorted_elo_ratings


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


def main():
    judges_list = get_judge_model_names()
    models_list = get_chatbot_model_names()
    for judge_name in judges_list:
        contest_data = read_json_to_list(f'judge_data/contests/{judge_name}_contest_data.json')
        elo_scores = calculate_elo(models_list, contest_data)
        write_dict_to_json(elo_scores, f'judge_data/elo_scores/{judge_name}_elo_scores.json')


if __name__ == "__main__":
    main()
