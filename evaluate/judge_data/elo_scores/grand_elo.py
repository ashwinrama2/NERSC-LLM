import json

chatbot_models = ["Vicuna", "Mistral", "Zephyr", "OpenChat", "Snorkel", "Llama3", "Starling", "StarlingCode", "Gemma", "Qwen"]
judge_models = ["Cohere", "Llama3_LBNL", "GPT3_5", "GPT4o", "Llama3_70b", "Claude_Haiku", "Claude_Sonnet", "Claude_Opus"]


def read_json_to_dict(filename):
    with open(filename, 'r') as file:
        try:
            content = file.read()
            if not content.strip():  # If the file is empty, load an empty dictionary
                return {}
            else:  # Otherwise, load the JSON into a dictionary
                content = json.loads(content)
                return {str(key):list(value) for key, value in content.items()}
        except json.JSONDecodeError:
            # Handle the case where the file is not in a valid JSON format
            raise ValueError("File contains invalid JSON")


elo_agg_score = {chatbot:0 for chatbot in chatbot_models}
elo_agg_stdev = {chatbot:0 for chatbot in chatbot_models}

for judge in judge_models:
    file_name = f"{judge}_elo_scores.json"
    judge_elo = read_json_to_dict(file_name)
    for chatbot in chatbot_models:
        score, stdev = judge_elo[chatbot]
        elo_agg_score[chatbot] += float(score)
        elo_agg_stdev[chatbot] += float(stdev)

final = {key: [round(elo_agg_score[key], 2), round(elo_agg_stdev[key], 2)] for key in elo_agg_score}

sorted_dict = dict(sorted(final.items(), key=lambda item: item[1][0], reverse=True))

print(sorted_dict)
