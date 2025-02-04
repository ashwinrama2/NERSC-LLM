import json

chatbot_models = ["Vicuna", "Mistral", "Zephyr", "OpenChat", "Snorkel", "Llama3", "Starling", "StarlingCode", "Gemma", "Qwen"]
judge_models = ["Cohere", "Llama3_LBNL", "GPT3_5", "GPT4o", "Llama3_70b", "Claude_Haiku", "Claude_Sonnet", "Gemma2_27b", "Claude_Opus"]


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

for judge in judge_models:
    file_name = f"{judge}_elo_scores.json"
    judge_elo = read_json_to_dict(file_name)
    chatbot_list = "\",\"".join(judge_elo.keys())
    print(f"\"{judge}\":[\"{chatbot_list}\"]")
