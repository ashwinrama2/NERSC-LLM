import json


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


def count_wins(data):
    contestants = {"None":0, "Mistral":0, "Zephyr":0, "Vicuna":0, "Gemma":0, "OpenChat":0, "Snorkel":0, "Starling":0, "Qwen":0, "StarlingCode":0, "Llama3":0}
    winners = {"None":0, "Mistral":0, "Zephyr":0, "Vicuna":0, "Gemma":0, "OpenChat":0, "Snorkel":0, "Starling":0, "Qwen":0, "StarlingCode":0, "Llama3":0}

    for entry in data:
        first = entry["first"]
        second = entry["second"]
        winner = entry['winner']

        contestants[first] += 1
        contestants[second] += 1
        winners[winner] += 1

    for key in winners:
        if contestants[key] == 0:
            winners[key] = 0
        else:
            winners[key] = winners[key]/contestants[key]
    
    return winners


judge_list = ["GPT4o", "GPT3_5", "Cohere", "Claude_Haiku", "Claude_Sonnet", "Claude_Opus", "Llama3_LBNL", "Llama3_70b"]
for judge in judge_list:
    contest_data = read_json_to_list(f"{judge}_contest_data.json")
    print("{" + str(judge) + ":" + str(count_wins(contest_data)) + "},")
