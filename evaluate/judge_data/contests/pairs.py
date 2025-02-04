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


from itertools import permutations

def count_wins(data, a, b):
    a_wins = 0
    b_wins = 0

    for entry in data:
        if (entry['first'] == a and entry['second'] == b):
            if entry['winner'] == a:
                a_wins += 1
            elif entry['winner'] == b:
                b_wins += 1

    contest = str(a) + "_" + str(b)
    if a_wins + b_wins == 0:
      return {contest:0}
    else:
      ratio = a_wins/(a_wins + b_wins)
      return {contest:ratio}

contest_data = read_json_to_list("Llama3_70b_contest_data.json")

items = ["Mistral", "Zephyr", "Vicuna", "Gemma", "OpenChat", "Snorkel", "Starling", "Qwen", "StarlingCode", "Llama3"]
pairs = list(permutations(items, 2))

res = [count_wins(contest_data, a, b) for a, b in pairs]
print(res)
