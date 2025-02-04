import json

models = ["Vicuna", "Mistral", "Zephyr", "OpenChat", "Snorkel", "Llama3", "Starling", "StarlingCode", "Gemma", "Qwen"]

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


for model in models:
    file_name = f"{model}_contest_data.json"
    model_prefs = read_json_to_list(file_name)
    print(model, len(model_prefs))
