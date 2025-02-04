import json

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

model_names = ["Gemma", "Llama2", "Llama3", "Mistral", "OpenChat", "Qwen", "Snorkel", "Starling", "StarlingCode", "Vicuna", "Zephyr"]

for model_name in model_names:
    file = f"{model_name}_answers.json"
    dct = read_json_to_dict(file)
    missing = []
    for i in range(1, 51):
        if i not in dct:
            missing.append(i)
    print(f"{model_name} missing: {missing}\n")
