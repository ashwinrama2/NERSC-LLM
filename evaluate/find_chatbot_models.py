#import sys;
#import os;
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')));
#import lmntfy.models.llm as module
#import inspect
from random import shuffle

#chatbot_model_names = list({name:obj for name, obj in inspect.getmembers(module) if inspect.isclass(obj) and issubclass(obj, module.LanguageModel) and obj is not module.LanguageModel and name not in ["Default", "Judge"]}.keys())

chatbot_model_names = ["Vicuna", "Mistral", "Zephyr", "OpenChat", "Snorkel", "Llama3", "Starling", "StarlingCode", "Gemma", "Qwen"]
shuffle(chatbot_model_names)


def get_chatbot_model_names():
    return chatbot_model_names


def main():
    for chatbot_model_name in chatbot_model_names:
        print(chatbot_model_name)


if __name__ == "__main__":
    main()
