#import sys;
#import os;
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')));
#import lmntfy.models.llm as module
#import inspect
from random import shuffle

#judge_model_names = list({name:obj for name, obj in inspect.getmembers(module) if inspect.isclass(obj) and issubclass(obj, module.LanguageModel) and obj is not module.LanguageModel and name not in ["Default", "Judge"]}.keys())

judge_model_names = ["Vicuna", "Mistral", "Zephyr", "OpenChat", "Snorkel", "Llama3", "Starling", "StarlingCode", "Gemma", "Qwen"]
shuffle(judge_model_names)


def get_judge_model_names():
    return judge_model_names


def main():
    for judge_model_name in judge_model_names:
        print(judge_model_name)


if __name__ == "__main__":
    main()
