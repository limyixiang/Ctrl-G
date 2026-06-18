import json
import datasets
import random
data_path = "open-r1/DAPO-Math-17k-Processed"
dataset = datasets.load_dataset(data_path, "all",split="train")

sample_num = 2000
output_path = "dapo_prompts_boxed_shuffled.json"

all_prompts = []
for d in dataset:
    question = d["prompt"]
    prompt = f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n"
    all_prompts.append(prompt)

random.shuffle(all_prompts)

with open(output_path, "w") as f:
    json.dump(all_prompts[:sample_num], f, indent=4)