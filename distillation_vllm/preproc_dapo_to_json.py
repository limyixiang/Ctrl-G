import json
import datasets
import random
data_path = "open-r1/DAPO-Math-17k-Processed"
dataset = datasets.load_dataset(data_path, "all",split="train")

sample_num = 1000000
output_path = "dapo_prompts_shuffled_full.json"

all_prompts = []
for d in dataset:
    all_prompts.append(d["source_prompt"][0]['content'])

random.shuffle(all_prompts)

with open(output_path, "w") as f:
    json.dump(all_prompts[:sample_num], f, indent=4)