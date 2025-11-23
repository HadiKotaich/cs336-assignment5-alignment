from datasets import load_dataset

dataset = load_dataset("allenai/tulu-3-sft-personas-math")

print(dataset["train"].column_names)  # first 5 rows
