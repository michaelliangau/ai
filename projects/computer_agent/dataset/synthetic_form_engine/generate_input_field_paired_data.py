from datasets import load_dataset

ds = load_dataset("rajpurkar/squad_v2")
ds = ds['train']

ds = ds.remove_columns([col for col in ds.column_names if col not in ["question", "answers"]])

# Filter out questions that do not have answers
ds = ds.filter(lambda example: len(example["answers"]["text"]) > 0)

ds.save_to_disk("../data/squad_v2_qa")
