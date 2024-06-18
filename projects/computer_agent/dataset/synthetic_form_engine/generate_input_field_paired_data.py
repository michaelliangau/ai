from datasets import load_dataset

ds = load_dataset("rajpurkar/squad_v2")
ds = ds['train']

ds = ds.remove_columns([col for col in ds.column_names if col not in ["question", "answers"]])

ds.save_to_disk("../data/squad_v2_qa")
