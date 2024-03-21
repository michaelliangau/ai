import json
from tqdm import tqdm
import os

# Load the data from the JSON file
data_path = "data/librispeech_test_clean_other.json"
with open(data_path, 'r') as file:
    data = json.load(file)

# Iterate through each element in the data
for idx, item in enumerate(tqdm(data)):
    audio_path = item["path"]
    file_stem = audio_path.split("/")[-1].split(".")[0]
    output_file_path = f"output_raw/{file_stem}.TextGrid"
    if not os.path.exists(output_file_path):
        print(f"Missing output file for {file_stem}")
