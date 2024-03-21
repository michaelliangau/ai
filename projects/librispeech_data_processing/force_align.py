"""
Use Gentle to force align

1. Start the Docker container `docker run -P lowerquality/gentle`
2. Run this file
"""

import subprocess
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
    audio_path = f"/Users/michael/Desktop/librispeech_test_clean_other_raw/{audio_path}"
    file_stem = audio_path.split("/")[-1].split(".")[0]
    transcript_text = item["text"]

    tmp_folder = "tmp/"
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
    transcript_file_path = os.path.join(tmp_folder, f"{idx}.txt")
    with open(transcript_file_path, 'w') as transcript_file:
        transcript_file.write(transcript_text)


    # Execute the curl command using subprocess
    curl_command = f'curl -F "audio=@{audio_path}" -F "transcript=@{tmp_folder}{idx}.txt" "http://localhost:55001/transcriptions?async=false"'
    response = subprocess.run(curl_command, shell=True, capture_output=True, text=True, check=True)
    output = json.loads(response.stdout)

    output_folder = "output_raw/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file_path = os.path.join(output_folder, f"{file_stem}.json")
    with open(output_file_path, 'w') as output_file:
        json.dump(output, output_file, indent=4)
