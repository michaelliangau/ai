"""
Use MFA to align.

1. docker image pull mmcauliffe/montreal-forced-aligner:latest
2. docker run --user root -it -v /Users/michael/Desktop/wip/:/data mmcauliffe/montreal-forced-aligner:latest
3. mfa model download acoustic english_us_arpa && mfa model download dictionary english_us_arpa
4. mfa align /data/ai/projects/librispeech_data_processing/tmp/mfa_input english_us_arpa english_us_arpa /data/ai/projects/librispeech_data_processing/output_raw
"""
import shutil
import json
from tqdm import tqdm
import os

# Load the data from the JSON file
data_path = "data/librispeech_test_clean_other.json"
with open(data_path, "r") as file:
    data = json.load(file)

# Iterate through each element in the data
for idx, item in enumerate(tqdm(data)):
    audio_path = item["path"]
    audio_path_full = (
        f"/Users/michael/Desktop/librispeech_test_clean_other_raw/{audio_path}"
    )
    file_stem = audio_path.split("/")[-1].split(".")[0]
    transcript_text = item["text"]

    mfa_input_folder = "tmp/mfa_input/"
    if not os.path.exists(mfa_input_folder):
        os.makedirs(mfa_input_folder)

    # Copy the audio file to the mfa_input folder
    shutil.copy(audio_path_full, os.path.join(mfa_input_folder, f"{file_stem}.flac"))

    # Write the label to a txt file in the mfa_input folder
    label_file_path = os.path.join(mfa_input_folder, f"{file_stem}.txt")
    with open(label_file_path, "w") as label_file:
        label_file.write(transcript_text)
