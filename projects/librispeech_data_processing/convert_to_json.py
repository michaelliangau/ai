import os
import json
import glob


def process_librispeech_data(raw_data_path, output_json_path):
    data = []
    txt_files = glob.glob(os.path.join(raw_data_path, "**/*.txt"), recursive=True)
    for txt_file_path in txt_files:
        with open(txt_file_path, "r") as txt_file:
            lines = txt_file.readlines()
            for line in lines:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    audio_id, transcript = parts
                    audio_path = os.path.join(
                        os.path.dirname(txt_file_path), audio_id + ".flac"
                    ).replace(
                        "/Users/michael/Desktop/librispeech_test_clean_other_raw/", ""
                    )
                    data.append({"path": audio_path, "text": transcript})

    with open(output_json_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


# Assuming the raw data path and desired output JSON path
raw_data_path = "/Users/michael/Desktop/librispeech_test_clean_other_raw"
output_json_path = "/Users/michael/Desktop/librispeech_test_clean_other.json"

# Process the data
process_librispeech_data(raw_data_path, output_json_path)
