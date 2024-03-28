"""
Convert textgrid files to hf dataset then upload to hub
"""
from tqdm import tqdm
import json
from datasets import Dataset
import os
import textgrid
from datasets import Audio


def textgrid_to_json(textgrid_path, tiers=None):
    """
    Convert a TextGrid file to a JSON format, with an option to specify which tiers to loop over.

    Args:
    textgrid_path (str): The path to the TextGrid file.
    tiers (list of str, optional): The list of tier names to extract text from. If None, all tiers are used.

    Returns:
    dict: A dictionary with 'path' and 'text' extracted from the specified tiers of the TextGrid file.
    """
    try:
        # Load the TextGrid file
        tg = textgrid.TextGrid.fromFile(textgrid_path)

        # Extract text from the specified TextGrid tiers
        extracted_texts = []
        for tier in tg.tiers:
            if tiers is None or tier.name in tiers:
                for interval in tier:
                    if interval.mark.strip() != "":
                        extracted_texts.append(
                            {
                                "text": interval.mark.strip(),
                                "start": int(interval.minTime * 1000),
                                "end": int(interval.maxTime * 1000),
                            }
                        )

        # Return the path and the combined text as a dictionary
        return extracted_texts
    except Exception as e:
        print(f"Error converting TextGrid to JSON for {textgrid_path}: {e}")
        return None


def convert_textgrid_to_dict(textgrid_folder):
    """
    Convert all textgrid files in the specified folder to a list of dictionaries.
    Each dictionary contains the path and the extracted text from the textgrid file.
    """
    # Placeholder for the list of dictionaries
    textgrid_dicts = []

    # Loop through each file in the textgrid folder
    for filename in tqdm(os.listdir(textgrid_folder)):
        if filename.endswith(".TextGrid"):
            # Construct the full path to the file
            filepath = os.path.join(textgrid_folder, filename)

            # Convert the textgrid file to a dictionary
            extracted_text = textgrid_to_json(filepath, tiers=["words"])

            # Append the dictionary to the list
            textgrid_dicts.append({"path": filepath, "text": extracted_text})

    return textgrid_dicts


def load_additional_data(json_file):
    """
    Load additional data from a JSON file.
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def create_hf_dataset(textgrid_folder, json_file):
    """
    Create a Hugging Face dataset from textgrid files and additional JSON data.
    """
    # Convert textgrid files to list of dicts
    textgrid_data = convert_textgrid_to_dict(textgrid_folder)

    # Load additional data from JSON
    additional_data = load_additional_data(json_file)

    # Filter out rows where 'path' does not contain 'test-clean'
    additional_data = [item for item in additional_data if "test-clean" in item["path"]]

    for additional_item in tqdm(additional_data):
        # Extract the file stem from the path
        file_stem = os.path.splitext(additional_item["path"])[0].split("/")[-1]

        # Find the corresponding words dict in textgrid_data
        corresponding_dict = next(
            (
                item
                for item in textgrid_data
                if os.path.splitext(item["path"])[0].split("/")[-1] == file_stem
            ),
            None,
        )

        # If a corresponding dict is found, merge the 'text' from additional_item into it
        additional_item["words"] = corresponding_dict["text"]

        # Upper case the words
        for word in additional_item["words"]:
            word["text"] = word["text"].upper()

        additional_item["path"] = (
            "/Users/michael/Desktop/librispeech_test_clean_other_raw/"
            + additional_item["path"]
        )

    data_dict = {
        key: [dic[key] for dic in additional_data] for key in additional_data[0]
    }
    dataset = Dataset.from_dict(data_dict)
    dataset = dataset.add_column("audio", dataset["path"])

    # Cast the 'path' column to an audio dataset
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

    dataset = dataset.remove_columns("path")

    return dataset


# Define the paths
textgrid_folder = (
    "/Users/michael/Desktop/wip/ai/projects/librispeech_data_processing/output_raw"
)
json_file = "/Users/michael/Desktop/wip/ai/projects/librispeech_data_processing/data/librispeech_test_clean_other.json"

# Create the dataset
dataset = create_hf_dataset(textgrid_folder, json_file)

# Upload to HF
dataset.push_to_hub("olympusmons/librispeech_asr_test_clean_word_timestamp")
