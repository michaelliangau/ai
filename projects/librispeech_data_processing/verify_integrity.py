"""
Check no not-found-audio in gentle outputs
"""

import os
import json

def verify_integrity():
    output_dir = "output_raw/"
    json_files = [pos_json for pos_json in os.listdir(output_dir) if pos_json.endswith('.json')]
    
    for json_file in json_files:
        with open(os.path.join(output_dir, json_file), 'r') as file:
            data = json.load(file)
            if "words" in data:
                for word in data["words"]:
                    if word["case"] != "success":
                        print(f"File {json_file} contains words not successfully aligned.")
                        break
                else:
                    print(f"File {json_file} verified successfully.")

if __name__ == "__main__":
    verify_integrity()
