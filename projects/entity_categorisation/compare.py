# Import necessary libraries
import json
from tqdm import tqdm
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd

# Initialize an empty list to store category descriptions
categories = []

# Initialize an empty list to store category file names
category_files = []

# Define the folder where category descriptions are stored
category_folder = "./category_data"

# Loop through each file in the category folder
for filename in os.listdir(category_folder):
    # Open each file and append its content (category description) to the categories list
    with open(os.path.join(category_folder, filename), "r") as f:
        categories.append(f.read())
        category_files.append(filename)

# Load the SentenceTransformer model for sentence embeddings
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Generate embeddings for the category descriptions
category_embeddings = model.encode(categories, normalize_embeddings=True)

# Define the folder where company descriptions are stored
company_folder = "./company_data"

# Compute targets
targets_df = pd.read_csv('/Users/michael/Desktop/test_set.csv')

# Iterating through each row in the DataFrame
targets_list = {}
for index, row in targets_df.iterrows():
    if pd.isna(row['Name']) or pd.isna(row['Description']) or pd.isna(row['VC Industry Classification']):
        continue
    target_file_name = row['VC Industry Classification'] + '.txt'
    target_file_name = target_file_name.replace(" / ", "_").replace(' ', '_').lower()

    # Get the index of this file_name in category_files
    file_index = category_files.index(target_file_name)

    # Actual file name
    file_name = row['Name'].replace(" / ", "_").replace(' ', '_').replace('.', '_').lower()
    file_name += '.txt'

    # Add this file_name and index to targets_list
    targets_list[file_name] = file_index

top_1_count = 0
top_3_count = 0
top_5_count = 0
total_files = 0

raw_outputs = []
for company_file in tqdm(os.listdir(company_folder)):
    # Open each file and read the company description
    with open(os.path.join(company_folder, company_file), "r") as f:
        company = [f.read()]

    # Generate embeddings for the company description
    company_embedding = model.encode(company, normalize_embeddings=True)

    # Calculate the similarity between the company and category embeddings
    similarity = company_embedding @ category_embeddings.T

    # Rank similarity (reversed)
    ranked_similarity = np.argsort(-similarity, axis=1)[0]

    # Get target from targets_list
    target = targets_list.get(company_file)
    if target is None:
        continue

    total_files += 1
    # Check if the target index is within the top 1, top 3 and top 5 predicted ranked_similarity values
    if target in ranked_similarity[:1]:
        top_1_count += 1
    if target in ranked_similarity[:3]:
        top_3_count += 1
    if target in ranked_similarity[:5]:
        top_5_count += 1

    # Save raw outputs
    raw_output = {
        "company_file": company_file,
        "ranked_similarity": [category_files[rs] for rs in ranked_similarity],
        "target": category_files[target]
    }
    raw_outputs.append(raw_output)

# Compute the percentage of files that are within the top 1, 3 and 5
top_1_percentage = (top_1_count / total_files) * 100
top_3_percentage = (top_3_count / total_files) * 100
top_5_percentage = (top_5_count / total_files) * 100

# Save raw outputs and accuracy to a json file
with open("./raw_outputs.json", "w") as f:
    json.dump({"accuracy": {"top_1": top_1_percentage, "top_3": top_3_percentage, "top_5": top_5_percentage}, "raw_outputs": raw_outputs}, f)

print(f"Top 1 accuracy: {top_1_percentage}%")
print(f"Top 3 accuracy: {top_3_percentage}%")
print(f"Top 5 accuracy: {top_5_percentage}%")
