# Import necessary libraries
import os
import numpy as np
from sentence_transformers import SentenceTransformer

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

# Loop through each file in the company folder
for company_file in os.listdir(company_folder):
    # Open each file and read the company description
    with open(os.path.join(company_folder, company_file), "r") as f:
        company = [f.read()]

    # Generate embeddings for the company description
    company_embedding = model.encode(company, normalize_embeddings=True)

    # Calculate the similarity between the company and category embeddings
    similarity = company_embedding @ category_embeddings.T

    # Rank similarity (reversed)
    ranked_similarity = np.argsort(-similarity, axis=1)

    # Print out the company file name and the category file names with the ranked similarities and similarity scores
    print(f"\nCompany file: {company_file}")
    for i in range(ranked_similarity.shape[1]):
        print(
            f"Category file: {category_files[ranked_similarity[0, i]]}, Similarity rank: {i}, Similarity score: {similarity[0, ranked_similarity[0, i]]}"
        )
