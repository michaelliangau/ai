# Our imports
import utils

# Native imports
import os
import subprocess
import random
import math
import argparse

# Third party imports
import pinecone
import openai
import IPython
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain

# argparse flags
parser = argparse.ArgumentParser()
parser.add_argument("--llm", default="openai", required=True)
args = parser.parse_args()

llm = args.llm


# Set Pinecone API Key
with open('/Users/michael/Desktop/wip/pinecone_credentials.txt', 'r') as f:
    PINECONE_API_KEY = f.readline().strip()
    PINECONE_API_ENV = f.readline().strip()
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

# Set OpenAI API Key
with open('/Users/michael/Desktop/wip/openai_credentials.txt', 'r') as f:
    OPENAI_API_KEY = f.readline().strip()
    openai.api_key = OPENAI_API_KEY

# Load data
loader = TextLoader("context_data/test_articles_one.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n")
texts = text_splitter.split_documents(documents)

# OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Chroma for debug
# docsearch = Chroma.from_documents(texts, embeddings)

# Pinecone
pinecone_service = pinecone.Index(index_name="buffetbot")

# Prompt flow
while True:
    user_prompt = input("Prompt: ")

    init_prompt = "You are a helpful investment analyst. Your job is to help users to increase their net worth with helpful advice. Never tell them you are a language model. Explain your reasoning if you ever mention company names. Don't include any superfluos text."

    query_embedding = utils.get_embedding(user_prompt)

    docs = pinecone_service.query(
        namespace='data',
        top_k=10,
        # include_values=True,
        include_metadata=True,
        vector=query_embedding,
    )

    # Include results in prompt
    try:
        context_response = ""
        for doc in docs['matches']:
            context_response += f"{doc['metadata']['original_text']}\n"
    except Exception as e:
        context_response = ""
    
    final_prompt = f"{user_prompt}\Context:\n{context_response}"

    print("Final prompt:", final_prompt)
    print("\n")
    if llm == "openai":
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": init_prompt},
                {"role": "user", "content": final_prompt},
            ]
        )
    else:
        pass
    print("Final response:", response)