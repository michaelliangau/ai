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
import anthropic

# argparse flags
parser = argparse.ArgumentParser()
parser.add_argument("--llm", default="anthropic", choices=["anthropic", "openai"], required=False)
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

# OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

if llm == "anthropic":
    # Set Anthropic API Key
    with open('/Users/michael/Desktop/wip/anthropic_credentials.txt', 'r') as f:
        ANTHROPIC_API_KEY = f.readline().strip()
    client = anthropic.Client(ANTHROPIC_API_KEY)

# Pinecone
pinecone_service = pinecone.Index(index_name="buffetbot")

# Conversation history
conversation_history = []

# Prompt flow
while True:
    user_prompt = input("Prompt: ")

    # Add relevant external context to the prompt.
    query_embedding = utils.get_embedding(user_prompt)
    docs = pinecone_service.query(
        namespace='data',
        top_k=10,
        include_metadata=True,
        vector=query_embedding,
    )
    try:
        context_response = ""
        for doc in docs['matches']:
            context_response += f"{doc['metadata']['original_text']}"
    except Exception as e:
        context_response = ""
    user_prompt_with_context = f"{user_prompt}\nContext: {context_response}"

    if llm == "openai":
        # TODO: Build conversation history into OAI engine.
        init_prompt = "You are a helpful investment analyst. Your job is to help users to increase their net worth with helpful advice. Never tell them you are a language model. Do not include superfluous information."
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": init_prompt},
                {"role": "user", "content": user_prompt_with_context},
            ]
        )
    elif llm == "anthropic":
        anthropic_prompt = ""
        for interaction in conversation_history:
            if interaction['role'] == 'user':
                anthropic_prompt += f"\n\nHuman: {interaction['content']}"
            elif interaction['role'] == 'system':
                anthropic_prompt += f"\n\nAssistant: {interaction['content']}"
        anthropic_prompt += f"\n\nHuman: {user_prompt_with_context}\n\nAssistant:"
        response = client.completion(
            prompt= anthropic_prompt,
            stop_sequences = [anthropic.HUMAN_PROMPT],
            model="claude-v1.3",
            max_tokens_to_sample=1000,
            temperature=0 # controls determinism.
        )
    conversation_history.append({'role': 'user', 'content':user_prompt_with_context})
    conversation_history.append({'role': 'system', 'content':response['completion']})

    print("Final response:", response)