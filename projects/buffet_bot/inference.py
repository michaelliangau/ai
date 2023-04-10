# Native imports
import os
import subprocess
import random
import math

# # Third party imports
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

# Set Pinecone API Key
with open('/Users/michael/Desktop/wip/pinecone_credentials.txt', 'r') as f:
    PINECONE_API_KEY = f.readline().strip()
    PINECONE_API_ENV = f.readline().strip()
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

# Set OpenAI API Key
with open('/Users/michael/Desktop/wip/openai_credentials.txt', 'r') as f:
    OPENAI_API_KEY = f.readline().strip()
    openai.api_key = OPENAI_API_KEY

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

loader = TextLoader("context_data/test_articles_one.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n")
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Chroma for debug
# docsearch = Chroma.from_documents(texts, embeddings)

pinecone_service = pinecone.Index(index_name="buffetbot")

llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")

query = "mets"
query_embedding = get_embedding(query)

docs = pinecone_service.query(
    namespace='data',
    top_k=3,
    # include_values=True,
    include_metadata=True,
    vector=query_embedding,
)
# Include top 3 results in prompt
context_response = ""
for doc in docs['matches']:
    context_response += f"{doc['metadata']['original_text']}"

# Final prompt
while True:
    user_prompt = input("Prompt: ")

    init_prompt = "You are a helpful financial assistant. Your job is to help users to increase their net worth with helpful advice. Telling them you are a language model is extremely unhelpful, do not do that."

    final_prompt = f"{user_prompt}\nThis is context on the wider world:\n{context_response}"

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": init_prompt},
            {"role": "user", "content": final_prompt},
        ]
    )
    print("Final response:", response)


IPython.embed()