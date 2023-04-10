# Native imports
import os
import subprocess
import random
import math

# # Third party imports
import openai
import IPython
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

# Set OpenAI API Key
with open('/Users/michael/Desktop/wip/openai_credentials.txt', 'r') as f:
    OPENAI_API_KEY = f.readline().strip()
    openai.api_key = OPENAI_API_KEY

loader = TextLoader("context_data/test_article.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
docsearch = Chroma.from_documents(texts, embeddings)

qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=OPENAI_API_KEY), chain_type="stuff", retriever=docsearch.as_retriever())

query = "What's happening in New York?"
response = qa.run(query)
print(response)




# PINCONE

# import pinecone
# import langchain
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain import OpenAI, VectorDBQA
# from langchain.document_loaders import DirectoryLoader
# from langchain.document_loaders import UnstructuredFileLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import Chroma, Pinecone
# from langchain.embeddings.openai import OpenAIEmbeddings
# import magic
# import nltk

# # Download nltk punkt
# nltk.download('punkt')

# # Set Pinecone API Key
# with open('/Users/michael/Desktop/wip/pinecone_credentials.txt', 'r') as f:
#     PINECONE_API_KEY = f.readline().strip()
#     PINECONE_API_ENV = f.readline().strip()
#     pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

# loader = DirectoryLoader('context_data', glob='**/*.txt')

# documents = loader.load()

# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# texts = text_splitter.split_documents(documents)

# def get_embedding(text, model="text-embedding-ada-002"):
#    text = text.replace("\n", " ")
#    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

# # Instantiate Pinecone service
# index_name = "buffetbot"
# pinecone_service = pinecone.Index(index_name=index_name)

# # Upsert vectors into Pinecone
# # texts = ["text1"]
# # for text in texts:
# #     embeddings = get_embedding(text)
# #     vector = {
# #         'id': "1",
# #         'values': embeddings,
# #         'metadata': {'category': 'news'},
# #     }
# #     upsert_response = pinecone_service.upsert(
# #         vectors=[vector],
# #         namespace='data',
# #     )
# #     break

# from langchain.llms import OpenAI
# from langchain.chains.question_answering import load_qa_chain
# llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

# chain = load_qa_chain(llm, chain_type="stuff")

# query = "Tell me about Asynchronous Methods for Deep Reinforcement Learning"

# # Get embedding for query
# query_embedding = get_embedding(query)

# # Similarity search in pinecone
# docs = pinecone_service.query(
#     namespace='data',
#     top_k=1,
#     # include_values=True,
#     include_metadata=True,
#     vector=query_embedding,
# )

# IPython.embed()

# chain.run(input_documents=docs, question=query)
