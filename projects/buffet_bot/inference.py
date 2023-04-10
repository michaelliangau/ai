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

loader = TextLoader("context_data/test_articles_one.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n")
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
docsearch = Chroma.from_documents(texts, embeddings)

qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=OPENAI_API_KEY), chain_type="stuff", retriever=docsearch.as_retriever())


while True:
    test_prompt = input("Prompt: ")
    context_response = qa.run(test_prompt)
    print(context_response)



exit()
# Input loop
# TODO increase context size for output
context_prompt = "The user is a value investor looking for leads on value stocks to investigate for potential investments. Your task is to provide context and relevant information that will help another language model to recommend specific stocks. Please analyze the most recent news articles and financial data, and provide a summary of the following topics:\n1. Industry trends and recent news events that may impact the valuation of companies in those sectors, presenting opportunities for value investors.\n2. Any recent macroeconomic developments, regulatory changes, or market shifts that could create opportunities or risks for value investors.\n3. Notable management changes, strategic decisions, or company announcements that could influence the long-term prospects of potential value stocks.\nYour goal is not to select specific stocks but to provide the essential context that another language model can use to recommend undervalued investment opportunities for the user."

context_response = qa.run(context_prompt)
print("Context response:", context_response)

# Final prompt
# Input user_prompt
user_prompt = input("Prompt: ")

init_prompt = "You are a helpful financial assistant. Include a brief explanation for each stock recommendation, including the reasons why they are considered quality investments. Only include information about the stocks in bullet points. Include information about its price/earnings ratio if it exists and list your confidence on this stock pick."

final_prompt = f"{user_prompt}\nContext on the macroeconomic environment:\n{context_response}"

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": init_prompt},
        {"role": "user", "content": final_prompt},
    ]
)
print("Final response:", response)
IPython.embed()



########
# Prompt: I am a value investor, can you give me a few leads on value stocks that I can look deeper into to invest myself? You are a helpful financial planning assistant, your job is help users to increase their net worth with helpful advice.
# Response: Certainly! Here are a few value stocks that you might want to consider researching further:\n\n1. Berkshire Hathaway (BRK-A): Warren Buffet's legendary investment conglomerate that has a diverse portfolio of companies in areas such as insurance, energy, and manufacturing.\n\n2. JPMorgan Chase & Co. (JPM): One of the largest financial institutions in the world, JPMorgan has a strong balance sheet and generates significant free cash flow, making it an attractive value investment.\n\n3. Johnson & Johnson (JNJ): J&J is a well-established company with a diverse portfolio of consumer health products and pharmaceuticals. It has a strong history of steady growth and dividend payments.\n\n4. Walgreens Boots Alliance, Inc. (WBA): Walgreens is a major player in the retail pharmacy industry with a strong store network and distribution platform. The company is currently undergoing a transformation to enhance its digital capabilities.\n\nPlease note that any investment decision should be made after conducting thorough due diligence, analyzing the company\u2019s financials and business fundamentals, and considering your own investment objectives and risk tolerance.
#########


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
