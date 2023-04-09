# Native imports
import os

# Third party imports
import openai
import pinecone
import langchain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import magic
import nltk
import IPython

# Download nltk punkt
nltk.download('punkt')

# Set Pinecone API Key
with open('/Users/michael/Desktop/wip/pinecone_credentials.txt', 'r') as f:
    PINECONE_API_KEY = f.readline().strip()
    PINECONE_API_ENV = f.readline().strip()
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

# Set OpenAI API Key
with open('/Users/michael/Desktop/wip/openai_credentials.txt', 'r') as f:
    OPENAI_API_KEY = f.readline().strip()
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

loader = DirectoryLoader('context_data', glob='**/*.txt')

documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

texts = text_splitter.split_documents(documents)

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

# Instantiate Pinecone service
index_name = "buffetbot"
pinecone_service = pinecone.Index(index_name=index_name)

# Get embeddings for a list of texts
texts = ["text1"]
for text in texts:
    embeddings = get_embedding(text)
    IPython.embed()
    pinecone_service.upsert(item_id=text, vectors=embeddings)

IPython.embed()