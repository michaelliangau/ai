# Native imports
import json
from typing import Union, List

# Third party imports
import IPython
import openai
import pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from tqdm import tqdm

class ContextGPT:
    def __init__(self, api_key: str):
        """Initialize the ContextGPT class.

        Args:
            api_key (str): The API key to use for authentication.
        """
        self.api_key = api_key  # TODO not implemented.

        # OpenAI
        with open("/Users/michael/Desktop/wip/openai_credentials.txt", "r") as f:
            OPENAI_API_KEY = f.readline().strip()
            openai.api_key = OPENAI_API_KEY
        
        # Pinecone
        with open("/Users/michael/Desktop/wip/pinecone_credentials.txt", "r") as f:
            PINECONE_API_KEY = f.readline().strip()
            PINECONE_API_ENV = f.readline().strip()
            pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

    def get_response(self, prompt: str, config: dict = None) -> str:
        """Get a response from the chatbot based on the given prompt and configuration.

        Args:
            prompt (str): The prompt to send to the chatbot.
            config (dict, optional): A dictionary containing the configuration for the chatbot and database.
                Defaults to None.

        Returns:
            str: The formatted response from the chatbot.
        """
        # Set default config
        if config is None or config["chatbot"] is None:
            config = {"chatbot": "chatgpt3.5-turbo", "database": None}

        # Extract variables from config
        chatbot = config["chatbot"]
        database = config["database"]

        # Inject context into prompt
        with open("./tests/data/test.txt", "r") as f:
            context = f.readlines()
        prompt_context = f"{prompt}\nAdditional context you may consider: {context}"

        # Get response
        init_prompt = "You are a helpful chatbot that helps users query their data in natural language. You are given a prompt and a context. You must return the response to the prompt based on the context."
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": init_prompt},
                {"role": "user", "content": prompt_context},
            ],
        )

        # Format response
        formatted_response = self._extract_response_content(response)

        return formatted_response

    def upload_data(self, data_file_path: Union[str, List[str]], index_name: str) -> None:
        """
        Segment the text from the provided file or list of values,
        create vectors in OpenAI, and insert them into the Pinecone database.

        Args:
            data_file_path (Union[str, List[str]]): The path to the data file or a list of values to process.
            index_name (str): The name of the Pinecone index to insert the vectors into.
        """
        # Convert single string object to a list
        if isinstance(data_file_path, str):
            data_file_path = [data_file_path]

        # Load the documents
        loader = TextLoader(data_file_path)
        documents = loader.load()

        # Segment the documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n")
        texts = text_splitter.split_documents(documents)

        # Initialize Pinecone service
        pinecone_service = pinecone.Index(index_name=index_name)

        # Iterate through the chunks and add them to Pinecone
        for idx, text in tqdm(enumerate(texts), total=len(texts)):
            try:
                # Create the embeddings using OpenAI
                embeddings = self._get_embedding(text.page_content)

                # Create the vector to be inserted into Pinecone
                vector = {
                    "id": str(idx),
                    "values": embeddings,
                    "metadata": {
                        "original_text": text.page_content,
                    },
                }

                # Insert the vector into Pinecone
                _ = pinecone_service.upsert(
                    vectors=[vector],
                    namespace="data",
                )
            except Exception as e:
                print(e)


    def _get_embedding(self, text, model="text-embedding-ada-002"):
        """
        Get embeddings for the given text using the specified OpenAI model.

        Args:
            text (str): The text to get embeddings for.
            model (str, optional): The name of the OpenAI model to use. Defaults to "text-embedding-ada-002".

        Returns:
            list: The embeddings for the given text.
        """
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]


    def _extract_response_content(self, response):
        return response["choices"][0]["message"]["content"]
