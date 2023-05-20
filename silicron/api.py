# Native imports
import json
import uuid
from typing import Union, List
import logging

# Third party imports
import IPython
import openai
import pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from tqdm import tqdm

# Our imports
import silicron.common.utils as utils

# Display logging messages in the terminal
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)


class Silicron:
    def __init__(self, api_key: str):
        """Initialize the Silicron class.

        Args:
            api_key (str): The API key to use for authentication.
        """
        self.api_key = api_key  # TODO not implemented.

        # Set credentials (OpenAI and Pinecone)
        utils.set_credentials()

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
        config = utils.set_config(config)

        # Extract variables from config
        chatbot = config["chatbot"]
        database = config["database"]

        # Inject context into prompt
        context = utils.get_context(prompt, database)
        prompt_context = f"{prompt}\nAdditional context for you: {context}"

        prompt_context = utils.trim_input(prompt_context)

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
        formatted_response = utils.extract_response_content(response)

        return formatted_response

    def upload_data(
        self, data_file_paths: Union[str, List[str]], index_name: str
    ) -> None:
        """
        Segment the text from the provided file or list of values,
        create vectors in OpenAI, and insert them into the Pinecone database.

        Args:
            data_file_paths (Union[str, List[str]]): The path to the data file or a list of values to process.
            index_name (str): The name of the Pinecone index to insert the vectors into.
        """
        # If the data_file_paths is a string, then it is a path to a file.
        if isinstance(data_file_paths, str):
            data_file_paths = [data_file_paths]

        for file_path in tqdm(data_file_paths):
            # open the text file in a single str object
            with open(file_path, "r") as f:
                text = f.read()

            # Initialize Pinecone service
            pinecone_service = pinecone.Index(index_name=index_name)

            # Iterate through the chunks and add them to Pinecone
            try:
                # Create the embeddings using OpenAI
                embeddings = utils.get_embedding(text)

                # Create the vector to be inserted into Pinecone
                vector = {
                    "id": str(uuid.uuid4()),
                    "values": embeddings,
                    "metadata": {
                        "original_text": text,
                    },
                }

                # Insert the vector into Pinecone
                _ = pinecone_service.upsert(
                    vectors=[vector],
                    namespace="data",
                )
                logging.info(
                    f"Successfully inserted vector into vector db: {file_path}"
                )
            except Exception as e:
                print(e)
