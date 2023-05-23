# Native imports
import json
import uuid
from typing import Union, List
import logging

# Third party imports
import IPython
import openai
import pinecone
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

# CONSTANTS
S3_BUCKET = "silicron"
CUSTOMER_ID = 0
PATH_TO_CREDENTIALS = "/Users/michael/Desktop/wip"


class Silicron:
    def __init__(self, api_key: str):
        """Initialize the Silicron class.

        Args:
            api_key (str): The API key to use for authentication.
        """
        self.api_key = api_key  # TODO not implemented.

        self.customer_id = CUSTOMER_ID  # TODO implement dynamic customer id.

        # S3 init
        self.s3 = utils.initialise_s3_session(
            f"{PATH_TO_CREDENTIALS}/aws_credentials.txt"
        )

        # OpenAI init
        with open(f"{PATH_TO_CREDENTIALS}/openai_credentials.txt", "r") as f:
            OPENAI_API_KEY = f.readline().strip()
            openai.api_key = OPENAI_API_KEY

        # Pinecone init
        with open(f"{PATH_TO_CREDENTIALS}/pinecone_credentials.txt", "r") as f:
            PINECONE_API_KEY = f.readline().strip()
            PINECONE_API_ENV = f.readline().strip()
            pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

    def ask(self, prompt: str, config: dict = None) -> str:
        """Get a response from the chatbot based on the given prompt and configuration.

        Args:
            prompt (str): The prompt to send to the chatbot.
            config (dict, optional): A dictionary containing the configuration for the chatbot and database.
                Defaults to None.

        Returns:
            Dict: API response.
        """
        # Set default config
        config = utils.set_config(config)

        # Extract variables from config
        chatbot = config["chatbot"]
        database = config["database"]

        # Inject context into prompt
        context, context_list = utils.get_context(prompt, database)
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
        llm_response = utils.extract_response_content(response)
        out_response = {
            "response": llm_response,
            "context_referenced": context_list,
        }

        return out_response

    def upload(self, data_file_paths: Union[str, List[str]], index_name: str) -> None:
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

                # Save the file to S3
                utils.upload_to_s3(
                    self.s3,
                    file_path,
                    S3_BUCKET,
                    f"customer_data/{self.customer_id}/chat/uploaded_data/{file_path.split('/')[-1]}",
                )
                logging.info(
                    f"Successfully uploaded file to S3: s3://{S3_BUCKET}/customer_data/{self.customer_id}/chat/uploaded_data/{file_path.split('/')[-1]}"
                )

            except Exception as e:
                print(e)
