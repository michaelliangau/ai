# Native imports
import json
import uuid
from typing import Union, List, Generator, Dict
import logging

# Third party imports
import openai
from tqdm import tqdm
import supabase

# Our imports
import backend.utils as utils

# Display logging messages in the terminal
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

# CONSTANTS
S3_BUCKET = "silicron"
PATH_TO_CREDENTIALS = "/Users/michael/Desktop/wip"


class Silicron:
    def __init__(self, user_id: str):
        """Initialize the Silicron class.

        Args:
            user_id (str): The user ID to use for authentication.
        """
        # Set user ID
        self.user_id = user_id

        # S3 init
        self.s3 = utils.initialise_s3_session(
            f"{PATH_TO_CREDENTIALS}/aws_credentials.txt"
        )

        # OpenAI init
        with open(f"{PATH_TO_CREDENTIALS}/openai_credentials.txt", "r") as f:
            OPENAI_API_KEY = f.readline().strip()
            openai.api_key = OPENAI_API_KEY

        # Supabase init
        with open(f"{PATH_TO_CREDENTIALS}/supabase_credentials.txt", "r") as f:
            SUPABASE_URL = f.readline().strip()
            SUPABASE_KEY = f.readline().strip()
            self.supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)
            self.supabase_embeddings_db_name = "embeddings"

    def chat(self, prompt: str, config: dict = None) -> str:
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
        context, context_list = utils.get_context(
            supabase_client=self.supabase_client,
            prompt=prompt,
            database=database,
            user_id=self.user_id,
            match_threshold=0.7,
            match_count=10,
        )
        prompt_context = f"{prompt}\nContext to reference: {context}"

        prompt_context = utils.trim_input(prompt_context)

        # Get response
        init_prompt = "You are a helpful chatbot that helps users answer their questions with some additional context."
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

    def upload(self, file_path: str, database: str, file_name: str) -> Dict[str, str]:
        """Segment the text from the provided file, create vectors in OpenAI,
        and insert them into the Supabase database.

        Args:
            file_path (str): The path to the data file to process.
            database (str): The name of the Supabase db data split to insert the vectors into.
            file_name (str): The name of the file to save to S3.

        Returns:
            Dict[str, str]: A dictionary with the result for the file, containing:
                - 'file': The file path,
                - 'status': 'success' or 'failure'.
        """
        # open the text file in a single str object
        with open(file_path, "r") as f:
            text = f.read()

        try:
            # Create the embeddings using OpenAI
            embedding = utils.get_embedding(text)

            # Create the vector to be inserted into Supabase embeddings db
            row = [
                {
                    "user_id": self.user_id,
                    "split": database,
                    "content": text,
                    "embedding": embedding,
                }
            ]
            response = (
                self.supabase_client.table(self.supabase_embeddings_db_name)
                .insert(row)
                .execute()
            )

            # Save the file to S3
            s3_uri = f"customer_data/{self.user_id}/chat/uploaded_data/{file_path.split('/')[-1]}"
            utils.upload_to_s3(
                self.s3,
                file_path,
                S3_BUCKET,
                s3_uri,
            )
            result = {
                "response": f"Successfully uploaded {file_name}",
                "response_code": 200,
            }

        except Exception as e:
            logging.error("Exception occurred", exc_info=True)
            result = {"response": f"Failed to upload {file_name}", "response_code": 500}

        return result
