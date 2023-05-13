# Native imports
import json

# Third party imports
import IPython
import openai


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

    def upload_data(self, data_file_path: str) -> str:
        """Upload a data file to the ContextGPT database.

        Args:
            data_file_path (str): The path to the data file to upload.

        Returns:
            str: The ID of the uploaded data file.
        """
        # TODO not implemented.
        pass

    def _extract_response_content(self, response):
        return response["choices"][0]["message"]["content"]
