# Native imports
from typing import Union, List, Dict, Any
import logging
import os

# Third party imports
import requests

# Our imports
import silicron.utils as utils

# Constants
STAGING_API_ENDPOINT = "https://wsesuzvgd0.execute-api.us-east-1.amazonaws.com/staging"


class Silicron:
    def __init__(self, api_key: str = ""):
        """Initialize the Silicron class.

        Args:
            api_key (str): The API key to use for authentication.
            api_endpoint (str): The API endpoint to use for requests.
            fn_endpoints (Dict[str, str]): A dictionary containing the API endpoints
                for each function.
            session (requests.Session): A requests session object to use for requests.
        """
        self.api_key = api_key  # TODO: Authenticate with API key
        self.api_endpoint = os.getenv(
            "SILICRON_LOCAL_API_ENDPOINT", STAGING_API_ENDPOINT
        )
        self.fn_endpoints = {
            "chat": f"{self.api_endpoint}/chat",
            "upload": f"{self.api_endpoint}/upload",
        }
        self.session = requests.Session()

        # Set logging level
        logging.basicConfig(level=logging.INFO)

    def chat(self, prompt: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Send a chat prompt to the Silicron API and get a response.

        Args:
            prompt (str): The chat prompt to send to the Silicron API.
            config (Dict[str, Any], optional): A dictionary containing additional
                configuration for the API. Defaults to None.

        Returns:
            Dict[str, Any]: The response from the Silicron API as a dictionary.

        Raises:
            requests.exceptions.HTTPError: If an HTTP error occurs.
            requests.exceptions.RequestException: If a request error occurs.
        """
        # Set default config if none provided
        config = utils.set_config(config)

        # HTTP headers for the request
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.api_key,
        }

        # HTTP body for the request
        body = {"prompt": prompt, "config": config}

        try:
            # Send POST request to Silicron API
            response = self.session.post(
                self.fn_endpoints["chat"], headers=headers, json=body
            )

            # Raise an HTTPError if the response contains an HTTP error status code
            response.raise_for_status()
        except requests.exceptions.HTTPError as http_err:
            return {"error": str(http_err)}
        except requests.exceptions.RequestException as req_err:
            return {"error": str(req_err)}

        # Return the JSON response body as a Python dictionary
        return response.json()

    def upload(self, file_paths: Union[str, List[str]], database: str = "dev") -> int:
        """Upload data to users' database.

        This function will upload the given data to the users' specified database,
        allowing it to be queried in function.

        Args:
            file_paths (Union[str, List[str]]): The path to the data file or a list of
                paths to process.
            database (str): The name of their database to get context from.
                Defaults to 'dev'.

        Returns:
            int: The HTTP status code returned from the server.
        """
        raise not NotImplementedError
