# Native imports
from typing import Union, List, Dict, Any
import logging
import os

# Third party imports
import requests
import tqdm

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

    def upload(
        self, files: Union[str, List[str]], database: str = "dev"
    ) -> List[Dict[str, Any]]:
        """Upload data to users' database.

        Args:
            files (Union[str, List[str]]): The path to the data file or a list of
                paths to process.
            database (str): The name of their database to get context from.
                Defaults to 'dev'.

        Returns:
            List[Dict[str, Any]]: The responses from the Silicron API as a list of dictionaries.

        Raises:
            requests.exceptions.HTTPError: If an HTTP error occurs.
            requests.exceptions.RequestException: If a request error occurs.
        """
        # Ensure files is a list
        if isinstance(files, str):
            files = [files]

        # HTTP headers for the request
        headers = {
            "Authorization": self.api_key,
        }

        responses = []

        for file in tqdm.tqdm(files, desc="Uploading files", unit="file"):
            try:
                # Open the file in binary mode
                with open(file, "rb") as f:
                    # HTTP body for the request
                    file_body = {"file": f}
                    data_body = {"database": database}

                    # Send POST request to Silicron API
                    response = self.session.post(
                        self.fn_endpoints["upload"],
                        headers=headers,
                        data=data_body,
                        files=file_body,
                    )

                    # Raise an HTTPError if the response contains an HTTP error status code
                    response.raise_for_status()

                    # Append the response to the list
                    response_json = response.json()
                    responses.append(response_json)

                    logging.info(
                        f"Uploaded {file} successfully, response: {response_json}"
                    )

            except requests.exceptions.HTTPError as http_err:
                logging.error(f"HTTP error occurred while uploading {file}: {http_err}")
            except requests.exceptions.RequestException as req_err:
                logging.error(
                    f"Request error occurred while uploading {file}: {req_err}"
                )

        # Return the JSON response bodies as a list of Python dictionaries
        return responses
