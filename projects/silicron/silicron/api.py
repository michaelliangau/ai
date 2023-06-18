# Native imports
from typing import Union, List, Dict, Any
import logging
import os

# Third party imports
import requests
import tqdm

# Constants
STAGING_API_ENDPOINT = "ADD_THE_STAGING_API_ENDPOINT_HERE"


class Silicron:
    """Silicron Class

    Attributes:
        api_key (str): The API key to use for authentication.
        chatbot (str): The chatbot to use for chat.
        database (str): The database to use for chat.
        api_endpoint (str): The API endpoint to use for requests.
        fn_endpoints (Dict[str, str]): A dictionary containing the API endpoints
            for each function.
        session (requests.Session): A requests session object to use for requests.
    """

    def __init__(
        self, api_key: str = "", chatbot: str = "chatgpt3.5-turbo", database: str = ""
    ):
        """Initialize the Silicron class.

        Args:
            api_key (str): The API key to use for authentication.
            chatbot (str): The chatbot to use for chat.
            database (str): The database to use for chat.
        """
        # Chatbot config.
        self.api_key = api_key
        self.chatbot = chatbot
        self.database = database

        # Network config.
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

    def chat(self, prompt: str) -> Dict[str, Any]:
        """Send a chat prompt to the Silicron API and get a response.

        Args:
            prompt (str): The chat prompt to send to the Silicron API.

        Returns:
            Dict[str, Any]: The response from the Silicron API as a dictionary.

        Raises:
            requests.exceptions.HTTPError: If an HTTP error occurs.
            requests.exceptions.RequestException: If a request error occurs.
        """
        # HTTP body for the request
        body = {
            "api_key": self.api_key,
            "prompt": prompt,
            "config": {
                "chatbot": self.chatbot,
                "database": self.database,
            },
        }

        try:
            # Send POST request to Silicron API
            response = self.session.post(self.fn_endpoints["chat"], json=body)

            # Raise an HTTPError if the response contains an HTTP error status code
            response.raise_for_status()

            # Parse the JSON response body into a Python dictionary
            response_dict = response.json()

            # Update the response_code
            response_dict["response_code"] = 200

            return response_dict

        except requests.exceptions.HTTPError as http_err:
            return {"response": str(http_err), "response_code": 500}
        except requests.exceptions.RequestException as req_err:
            return {"response": str(req_err), "response_code": 500}

    def upload(self, files: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """Upload data to users' database.

        We can't use the json parameter in the requests.post() method because
        we need to send a file in the request body. JSON parameter is still the
        preferred way to send data to the API.

        TODO: Uploading a file into a non-existent index returns a 200 status code.

        Args:
            files (Union[str, List[str]]): The path to the data file or a list of
                paths to process.

        Returns:
            List[Dict[str, Any]]: The responses from the Silicron API as a list of dictionaries.

        Raises:
            requests.exceptions.HTTPError: If an HTTP error occurs.
            requests.exceptions.RequestException: If a request error occurs.
        """
        # Ensure files is a list
        if isinstance(files, str):
            files = [files]

        responses = []

        for file in tqdm.tqdm(files, desc="Uploading files", unit="file"):
            try:
                # Open the file in binary mode
                with open(file, "rb") as f:
                    # HTTP body for the request
                    file_body = {"file": f}
                    data_body = {"api_key": self.api_key, "database": self.database}

                    # Send POST request to Silicron API
                    response = self.session.post(
                        self.fn_endpoints["upload"],
                        data=data_body,
                        files=file_body,
                    )

                    # Raise an HTTPError if the response contains an HTTP error status code
                    response.raise_for_status()

                    # Convert the response to a JSON object
                    response_json = response.json()

                    # Add a response_code field to the response
                    response_json["response_code"] = response.status_code

                    responses.append(response_json)  # Append the dictionary directly

            except FileNotFoundError as fnf_err:
                logging.error(f"File not found: {file}. Error: {fnf_err}")
                responses.append(
                    {"response": f"File not found: {file}", "response_code": 404}
                )
            except requests.exceptions.HTTPError as http_err:
                logging.error(f"HTTP error occurred while uploading {file}: {http_err}")
                if response.status_code == 403:
                    responses.append(
                        {"response": "Invalid API Key", "response_code": 403}
                    )
                    break
                else:
                    responses.append(
                        {"response": "HTTP error occurred", "response_code": 500}
                    )
            except requests.exceptions.RequestException as req_err:
                logging.error(
                    f"Request error occurred while uploading {file}: {req_err}"
                )
                responses.append(
                    {"response": "Request error occurred", "response_code": 500}
                )
            except Exception as e:
                logging.error(
                    f"An unexpected error occurred while uploading {file}: {e}"
                )
                responses.append(
                    {"response": "Unexpected error occurred", "response_code": 500}
                )

        return responses
