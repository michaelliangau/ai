# Native imports
from typing import Union, List, Dict, Any
import logging
import os

# Third party imports
import requests
import tqdm

# Our imports
import silicron.utils as utils
import silicron.models as models

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
        self.api_key = api_key
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
        """Send a chat prompt to the Silicron API and get a response.

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

        # HTTP body for the request
        body = {
            "api_key": self.api_key,
            "prompt": prompt,
            "config": config
        }

        try:
            # Send POST request to Silicron API
            response = self.session.post(
                self.fn_endpoints["chat"], json=body
            )

            # Raise an HTTPError if the response contains an HTTP error status code
            response.raise_for_status()

            # Parse the JSON response body into a Python dictionary
            response_dict = response.json()

            # Update the response_code
            response_dict['response_code'] = 200

            # Create an instance of ChatResponse
            chat_response = models.ChatResponse(**response_dict)

            return chat_response.dict()

        except requests.exceptions.HTTPError as http_err:
            return {"response": str(http_err), "response_code": 500}
        except requests.exceptions.RequestException as req_err:
            return {"response": str(req_err), "response_code": 500}


    def upload(
        self, files: Union[str, List[str]], database: str = "dev"
    ) -> List[Dict[str, Any]]:
        """Upload data to users' database.

        We can't use the json parameter in the requests.post() method because
        we need to send a file in the request body. JSON parameter is still the
        preferred way to send data to the API.

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

        responses = []

        for file in tqdm.tqdm(files, desc="Uploading files", unit="file"):
            try:
                # Open the file in binary mode
                with open(file, "rb") as f:
                    # HTTP body for the request
                    file_body = {"file": f}
                    data_body = {
                        "api_key": self.api_key,
                        "database": database
                    }

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

                    # Enforce response schema
                    responses.append(models.UploadResponse(**response_json))

            except FileNotFoundError as fnf_err:
                logging.error(f"File not found: {file}. Error: {fnf_err}")
                responses.append(
                    models.UploadResponse(response=f"File not found: {file}", response_code=404)
                )
            except requests.exceptions.HTTPError as http_err:
                logging.error(f"HTTP error occurred while uploading {file}: {http_err}")
                if response.status_code == 403:
                    responses.append(
                        models.UploadResponse(response="Invalid API Key", response_code=403)
                    )
                    break
                else:
                    responses.append(
                        models.UploadResponse(response="HTTP error occurred", response_code=500)
                    )
            except requests.exceptions.RequestException as req_err:
                logging.error(
                    f"Request error occurred while uploading {file}: {req_err}"
                )
                responses.append(
                    models.UploadResponse(response="Request error occurred", response_code=500)
                )
            except Exception as e:
                logging.error(
                    f"An unexpected error occurred while uploading {file}: {e}"
                )
                responses.append(
                    UploadResponse(response="Unexpected error occurred", response_code=500)
                )

        return responses