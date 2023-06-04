import pytest
import sys
import subprocess

# Our imports
sys.path.append("..")  # Adds higher directory to python modules path.
import silicron

# Constants
VALID_API_KEY = "dev"  # Make sure that the DynamoDB contains this.
INVALID_API_KEY = (
    "invalid_api_key"  # Make sure that the DynamoDB does not contain this.
)


class TestSilicron:
    @pytest.fixture(autouse=True)
    def setup_bot(self):
        # Check if uvicorn is running on port 8000
        try:
            result = subprocess.check_output(["lsof", "-i", ":8000"])
        except subprocess.CalledProcessError:
            raise Exception(
                "Uvicorn is not running on port 8000. Please start it in another screen with `make debug`"
            )

        self.valid_bot = silicron.Silicron(api_key=VALID_API_KEY)
        self.invalid_bot = silicron.Silicron(api_key=INVALID_API_KEY)

    def test_chat_successful(self):
        prompt = "test prompt"
        config = {"chatbot": "chatgpt3.5-turbo", "database": "0-dev"}
        response = self.valid_bot.chat(prompt, config)
        assert response["response_code"] == 200

    def test_chat_invalid_api_key(self):
        prompt = "test prompt"
        config = {"chatbot": "chatgpt3.5-turbo", "database": "0-dev"}
        response = self.invalid_bot.chat(prompt, config)
        assert response["response_code"] == 500

    def test_chat_unsupported_chatbot(self):
        prompt = "test prompt"
        config = {"chatbot": "chatgpt420", "database": "0-dev"}
        with pytest.raises(ValueError):
            self.valid_bot.chat(prompt, config)

    def test_upload_list_successful(self):
        response = self.valid_bot.upload(
            ["tests/data/short_text_file.txt", "tests/data/long_text_file.txt"],
            "0-dev",
        )
        for r in response:
            assert r.response_code == 200

    def test_upload_string_successful(self):
        response = self.valid_bot.upload(
            "tests/data/short_text_file.txt",
            "0-dev",
        )
        print(response)
        assert response[0].response_code == 200

    def test_upload_invalid_api_key(self):
        response = self.invalid_bot.upload(
            "tests/data/short_text_file.txt",
            "0-dev",
        )
        assert response[0].response_code == 403

    def test_upload_nonexistent_file(self):
        response = self.valid_bot.upload(["tests/data/nonexistent.txt"], "0-dev")
        assert response[0].response_code == 404
