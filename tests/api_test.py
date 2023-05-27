import pytest
import subprocess
import time

# Our imports
import sys

sys.path.append("..")  # Adds higher directory to python modules path.
import silicron


def test_chat_successful():
    bot = silicron.Silicron(api_key="test")
    prompt = "test prompt"
    config = {"chatbot": "chatgpt3.5-turbo", "database": "test-index"}
    response = bot.chat(prompt, config)
    assert response["response_code"] == 200


def test_chat_unsupported_chatbot():
    bot = silicron.Silicron(api_key="test")
    prompt = "test prompt"
    config = {"chatbot": "chatgpt420", "database": "test-index"}
    try:
        response = bot.chat(prompt, config)
    except ValueError:
        assert True
    else:
        assert False, "Expected ValueError to be raised"


def test_upload_list_successful():
    bot = silicron.Silicron(api_key="test")
    response = bot.upload(
        ["tests/data/short_text_file.txt", "tests/data/long_text_file.txt"],
        "test-index",
    )
    for r in response:
        assert r["response_code"] == 200


def test_upload_string_successful():
    bot = silicron.Silicron(api_key="test")
    response = bot.upload(
        "tests/data/short_text_file.txt",
        "test-index",
    )
    assert response[0]["response_code"] == 200


def test_upload_nonexistent_file():
    bot = silicron.Silicron(api_key="test")
    response = bot.upload(["tests/data/nonexistent.txt"], "test-index")
    assert response[0]["response_code"] == 500
