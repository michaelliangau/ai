from fastapi.testclient import TestClient

# Enable us to import from the parent directory
import sys

sys.path.append(".")
from main import app

client = TestClient(app)


# TODO: Write a test that tests for an API key that exists


# TODO: Write a test that tests for an API key that doesn't exist
