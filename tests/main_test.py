from fastapi.testclient import TestClient

# Enable us to import from the parent directory
import sys
sys.path.append('.')
from main import app

client = TestClient(app)

# TODO: Make tests more flushed out once the API is more flushed out
def test_read_root():
    response = client.get("/hello")
    assert response.status_code == 200

def test_example():
    response = client.get("/example")
    assert response.status_code == 200
    # To make this work, you'd have to mock the response from silicron.Silicron
    # assert response.json() == {"expected": "response"}
