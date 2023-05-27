from fastapi.testclient import TestClient

# Enable us to import from the parent directory
import sys

sys.path.append(".")
from main import app

client = TestClient(app)


# TODO: Make tests more flushed out once the API is more flushed out
# def test_read_root():
#     response = client.get("/hello")
#     assert response.status_code == 200
