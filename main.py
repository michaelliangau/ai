import os

from fastapi import FastAPI
from mangum import Mangum

stage = os.environ.get('STAGE', None)
openapi_prefix = f"/{stage}" if stage else "/"

app = FastAPI(title="MyAwesomeApp", openapi_prefix=openapi_prefix) # Here is the magic


@app.get("/hello")
def hello_world():
    return {"message": "Hello World"}


handler = Mangum(app)