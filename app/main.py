from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from mangum import Mangum
import os

stage = os.environ.get('STAGE', None)
openapi_prefix = f"/{stage}" if stage else "/"

app = FastAPI(title="Silicron", openapi_prefix=openapi_prefix)

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    data = {"items": [{"id": 1, "name": "item1"}, {"id": 2, "name": "item2"}]}
    return templates.TemplateResponse("index.html", {"request": request, "items": data["items"]})

handler = Mangum(app)
