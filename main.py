# Native Imports
import os

# Third Party Imports
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from mangum import Mangum

# Local imports
import silicron_backend as silicron

# Environment stage (development/production) defaulting to root if not set.
stage = os.environ.get("STAGE", None)
openapi_prefix = f"/{stage}" if stage else "/"

# Create FastAPI instance
app = FastAPI(title="Silicron", root_path=openapi_prefix)

# Configure templating with Jinja2Templates
templates = Jinja2Templates(directory="templates")


@app.get("/hello", response_class=HTMLResponse)
def root(request: Request):
    """
    Function to handle the root ('/') route of the application.

    Args:
        request (Request): The request object.

    Returns:
        HTMLResponse: The rendered template as an HTML response.
    """
    data = {"items": [{"id": 1, "name": "item1"}, {"id": 2, "name": "item2"}]}
    return templates.TemplateResponse(
        "index.html", {"request": request, "items": data["items"]}
    )


@app.get("/example")
async def example():
    """
    Function to handle the '/example' route of the application.

    Returns:
        JSONResponse: The response from the bot.
    """
    # Set key
    API_KEY = "your_api_key_here"
    bot = silicron.Silicron(API_KEY)

    # Upload data
    data_file_paths = ["tests/data/test.txt"]
    bot.upload(data_file_paths, index_name="test-index")

    # Get response
    prompt = "Who is Michael Liang?"
    config = {"chatbot": None, "database": "test-index"}
    response = bot.ask(prompt, config=config)

    return JSONResponse(content=response)


# Initialize Mangum for AWS Lambda integration
handler = Mangum(app)
