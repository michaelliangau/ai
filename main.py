# Native Imports
import os

# Third Party Imports
from fastapi import FastAPI, Request, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import mangum
import IPython

# Local imports
import silicron_backend.api as silicron_api
import silicron_backend.models as silicron_models

# Environment stage (development/production) defaulting to root if not set.
stage = os.environ.get("STAGE", None)
openapi_prefix = f"/{stage}" if stage else "/"

# Create FastAPI instance
app = FastAPI(title="Silicron", root_path=openapi_prefix)

# Configure templating with Jinja2Templates
templates = Jinja2Templates(directory="templates")


@app.get("/hello", response_class=HTMLResponse)
def root(request: Request):
    """Function to handle the root ('/') route of the application.

    Args:
        request (Request): The request object.

    Returns:
        HTMLResponse: The rendered template as an HTML response.
    """
    data = {"items": [{"id": 1, "name": "item1"}, {"id": 2, "name": "item2"}]}
    return templates.TemplateResponse(
        "index.html", {"request": request, "items": data["items"]}
    )


@app.post("/chat")
async def chat_endpoint(chat_input: silicron_models.ChatInput):
    """Function to handle the '/chat' route of the application.

    Args:
        chat_input (ChatInput): A Pydantic model representing the incoming payload,
            which should include a 'body' field with 'prompt' and a 'config' dictionary.

    Returns:
        JSONResponse: The response from the bot.
    """

    # Extract data from chat_input
    prompt = chat_input.prompt
    config = chat_input.config

    # Initialize bot instance
    API_KEY = "your_api_key_here"
    bot = silicron_api.Silicron(API_KEY)

    # Get response
    response = bot.chat(prompt, config=config)

    return JSONResponse(content=response)


@app.post("/upload")
async def upload_endpoint(file: UploadFile, database: str = Form(...)):
    """Function to handle the '/upload' route of the application.

    Args:
        file (UploadFile): The file to be processed and inserted into Pinecone database.
        database (str): The name of the Pinecone index to insert the vectors into.

    Returns:
        JSONResponse: The result of the operation for each file uploaded.
    """
    # Initialize bot instance
    API_KEY = "your_api_key_here"
    bot = silicron_api.Silicron(API_KEY)

    # Read file content
    file_content = await file.read()

    # Write file content to a temp file
    file_name = file.filename
    file_path = f"/tmp/{file_name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    try:
        # Process file
        result = bot.upload(file_path, database, file_name)

        # Return operation result
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# Initialize Mangum for AWS Lambda integration
handler = mangum.Mangum(app)
