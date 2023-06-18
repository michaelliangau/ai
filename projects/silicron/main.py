# Native Imports
import os

# Third Party Imports
from fastapi import FastAPI, Request, UploadFile, HTTPException, Form, APIRouter
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import mangum

# import IPython
from botocore.exceptions import BotoCoreError, ClientError
import boto3
from dotenv import load_dotenv

# Local imports
import backend.api as backend_api
import backend.models as backend_models

# Load environment variables
load_dotenv()

# Environment stage (development/production) defaulting to root if not set.
stage = os.environ.get("STAGE", None)
openapi_prefix = f"/{stage}" if stage else "/"

# Create FastAPI instance
app = FastAPI(title="Silicron", root_path=openapi_prefix)

# Configure templating with Jinja2Templates.
templates = Jinja2Templates(directory="templates")

# Define your DynamoDB resource using boto3
dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
table = dynamodb.Table(
    "silicron_dev_api_keys"
)  # TODO (GA): Change this to silicron_prod_api_keys


@app.post("/chat")
async def chat_endpoint(body: backend_models.ChatInput):
    """Function to handle the '/chat' route of the application.

    Args:
        body (backend_models.ChatInput): The request body.

    Returns:
        JSONResponse: The response from the bot.
    """
    # Get request body
    prompt = body.prompt
    config = body.config
    api_key = body.api_key

    # Check if API Key exists in DynamoDB table and get user_id
    try:
        response = table.get_item(Key={"api_key": api_key})
    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=500, detail="An unknown error occurred")
    if "Item" not in response:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    if "user_id" not in response["Item"]:
        raise HTTPException(status_code=500, detail="Retrieving account details failed")
    user_id = response["Item"]["user_id"]

    # Initialize bot instance
    bot = backend_api.Silicron(user_id)

    # Get response
    response = bot.chat(prompt, config=config)

    return JSONResponse(content=response)


@app.post("/upload")
async def upload_endpoint(
    file: UploadFile, api_key: str = Form(...), database: str = Form("")
):
    """Function to handle the '/upload' route of the application.

    Args:
        file (UploadFile): The file to be processed and inserted into Supabase database.
        api_key (str): The API key of the user.
        database (str): The name of the Supabase index to insert the vectors into.

    Returns:
        JSONResponse: The result of the operation for each file uploaded.
    """
    # Check if API Key exists in DynamoDB table and get user_id
    try:
        response = table.get_item(Key={"api_key": api_key})
    except (BotoCoreError, ClientError) as error:
        raise HTTPException(status_code=400, detail=str(error))

    if "Item" not in response:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    if "user_id" not in response["Item"]:
        raise HTTPException(status_code=500, detail="Retrieving account details failed")
    user_id = response["Item"]["user_id"]

    # Initialize bot instance
    bot = backend_api.Silicron(user_id)

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
        # Delete temp file
        if os.path.exists(file_path):
            os.remove(file_path)


# Initialize Mangum for AWS Lambda integration
handler = mangum.Mangum(app)
