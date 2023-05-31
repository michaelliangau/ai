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

# Local imports
import silicron_backend.api as silicron_api
import silicron_backend.models as silicron_models

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


@app.get("/api/python")
def hello_world():
    return {"message": "Hello World"}


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    """Function to handle the root ('/') route of the application.

    TODO (P0): Return a simple landing page with:
    - Minimal connection to React frontend (P0)
    - Typeform link (P0) - MVP use a typeform link, P1 build the sign up flows etc.
    - Add Google Analytics (P0)
    - Sign up/login (P1).

    Home page content should have:
    - 1 sentence - What does this app do?
    - 1 code block - how do I use it?
    - 1 sign up for early access button.
    - Use a nice looking template (example webste - https://www.assemblyai.com/)
    - Clean code (build in a way a non-FE developer can extend it)
    - Nothing else.

    Args:
        request (Request): The request object.

    Returns:
        HTMLResponse: The rendered template as an HTML response.
    """
    # TODO: Delete this - example snippet how to send data to the template
    # data = {"items": [{"id": 1, "name": "item1"}, {"id": 2, "name": "item2"}]}
    # return templates.TemplateResponse(
    #     "index.html", {"request": request, "items": data["items"]}
    # )
    raise NotImplementedError


# TODO (P1): Login/Sign up flow (Google sign in only)


# TODO (P1): Payment flow (Stripe - link a payment method to a user)


# TODO (P2): Dashboard (simple metrics)


@app.post("/chat")
async def chat_endpoint(body: silicron_models.ChatInput):
    """Function to handle the '/chat' route of the application.

    Args:
        body (silicron_models.ChatInput): The request body.

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
    except (BotoCoreError, ClientError) as error:
        raise HTTPException(status_code=400, detail=str(error))
    if "Item" not in response:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    if "user_id" not in response["Item"]:
        raise HTTPException(
            status_code=500, detail="Retrieving account details failed")
    user_id = response["Item"]["user_id"]

    # Initialize bot instance
    bot = silicron_api.Silicron(user_id)

    # Get response
    response = bot.chat(prompt, config=config)

    return JSONResponse(content=response)


@app.post("/upload")
async def upload_endpoint(
    file: UploadFile, api_key: str = Form(...), database: str = Form(...)
):
    """Function to handle the '/upload' route of the application.

    Args:
        file (UploadFile): The file to be processed and inserted into Pinecone database.
        api_key (str): The API key of the user.
        database (str): The name of the Pinecone index to insert the vectors into.

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
        raise HTTPException(
            status_code=500, detail="Retrieving account details failed")
    user_id = response["Item"]["user_id"]

    # Initialize bot instance
    bot = silicron_api.Silicron(user_id)

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
