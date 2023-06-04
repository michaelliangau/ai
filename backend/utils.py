import openai
import supabase
import boto3


def get_embedding(text: str, model: str = "text-embedding-ada-002"):
    """
    Get embeddings for the given text using the specified OpenAI model.

    Args:
        text (str): The text to get embeddings for.
        model (str, optional): The name of the OpenAI model to use. Defaults to "text-embedding-ada-002".

    Returns:
        list: The embeddings for the given text.
    """
    text = text.replace("\n", " ")
    embedding = openai.Embedding.create(input=[text], model=model)["data"][0][
        "embedding"
    ]
    return embedding


def get_context(
    supabase_client: supabase.Client,
    prompt: str,
    database: str,
    user_id: int,
    match_threshold: float = 0.8,
    match_count: int = 10,
):
    """
    Get the context for the given prompt from the database.

    Args:
        supabase_client (supabase.Client): The Supabase client.
        prompt (str): The prompt to get the context for.
        database (str): The name of the Supabase db data split to get the context from.
        user_id (int): The ID of the user to get the context for.
        match_threshold (float, optional): The minimum similarity threshold for the context. Defaults to 0.8.
        match_count (int, optional): The number of context items to return. Defaults to 10.

    Returns:
        context (str): The context for the given prompt.
        context_list (list): A list of the context items.
    """
    # Get result from vector db
    query_embedding = get_embedding(
        prompt
    )  # TODO we need the embedding of a test answer.

    response = supabase_client.rpc(
        "search_embeddings",
        params={
            "query_embedding": query_embedding,
            "user_id": user_id,
            "split": database,
            "match_threshold": match_threshold,
            "match_count": match_count,
        },
    ).execute()
    response_data = response.data

    # Extract the context from the response
    context = ""
    context_list = []
    if len(response.data) > 0:
        for data in response_data:
            data_obj = {
                "content": data["content"],
                "similarity": data["similarity"],
            }
            context_list.append(data_obj)
            context += f"{data['content']}\n"

    return context, context_list


def extract_response_content(response):
    """Extract the response content from the OpenAI response."""
    extracted_response = response["choices"][0]["message"]["content"]
    return extracted_response


def set_config(config=None):
    """Set the default config."""
    default_config = {"chatbot": "chatgpt3.5-turbo", "database": ""}

    if config:
        default_config.update((k, v) for k, v in config.items() if v is not None)

    return default_config


def trim_input(input_text: str, max_length: int = 4096) -> str:
    """
    Trims the input string to a specified maximum length.

    If the input string is longer than the maximum length, it's trimmed to the maximum length.

    Args:
        input_text (str): The input string to be trimmed.
        max_length (int): The maximum length for the input string.

    Returns:
        str: The trimmed input string.
    """

    return input_text[:max_length]


def initialise_s3_session(credentials_path):
    """Initialise an S3 session using the credentials in the given file.

    Args:
        credentials_path (str): The path to the credentials file.
    """
    with open(credentials_path, "r") as f:
        AWS_ACCESS_KEY_ID = f.readline().strip()
        AWS_SECRET_ACCESS_KEY = f.readline().strip()
        s3 = boto3.resource(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
    return s3


def upload_to_s3(s3, file_path, bucket_name, destination_path):
    """Upload a file to S3.

    Args:
        s3 (boto3.resource): The S3 session.
        file_path (str): The path to the file to upload.
        bucket_name (str): The name of the bucket to upload to.
        destination_path (str): The path to upload the file to.
    """
    s3.Bucket(bucket_name).upload_file(file_path, destination_path)
