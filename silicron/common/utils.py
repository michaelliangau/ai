import openai
import pinecone
import IPython

def set_credentials():
    """Set the credentials for OpenAI and Pinecone."""
    # OpenAI
    with open("/Users/michael/Desktop/wip/openai_credentials.txt", "r") as f:
        OPENAI_API_KEY = f.readline().strip()
        openai.api_key = OPENAI_API_KEY

    # Pinecone
    with open("/Users/michael/Desktop/wip/pinecone_credentials.txt", "r") as f:
        PINECONE_API_KEY = f.readline().strip()
        PINECONE_API_ENV = f.readline().strip()
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

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


def get_context(prompt: str, database: str, top_k: int = 10, namespace: str = "data"):
    """
    Get the context for the given prompt from the database.

    Args:
        prompt (str): The prompt to get the context for.
        top_k (int, optional): The number of documents to return. Defaults to 10.
        database (str): The name of the database to get the context from.

    Returns:
        str: The context for the given prompt.
    """
    pinecone_service = pinecone.Index(index_name=database)
    query_embedding = get_embedding(prompt)
    response = pinecone_service.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
    )
    try:
        context = ""
        for doc in response["matches"]:
            context += f"{doc['metadata']['original_text']}"
    except Exception as e:
        context = ""
    return context


def extract_response_content(response):
    """Extract the response content from the OpenAI response."""
    extracted_response = response["choices"][0]["message"]["content"]
    return extracted_response


def set_config(config=None):
    """Set the default config."""
    default_config = {"chatbot": "chatgpt3.5-turbo", "database": "test-index"}

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
