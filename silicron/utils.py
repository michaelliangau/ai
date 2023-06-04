# Define supported chatbots
SUPPORTED_CHATBOTS = ["chatgpt3.5-turbo"]


def set_config(config=None):
    """Set the default config.

    Default config is:
    {
        "chatbot": "chatgpt3.5-turbo",
        "database": ""
    }

    Args:
        config (dict, optional): A dictionary containing the configuration for the
            chatbot and database. Defaults to None.

    Raises:
        ValueError: If the provided chatbot is not supported.
    """
    default_config = {"chatbot": "chatgpt3.5-turbo", "database": ""}
    if config:
        for k, v in config.items():
            if v is not None:
                if k == "chatbot":
                    if v not in SUPPORTED_CHATBOTS:
                        raise ValueError(f"Chatbot '{v}' is not supported.")
                default_config[k] = v
    return default_config
