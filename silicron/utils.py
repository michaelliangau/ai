def set_config(config=None):
    """Set the default config.

    Default config is:
    {
        "chatbot": "chatgpt3.5-turbo",
        "database": "dev"
    }

    Args:
        config (dict, optional): A dictionary containing the configuration for the
            chatbot and database. Defaults to None.
    """
    default_config = {"chatbot": "chatgpt3.5-turbo", "database": "dev"}
    if config:
        default_config.update((k, v) for k, v in config.items() if v is not None)
    return default_config
