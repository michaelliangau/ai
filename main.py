import contextgpt

API_KEY = "your_api_key_here"
bot = contextgpt.ContextGPT(API_KEY)

prompt = "What is the capital of France?"

config = {"chatbot": None, "database": None}

response = bot.get_response(prompt, config=config)

print(response)
