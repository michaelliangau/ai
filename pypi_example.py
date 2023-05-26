import silicron

bot = silicron.Silicron(api_key="test")
prompt = "test prompt"
config = {"chatbot": "chatgpt3.5-turbo", "database": "test-index"}

response = bot.chat(prompt, config)
print("response", response)
