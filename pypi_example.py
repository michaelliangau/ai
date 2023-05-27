import silicron

bot = silicron.Silicron(api_key="test")
prompt = "test prompt"
config = {"chatbot": "chatgpt3.5-turbo", "database": "test-index"}

response = bot.chat(prompt, config)
print("chat response", response)

response = bot.upload(
    ["tests/web/data/short_text_file.txt", "tests/web/data/long_text_file.txt"],
    "test-index",
)
print("upload response", response)
