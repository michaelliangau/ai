import silicron

# Initialize bot instance
bot = silicron.Silicron(api_key="dev")

# Chat with the bot
# prompt = "Return with Yes."
# config = {"chatbot": "chatgpt3.5-turbo", "database": "test-index"}
# response = bot.chat(prompt, config)
# print("chat response", response)

# Upload files
response = bot.upload(
    ["tests/data/short_text_file.txt", "tests/data/long_text_file.txt"],
    "test-index",
)
print("upload response", response)
