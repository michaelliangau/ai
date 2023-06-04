import silicron

# The current active test credentials that work are:
# - Silicron api_key = "dev" returns user_id = 1
# - Supabase default database split = ""
# Initialize bot instance
bot = silicron.Silicron(api_key="dev")

# Chat with the bot
prompt = "Return with Yes."
config = {"chatbot": "chatgpt3.5-turbo", "database": ""}
response = bot.chat(prompt, config)
print("chat response", response)

# Upload files
# response = bot.upload(
#     ["tests/data/short_text_file.txt", "tests/data/long_text_file.txt"],
#     "test",
# )
# print("upload response", response)
