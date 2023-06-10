import silicron

# The current active test credentials that work are:
# - Silicron api_key = "dev" returns user_id = 1
# - Supabase default database split = ""
# Initialize bot instance
bot = silicron.Silicron(api_key="dev", chatbot="chatgpt3.5-turbo", database="")

# Chat with the bot
prompt = "Return with Yes."
response = bot.chat(prompt)
print("chat response", response)

# Upload files
response = bot.upload(
    ["tests/data/short_text_file.txt", "tests/data/long_text_file.txt"]
)
print("upload response", response)
