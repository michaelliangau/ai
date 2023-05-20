import silicron

API_KEY = "your_api_key_here"
bot = silicron.Silicron(API_KEY)

# Upload data
data_file_path = "./tests/data/test.txt"
data_file_id = bot.upload_data(data_file_path, index_name="test-index")


prompt = "Who is Michael Liang?"

config = {"chatbot": None, "database": None}

response = bot.get_response(prompt, config=config)

print(response)
