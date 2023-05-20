import silicron

# Set key
API_KEY = "your_api_key_here"
bot = silicron.Silicron(API_KEY)

# Upload data
data_file_paths = "./tests/data/test.txt"
bot.upload_data(data_file_paths, index_name="test-index")

# Get response
prompt = "Who is Michael Liang?"
config = {"chatbot": None, "database": "test-index"}
response = bot.get_response(prompt, config=config)

print(response)
