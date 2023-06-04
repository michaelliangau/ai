import silicron

# The current active test credentials that work are:
# - Silicron api_key = "dev" returns user_id = 1
# - Pinecone database = "0-dev"
# Initialize bot instance
bot = silicron.Silicron(api_key="dev")

# Chat with the bot
# prompt = "Return with Yes."
# config = {"chatbot": "chatgpt3.5-turbo", "database": "0-dev"}
# response = bot.chat(prompt, config)
# print("chat response", response)

# Upload files
response = bot.upload(
    ["tests/data/short_text_file.txt", "tests/data/long_text_file.txt"],
    "test",
)
print("upload response", response)

exit()

# SUPABASE MVP TODO DELETE
from supabase import create_client, Client
import numpy as np

url: str = "https://rhrszihruweifvjczkmh.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJocnN6aWhydXdlaWZ2amN6a21oIiwicm9sZSI6ImFub24iLCJpYXQiOjE2ODU2MDUzMjAsImV4cCI6MjAwMTE4MTMyMH0.0ge0isccOYhI5--jinj8iAcMCZHWSpD6UXpYfAuLzgc"
supabase: Client = create_client(url, key)


# Insert data
# data = [
#     {"user_id": 1, "split": "all", "content": "Test", "embedding": np.random.rand(1536).tolist()},
#     {"user_id": 2, "split": "all", "content": "Another Test", "embedding": np.random.rand(1536).tolist()},
# ]
# response = supabase.table("embeddings").insert(data).execute()


# Similarity search
response = supabase.rpc("search_embeddings", params={
    "query_embedding": np.random.rand(1536).tolist(),
    "user_id": 1,
    "split": "all",
    "match_threshold": 0.1,
    "match_count": 10,
}).execute()

print(response)

