# import silicron

# # Initialize bot instance
# bot = silicron.Silicron(api_key="dev")

# # Chat with the bot
# prompt = "Return with Yes."
# config = {"chatbot": "chatgpt3.5-turbo", "database": "test-index"}
# response = bot.chat(prompt, config)
# print("chat response", response)

# # Upload files
# response = bot.upload(
#     ["tests/data/short_text_file.txt", "tests/data/long_text_file.txt"],
#     "0-dev",
# )
# print("upload response", response)



from supabase import create_client, Client
import numpy as np
import asyncio

url: str = "https://rhrszihruweifvjczkmh.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJocnN6aWhydXdlaWZ2amN6a21oIiwicm9sZSI6ImFub24iLCJpYXQiOjE2ODU2MDUzMjAsImV4cCI6MjAwMTE4MTMyMH0.0ge0isccOYhI5--jinj8iAcMCZHWSpD6UXpYfAuLzgc"
supabase: Client = create_client(url, key)

# Create new table
# TODO programmatically create a new table




# Upload data
# data = supabase.table("documents").insert(
#     {"id": 0,
#     "embedding": np.random.rand(1536).tolist(),
#     "content": "Test"
#     }
#     ).execute()
# print(data)    
# assert len(data.data) > 0


# Similarity search
# TODO dynamic function on different tables
response = supabase.rpc("match_documents", params={
    "query_embedding": np.random.rand(1536).tolist(),
    "match_threshold": 0.8,
    "match_count": 10,
}).execute()

print(response)

