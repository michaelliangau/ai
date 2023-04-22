import json

categories = set()

with open("../context_data/huff_news_2012_2021.json", "r") as f:
    for line in f:
        data = json.loads(line)
        category = data.get("category")
        if category:
            categories.add(category)

print(list(categories))
