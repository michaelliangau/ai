import re
import json
from tqdm import tqdm


def analyze_impact(news_item):
    impact_keywords = [
        "stock",
        "market",
        "economy",
        "recession",
        "inflation",
        "interest rate",
        "unemployment",
        "GDP",
        "gross domestic product",
        "economic growth",
        "trade",
        "import",
        "export",
        "financial",
        "crisis",
        "central bank",
        "stimulus",
        "quantitative easing",
        "debt",
        "bankrupt",
        "default",
        "rating downgrade",
        "investment",
        "monetary policy",
        "fiscal policy",
        "business",
        "industry",
        "consumer",
        "technology",
        "energy",
        "oil",
        "commodity",
        "currency",
        "exchange rate",
        "regulation",
        "employment",
        "earnings",
        "profit",
        "merger",
        "acquisition",
        "IPO",
        "initial public offering",
        "real estate",
        "housing",
        "U.S.",
        "US",
    ]

    categories_to_include = [
        "THE WORLDPOST",
        "POLITICS",
        "BUSINESS",
        "WORLDPOST",
        "WORLD NEWS",
        "U.S. NEWS",
        "TECH",
        "SCIENCE",
        "MONEY",
        "MEDIA",
    ]

    headline = news_item["headline"].lower()
    short_description = news_item["short_description"].lower()
    category = news_item["category"]

    text = headline + " " + short_description

    impact_score = 0

    # Add points to the impact score if the category is included
    if category in categories_to_include:
        impact_score += 1

    for keyword in impact_keywords:
        if re.search(r"\b" + keyword.lower() + r"\b", text):
            impact_score += 1

    return impact_score


file_path = "../context_data/huff_news_2012_2021.json"
output_path = "../context_data/huff_news_with_impact_scores.json"

with open(file_path, "r") as input_file, open(output_path, "w") as output_file:
    for line in tqdm(input_file):
        news_item = json.loads(line)
        impact_score = analyze_impact(news_item)
        news_item["impact_score"] = impact_score
        output_file.write(json.dumps(news_item) + "\n")
