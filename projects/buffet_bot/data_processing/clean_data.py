"""Remove URL lines from the test articles file."""

with open("../context_data/nytimes_news_articles.txt", "r") as file:
    lines = file.readlines()

filtered_lines = [line for line in lines if not line.startswith("URL:")]

with open("../context_data/nytimes_news_articles_clean.txt", "w") as file:
    file.writelines(filtered_lines)
