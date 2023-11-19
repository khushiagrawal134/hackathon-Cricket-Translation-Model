import requests
from bs4 import BeautifulSoup
from transformers import MBartForConditionalGeneration, MBart50Tokenizer

# Define a function to scrape cricket news articles from a website
def scrape_cricket_articles(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    articles = []

    for article in soup.find_all('article'):
        title = article.find('h2').text
        content = article.find('p').text
        articles.append({'title': title, 'content': content})

    return articles

# Example usage
url = 'https://www.cricbuzz.com/'
cricket_articles = scrape_cricket_articles(url)

# Import the model and tokenizer for multi-language translation
model_name = "facebook/mbart-large-50-many-to-many-mmt"  # English to multiple languages
tokenizer = MBart50Tokenizer.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# Example usage
source_text = "He hit a boundary in the 10th over."

# Tokenize and translate using the model
input_ids = tokenizer.encode(source_text, return_tensors="pt", max_length=1024, truncation=True)
translated_ids = model.generate(input_ids, max_length=50, num_beams=5, do_sample=True, no_repeat_ngram_size=2, top_k=50, top_p=0.95, length_penalty=0.8)
translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)

print(translated_text)
