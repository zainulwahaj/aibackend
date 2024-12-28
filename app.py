from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
from collections import deque
from urllib.parse import urlparse
from flask_cors import CORS
from newspaper import Article
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor

#########################################################
#                  INITIAL SETUP                        #
#########################################################

app = Flask(__name__)
CORS(app)
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

#########################################################
#          PERFORMANCE / CONFIGURATION CONSTANTS        #
#########################################################

MIN_TEXT_LENGTH = 500
MAX_BART_INPUT_TOKENS = 1024
BART_NUM_BEAMS = 2
DEFAULT_DEPTH = 5

#########################################################
#                HELPER FUNCTIONS                       #
#########################################################

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def fetch_article_text(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def summarize_text(text):
    inputs = bart_tokenizer.encode(
        "Summarize: " + text,
        return_tensors="pt",
        max_length=MAX_BART_INPUT_TOKENS,
        truncation=True
    )
    summary_ids = bart_model.generate(
        inputs,
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=BART_NUM_BEAMS,
        early_stopping=True
    )
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def map_stars_to_sentiment(star_label):
    try:
        stars = int(star_label[0])
    except:
        return "NEUTRAL"
    return "NEGATIVE" if stars <= 2 else "NEUTRAL" if stars == 3 else "POSITIVE"

def analyze_sentiment(text):
    raw_result = sentiment_analyzer(text)[0]
    star_label = raw_result["label"]
    converted_label = map_stars_to_sentiment(star_label)
    return {
        "raw_label": star_label,
        "converted_label": converted_label,
        "score": raw_result["score"]
    }

#########################################################
#               BFS AND DFS SCRAPING                    #
#########################################################

def bfs_scrape(start_url, max_pages):
    visited = set()
    queue = deque([start_url])
    results = []
    pages_crawled = 0

    while queue and pages_crawled < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            if 'text/html' in response.headers.get('Content-Type', ''):
                page_text = fetch_article_text(url)
                if len(page_text.strip()) < MIN_TEXT_LENGTH:
                    continue
                summary = summarize_text(page_text)
                sentiment = analyze_sentiment(summary)
                results.append({
                    "url": url,
                    "summary": summary,
                    "star_label": sentiment["raw_label"],
                    "sentiment_label": sentiment["converted_label"],
                    "sentiment_score": sentiment["score"]
                })
                pages_crawled += 1
                soup = BeautifulSoup(response.text, "html.parser")
                for link in soup.find_all('a', href=True):
                    full_url = requests.compat.urljoin(url, link['href'])
                    if is_valid_url(full_url) and full_url not in visited:
                        queue.append(full_url)
        except Exception as e:
            print(f"Error scraping {url}: {e}")
    return results

def dfs_scrape(start_url, max_pages):
    visited = set()
    stack = [start_url]
    results = []
    pages_crawled = 0

    while stack and pages_crawled < max_pages:
        url = stack.pop()
        if url in visited:
            continue
        visited.add(url)
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            if 'text/html' in response.headers.get('Content-Type', ''):
                page_text = fetch_article_text(url)
                if len(page_text.strip()) < MIN_TEXT_LENGTH:
                    continue
                summary = summarize_text(page_text)
                sentiment = analyze_sentiment(summary)
                results.append({
                    "url": url,
                    "summary": summary,
                    "star_label": sentiment["raw_label"],
                    "sentiment_label": sentiment["converted_label"],
                    "sentiment_score": sentiment["score"]
                })
                pages_crawled += 1
                soup = BeautifulSoup(response.text, "html.parser")
                for link in soup.find_all('a', href=True):
                    full_url = requests.compat.urljoin(url, link['href'])
                    if is_valid_url(full_url) and full_url not in visited:
                        stack.append(full_url)
        except Exception as e:
            print(f"Error scraping {url}: {e}")
    return results

#########################################################
#               FLASK ROUTES / ENDPOINTS                #
#########################################################

@app.route('/analyse', methods=['POST'])
def analyse():
    data = request.get_json()
    urls = data.get('urls', [data.get('url')])  # Support batch or single URL
    method = data.get('method', 'bfs').lower()
    depth = data.get('depth', DEFAULT_DEPTH)

    if not urls or not all(is_valid_url(url) for url in urls):
        return jsonify({"error": "A valid 'url' or 'urls' is required."}), 400
    if method not in ['bfs', 'dfs']:
        return jsonify({"error": "Method must be 'bfs' or 'dfs'"}), 400

    # Use ThreadPoolExecutor to handle multiple URLs
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda url: bfs_scrape(url, depth) if method == 'bfs' else dfs_scrape(url, depth), urls))
        flattened_results = [item for sublist in results for item in sublist]

    return jsonify(flattened_results), 200

#########################################################
#                  RUN THE APP                          #
#########################################################

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
