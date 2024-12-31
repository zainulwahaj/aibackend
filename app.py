from flask import Flask, request, jsonify
import aiohttp
import asyncio
from bs4 import BeautifulSoup, SoupStrainer
from collections import deque
from urllib.parse import urlparse, urljoin
from flask_cors import CORS
from newspaper import Article
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor
import nest_asyncio

# Apply nest_asyncio to allow asyncio within Flask
nest_asyncio.apply()

# Initialize the Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS

# Configuration constants
MIN_TEXT_LENGTH = 100
MAX_BART_INPUT_TOKENS = 1024
BART_NUM_BEAMS = 2
DEFAULT_DEPTH = 5
MAX_WORKERS = 4  # Adjust based on your server's CPU

# Initialize models and tokenizer
bart_tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')  # Smaller, faster model
bart_model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6')

# Sentiment analysis model loaded
# Using a multi-class model that provides 1-5 star ratings
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

# Initialize ThreadPoolExecutor for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Helper functions
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def normalize_url(url):
    return url.rstrip('/').lower()

async def fetch(session, url):
    try:
        async with session.get(url, timeout=10) as response:
            if 'text/html' in response.headers.get('Content-Type', ''):
                text = await response.text()
                return text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    return None

def fetch_article_text_sync(url):
    try:
        article = Article(url, fetch_images=False, request_timeout=10)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Error parsing article {url}: {e}")
        return ""

def summarize_text_sync(text):
    try:
        inputs = bart_tokenizer.encode(
            "summarize: " + text,
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
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return ""

def analyze_sentiment_sync(text):
    try:
        result = sentiment_analyzer(text)[0]
        label = result['label']
        score = result['score']
        # Extract star rating from label (e.g., "5 stars" -> 5)
        stars = int(label.split()[0])
        # Map stars to sentiment
        if stars <= 2:
            sentiment_label = "NEGATIVE"
        elif stars == 3:
            sentiment_label = "NEUTRAL"
        else:
            sentiment_label = "POSITIVE"
        # Create star_label as "1 star", "2 stars", etc.
        star_label = f"{stars} star" if stars == 1 else f"{stars} stars"
        return {
            "star_label": star_label,
            "sentiment_label": sentiment_label,
            "sentiment_score": score
        }
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return {
            "star_label": "3 stars",
            "sentiment_label": "NEUTRAL",
            "sentiment_score": 0.0
        }

async def process_url(session, url, method, max_pages):
    visited = set()
    results = []
    pages_crawled = 0

    if method == 'bfs':
        queue = deque([url])
    else:  # dfs
        queue = [url]

    while queue and pages_crawled < max_pages:
        if method == 'bfs':
            current_url = queue.popleft()
        else:
            current_url = queue.pop()

        normalized_current_url = normalize_url(current_url)
        if normalized_current_url in visited:
            continue
        visited.add(normalized_current_url)

        html = await fetch(session, current_url)
        if html is None:
            continue

        # Parse links
        links = BeautifulSoup(html, "html.parser", parse_only=SoupStrainer('a'))
        for link in links.find_all('a', href=True):
            full_url = urljoin(current_url, link['href'])
            normalized_full_url = normalize_url(full_url)
            if is_valid_url(full_url) and normalized_full_url not in visited:
                if method == 'bfs':
                    queue.append(full_url)
                else:
                    queue.append(full_url)

        # Extract and process article text
        loop = asyncio.get_event_loop()
        page_text = await loop.run_in_executor(executor, fetch_article_text_sync, current_url)
        if len(page_text.strip()) < MIN_TEXT_LENGTH:
            print(f"Skipped {current_url} due to insufficient content.")
            continue

        # Summarize text
        summary = await loop.run_in_executor(executor, summarize_text_sync, page_text)
        if not summary:
            continue

        # Analyze sentiment
        sentiment = await loop.run_in_executor(executor, analyze_sentiment_sync, summary)

        results.append({
            "url": current_url,
            "summary": summary,
            "star_label": sentiment["star_label"],
            "sentiment_label": sentiment["sentiment_label"],
            "sentiment_score": sentiment["sentiment_score"]
        })
        pages_crawled += 1

    return results

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

    async def run():
        async with aiohttp.ClientSession(headers={"User-Agent": "Mozilla/5.0"}) as session:
            tasks = [process_url(session, url, method, depth) for url in urls]
            results = await asyncio.gather(*tasks)
            flattened_results = [item for sublist in results for item in sublist]
            return flattened_results

    loop = asyncio.get_event_loop()
    try:
        results = loop.run_until_complete(run())
    except Exception as e:
        print(f"Error during processing: {e}")
        return jsonify({"error": "Internal server error."}), 500

    return jsonify(results), 200

# Main entry to run the application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5959, debug=False)
