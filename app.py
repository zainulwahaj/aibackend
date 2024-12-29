from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from collections import deque
import logging
import asyncio
import httpx
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
from newspaper import Article
from concurrent.futures import ThreadPoolExecutor
import os
import warnings

# Disable symlink warnings from huggingface_hub
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
warnings.filterwarnings("ignore", category=UserWarning, module='huggingface_hub')

# Setup logging to capture only WARNING level and above
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

#########################################################
#                  INITIAL SETUP                        #
#########################################################

app = Flask(__name__)
# Enable CORS for all routes
CORS(app)

# Initialize ThreadPoolExecutor for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=10)

# Initialize global variables for models
bart_tokenizer = None
bart_model = None
sentiment_analyzer = None

MIN_TEXT_LENGTH = 100  # Minimum length for meaningful text

def initialize_models():
    global bart_tokenizer, bart_model, sentiment_analyzer
    try:
        # Initialize the BART summarizer
        logging.warning("Loading BART tokenizer and model...")
        bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        logging.warning("BART models loaded successfully.")

        # Initialize the star-based sentiment analysis pipeline
        logging.warning("Loading sentiment analysis pipeline...")
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )
        logging.warning("Sentiment analysis pipeline loaded successfully.")
    except Exception as e:
        logging.error(f"Error initializing models: {e}")
        raise e

#########################################################
#          PERFORMANCE / CONFIGURATION CONSTANTS        #
#########################################################

MAX_BART_INPUT_TOKENS = 1024        # Maximum tokens to feed into BART
BART_NUM_BEAMS = 2                  # Lower beams => faster summarization
DEFAULT_DEPTH = 5                   # Default number of pages to crawl if none provided
CONCURRENT_REQUESTS = 10            # Number of concurrent HTTP requests

#########################################################
#                HELPER FUNCTIONS                       #
#########################################################

def is_valid_url(url):
    """
    Checks if the provided URL is valid (scheme & netloc).
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

async def fetch_url(client, url):
    """
    Asynchronously fetches the content of a URL using httpx.
    Returns the response text or None if failed.
    """
    try:
        response = await client.get(url, timeout=10)
        if 'text/html' in response.headers.get('Content-Type', ''):
            return response.text
        else:
            logging.warning(f"Non-HTML content at {url}, skipping.")
            return None
    except Exception as e:
        logging.error(f"Error fetching URL {url}: {e}")
        return None

def extract_text(url, html_content):
    """
    Extracts main text from HTML content using Newspaper3k.
    If fails, uses BeautifulSoup as a fallback.
    Returns the extracted text.
    """
    try:
        article = Article(url)
        article.set_html(html_content)
        article.parse()
        text = article.text.strip()
        if len(text) >= MIN_TEXT_LENGTH:
            return text
    except Exception as e:
        logging.warning(f"Newspaper3k failed for {url}: {e}")

    # Fallback to BeautifulSoup
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs]).strip()
        if len(text) >= MIN_TEXT_LENGTH:
            return text
    except Exception as e:
        logging.error(f"BeautifulSoup failed for {url}: {e}")
    return ""

def extract_links(url, html_content):
    """
    Extracts and validates all links from the given HTML content.
    Returns a list of valid URLs.
    """
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        links = [urljoin(url, link['href']) for link in soup.find_all('a', href=True)]
        valid_links = [link for link in links if is_valid_url(link)]
        return valid_links
    except Exception as e:
        logging.error(f"Error extracting links from {url}: {e}")
        return []

def summarize_text_sync(text):
    """
    Summarizes the text using a BART model.
    This is a synchronous function to be run in ThreadPoolExecutor.
    """
    try:
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
    except Exception as e:
        logging.error(f"Error summarizing text: {e}")
        return ""

def analyze_sentiment_sync(text):
    """
    Analyzes the sentiment of the text using the sentiment analysis pipeline.
    This is a synchronous function to be run in ThreadPoolExecutor.
    """
    try:
        result = sentiment_analyzer(text)[0]
        star_label = result["label"]
        stars = int(star_label[0])
        sentiment_label = "NEGATIVE" if stars <= 2 else "NEUTRAL" if stars == 3 else "POSITIVE"
        return {
            "raw_label": star_label,
            "converted_label": sentiment_label,
            "score": result["score"]
        }
    except Exception as e:
        logging.error(f"Error analyzing sentiment: {e}")
        return {
            "raw_label": "N/A",
            "converted_label": "NEUTRAL",
            "score": 0.0
        }

async def process_url(client, url):
    """
    Processes a single URL: fetches content, extracts text, summarizes, and analyzes sentiment.
    Returns the result dictionary or None if skipped.
    """
    html_content = await fetch_url(client, url)
    if not html_content:
        return None

    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(executor, extract_text, url, html_content)
    if not text:
        logging.warning(f"Skipping {url}, text too short or unavailable.")
        return None

    # Summarize text
    summary = await loop.run_in_executor(executor, summarize_text_sync, text)
    sentiment = await loop.run_in_executor(executor, analyze_sentiment_sync, summary)
    links = await loop.run_in_executor(executor, extract_links, url, html_content)

    return {
        "url": url,
        "summary": summary,
        "star_label": sentiment["raw_label"],
        "sentiment_label": sentiment["converted_label"],
        "sentiment_score": sentiment["score"],
        "links": links
    }
async def bfs_scrape_async(start_url, max_pages):
    """
    Asynchronously performs BFS scraping from the start_url up to max_pages.
    Utilizes concurrency and batch processing for speed optimization.
    """
    logging.warning(f"Starting BFS scraping from {start_url} with max_pages={max_pages}")
    visited = set()
    queue = deque([start_url])
    results = []
    pages_crawled = 0

    async with httpx.AsyncClient() as client:
        while queue and pages_crawled < max_pages:
            current_batch = []
            while queue and len(current_batch) < CONCURRENT_REQUESTS and pages_crawled + len(current_batch) < max_pages:
                url = queue.popleft()
                if url in visited:
                    continue
                visited.add(url)
                current_batch.append(url)

            tasks = [asyncio.create_task(process_url(client, url)) for url in current_batch]
            completed, pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

            for task in completed:
                result = task.result()
                if result:
                    results.append({
                        "url": result['url'],
                        "summary": result['summary'],
                        "star_label": result['star_label'],
                        "sentiment_label": result['sentiment_label'],
                        "sentiment_score": result['sentiment_score']
                    })
                    pages_crawled += 1
                    logging.warning(f"Page crawled: {result['url']} (Total crawled: {pages_crawled})")

                    # Add new links to the queue using extracted links
                    for link in result['links']:
                        if link not in visited and is_valid_url(link):
                            queue.append(link)

    logging.warning(f"BFS scraping completed. Total pages crawled: {pages_crawled}")
    return results

async def dfs_scrape_async(start_url, max_pages):
    """
    Asynchronously performs DFS scraping from the start_url up to max_pages.
    Utilizes concurrency and batch processing for speed optimization.
    """
    logging.warning(f"Starting DFS scraping from {start_url} with max_pages={max_pages}")
    visited = set()
    stack = [start_url]
    results = []
    pages_crawled = 0

    async with httpx.AsyncClient() as client:
        while stack and pages_crawled < max_pages:
            current_batch = []
            while stack and len(current_batch) < CONCURRENT_REQUESTS and pages_crawled + len(current_batch) < max_pages:
                url = stack.pop()
                if url in visited:
                    continue
                visited.add(url)
                current_batch.append(url)

            tasks = [asyncio.create_task(process_url(client, url)) for url in current_batch]
            completed, pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

            for task in completed:
                result = task.result()
                if result:
                    results.append({
                        "url": result['url'],
                        "summary": result['summary'],
                        "star_label": result['star_label'],
                        "sentiment_label": result['sentiment_label'],
                        "sentiment_score": result['sentiment_score']
                    })
                    pages_crawled += 1
                    logging.warning(f"Page crawled: {result['url']} (Total crawled: {pages_crawled})")

                    # Add new links to the stack using extracted links
                    for link in result['links']:
                        if link not in visited and is_valid_url(link):
                            stack.append(link)

    logging.warning(f"DFS scraping completed. Total pages crawled: {pages_crawled}")
    return results

#########################################################
#               FLASK ROUTES / ENDPOINTS                #
#########################################################

@app.route('/analyse', methods=['POST'])
async def analyse():
    """
    Endpoint that accepts JSON:
    {
      "url": "<start_url>",
      "method": "bfs" or "dfs",
      "depth": <integer number of pages to scrape>
    }

    Returns a JSON list of results (one object per crawled page),
    including star_label, sentiment_label, and sentiment_score.
    """
    data = request.get_json()
    if not data:
        logging.warning("No JSON body provided in the request.")
        return jsonify({"error": "No JSON body provided."}), 400

    start_url = data.get('url')
    method = data.get('method', 'bfs').lower()
    depth = data.get('depth', DEFAULT_DEPTH)

    # Validate inputs
    if not start_url or not is_valid_url(start_url):
        logging.warning("Invalid or missing 'url' in the request.")
        return jsonify({"error": "A valid 'url' is required."}), 400
    if not isinstance(depth, int) or depth < 1:
        logging.warning("Invalid 'depth' value in the request.")
        return jsonify({"error": "'depth' must be a positive integer."}), 400
    if method not in ['bfs', 'dfs']:
        logging.warning("Invalid 'method' value in the request.")
        return jsonify({"error": "Method must be 'bfs' or 'dfs'."}), 400

    # Run BFS or DFS
    logging.warning(f"Received analyse request: method={method}, depth={depth}, start_url={start_url}")
    if method == 'bfs':
        results = await bfs_scrape_async(start_url, depth)
    else:
        results = await dfs_scrape_async(start_url, depth)

    logging.warning(f"Analyse completed. Returning {len(results)} results.")
    return jsonify(results), 200

#########################################################
#                  RUN THE APP                          #
#########################################################

if __name__ == '__main__':
    try:
        initialize_models()
    except Exception as e:
        logging.critical(f"Failed to initialize models: {e}")
        exit(1)

    # Run the Flask app with asyncio support using Hypercorn
    # Instead of using app.run(), use Hypercorn to serve the app
    # Install Hypercorn via pip: pip install hypercorn

    import hypercorn.asyncio
    import hypercorn.config

    config = hypercorn.config.Config()
    config.bind = ["0.0.0.0:5000"]

    asyncio.run(hypercorn.asyncio.serve(app, config))
