"""
Tools for the multi-agent research pipeline.
- web_search: Tavily-powered web search returning structured results.
- scrape_url: HTTP-based webpage scraper with HTML cleaning.
"""

from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
import json
import logging
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from urllib.parse import urlparse

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tavily client
# ---------------------------------------------------------------------------
_tavily_api_key = os.getenv("TAVILY_API_KEY")
if not _tavily_api_key:
    logger.warning("TAVILY_API_KEY is not set. Web search will fail.")
    tavily = None
else:
    tavily = TavilyClient(api_key=_tavily_api_key)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_SEARCH_RESULTS = 8
SCRAPE_TIMEOUT = 12  # seconds
MAX_SCRAPE_CHARS = 6000  # characters of cleaned text to return per page
SCRAPE_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_domain(url: str) -> str:
    """Extract the domain name from a URL."""
    try:
        return urlparse(url).netloc
    except Exception:
        return ""


def deduplicate_results(results: list[dict]) -> list[dict]:
    """Remove duplicate URLs from a list of search result dicts."""
    seen_urls = set()
    unique = []
    for r in results:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique.append(r)
    return unique


# ---------------------------------------------------------------------------
# Web Search Tool
# ---------------------------------------------------------------------------

@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information.

    Returns a JSON array of objects, each with keys:
    title, url, snippet, source_domain, published_date (if available).
    """
    if tavily is None:
        return json.dumps({
            "error": "TAVILY_API_KEY is not configured. Cannot perform web search.",
            "results": []
        })

    try:
        raw = tavily.search(query=query, max_results=MAX_SEARCH_RESULTS)
    except Exception as e:
        logger.error("Tavily search failed: %s", e)
        return json.dumps({
            "error": f"Search failed: {str(e)}",
            "results": []
        })

    results = []
    for r in raw.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", "")[:500],
            "source_domain": _extract_domain(r.get("url", "")),
            "published_date": r.get("published_date", None),
        })

    results = deduplicate_results(results)
    logger.info("Web search for '%s' returned %d results.", query, len(results))

    return json.dumps({"results": results}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# URL Scraper Tool
# ---------------------------------------------------------------------------

# Tags to remove for cleaner text extraction
_REMOVE_TAGS = [
    "script", "style", "nav", "footer", "header", "aside",
    "form", "iframe", "noscript", "svg", "button",
]

# Content-bearing tags to prioritise when extracting article body
_ARTICLE_SELECTORS = [
    "article",
    "main",
    '[role="main"]',
    ".post-content",
    ".article-body",
    ".entry-content",
    "#content",
]


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    reraise=True,
)
def _fetch_page(url: str) -> requests.Response:
    """Fetch a page with retries for transient network errors."""
    resp = requests.get(
        url,
        timeout=SCRAPE_TIMEOUT,
        headers={"User-Agent": SCRAPE_USER_AGENT},
        allow_redirects=True,
    )
    resp.raise_for_status()
    return resp


def _extract_text(html: str) -> str:
    """Extract clean text from HTML, preferring article content."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise tags
    for tag_name in _REMOVE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Try to find the main article body first
    article_text = ""
    for selector in _ARTICLE_SELECTORS:
        element = soup.select_one(selector)
        if element:
            article_text = element.get_text(separator="\n", strip=True)
            if len(article_text) > 200:
                break

    if len(article_text) > 200:
        text = article_text
    else:
        # Fall back to full body text
        body = soup.find("body")
        text = body.get_text(separator="\n", strip=True) if body else soup.get_text(separator="\n", strip=True)

    # Collapse multiple blank lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading.

    Returns a JSON object with keys: url, title, content, success, error.
    """
    result = {
        "url": url,
        "title": "",
        "content": "",
        "success": False,
        "error": None,
    }

    try:
        resp = _fetch_page(url)

        # Check content type — only process HTML
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            result["error"] = f"Unsupported content type: {content_type}"
            return json.dumps(result, ensure_ascii=False)

        soup = BeautifulSoup(resp.text, "html.parser")
        title_tag = soup.find("title")
        result["title"] = title_tag.get_text(strip=True) if title_tag else ""

        text = _extract_text(resp.text)
        result["content"] = text[:MAX_SCRAPE_CHARS]
        result["success"] = True

        logger.info("Scraped %s — %d chars extracted.", url, len(result["content"]))

    except requests.HTTPError as e:
        result["error"] = f"HTTP error {e.response.status_code}: {str(e)}"
        logger.warning("HTTP error scraping %s: %s", url, e)
    except requests.Timeout:
        result["error"] = "Request timed out."
        logger.warning("Timeout scraping %s", url)
    except requests.ConnectionError as e:
        result["error"] = f"Connection error: {str(e)}"
        logger.warning("Connection error scraping %s: %s", url, e)
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"
        logger.error("Unexpected error scraping %s: %s", url, e, exc_info=True)

    return json.dumps(result, ensure_ascii=False)