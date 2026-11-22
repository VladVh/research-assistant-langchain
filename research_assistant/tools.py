"""Tools the research agent can call: web search and page reading."""

from __future__ import annotations

import re

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain_core.tools import tool

MAX_PAGE_CHARS = 8_000
USER_AGENT = "Mozilla/5.0 (compatible; research-assistant/0.1)"


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web. Returns titles, URLs and snippets of the top results.

    Use this to discover sources. Snippets are short; call fetch_page on a
    promising URL to read the actual content before relying on it.
    """
    try:
        results = DDGS().text(query, max_results=max(1, min(max_results, 10)))
    except Exception as exc:  # network errors, rate limits, no results
        return f"Search failed: {exc}"
    if not results:
        return "No results."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.get('title', '')}\n   URL: {r.get('href', '')}\n   {r.get('body', '')}")
    return "\n".join(lines)


@tool
def fetch_page(url: str) -> str:
    """Download a web page and return its readable text (truncated).

    Use this to read a source before citing it.
    """
    try:
        resp = httpx.get(url, headers={"User-Agent": USER_AGENT}, timeout=15, follow_redirects=True)
        resp.raise_for_status()
    except Exception as exc:
        return f"URL: {url}\nFailed to fetch page: {exc}"

    if "html" not in resp.headers.get("content-type", "html"):
        return f"URL: {url}\nUnsupported content type: {resp.headers.get('content-type')}"

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    text = re.sub(r"\n\s*\n+", "\n\n", soup.get_text("\n"))
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) > MAX_PAGE_CHARS:
        text = text[:MAX_PAGE_CHARS] + "\n...[truncated]"
    return f"URL: {url}\nTitle: {title}\n\n{text}"


TOOLS = [web_search, fetch_page]
