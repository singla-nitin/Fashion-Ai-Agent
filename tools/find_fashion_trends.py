import os
import requests
from bs4 import BeautifulSoup
from langchain.tools import tool

def fetch_search_results(query: str, num_results: int = 5):
    """
    Uses DuckDuckGo to fetch search result links for the query.
    """
    search_url = f"https://duckduckgo.com/html/?q={query.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    results = []
    for a in soup.find_all('a', class_='result__a', limit=num_results):
        href = a.get('href')
        if href:
            results.append(href)
    return results

@tool("web_search", return_direct=True)
def web_search(prompt: str, user_id: str = None) -> str:
    """
    Performs a web search for fashion trends and returns a summarized string of key points from the top results.
    Args:
        prompt: The search query about fashion trends (e.g., '2025 summer fashion trends').
        user_id: The ID of the user making the request (automatically handled).
    Returns:
        str: A summary of the fashion trends found online.
    """
    try:
        links = fetch_search_results(prompt)
        summaries = []
        for url in links:
            try:
                page = requests.get(url, timeout=5)
                soup = BeautifulSoup(page.text, 'html.parser')
                # Try to extract main content: headlines, paragraphs
                title = soup.title.string if soup.title else url
                paragraphs = soup.find_all('p')
                text = ' '.join([p.get_text() for p in paragraphs[:3]])
                summaries.append(f"{title}: {text}\nSource: {url}\n")
            except Exception:
                continue
        if not summaries:
            return "No relevant results found."
        return '\n---\n'.join(summaries)
    except Exception as e:
        return f"Failed to perform web search: {e}"
