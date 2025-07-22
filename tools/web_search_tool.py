from duckduckgo_search import DDGS
from langchain_core.tools import tool

@tool
def web_search_tool(query: str) -> str:
    """
    Search the web using DuckDuckGo and return top results.
    Useful for finding trends, sourcing fabrics, or general queries.
    """
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, region='wt-wt', safesearch='Moderate', max_results=5)
            if not results:
                return "No search results found."

            formatted_results = [
                f"{idx + 1}. {r['title']} - {r['href']}\n{r['body']}"
                for idx, r in enumerate(results)
            ]
            return "\n\n".join(formatted_results)

    except Exception as e:
        return f"Search failed: {str(e)}"
