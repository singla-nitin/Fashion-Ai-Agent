import os
import requests
from bs4 import BeautifulSoup
from langchain.tools import tool

def fetch_search_results(query: str, num_results: int = 5):
    """
    Uses DuckDuckGo to fetch search result links for the query.
    """
    try:
        search_url = f"https://duckduckgo.com/html/?q={query.replace(' ', '+')}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        # Try multiple selectors for DuckDuckGo results
        selectors = [
            'a.result__a',
            'a[class*="result"]',
            'h2 a',
            '.result h2 a'
        ]
        
        for selector in selectors:
            links = soup.select(selector)
            if links:
                for a in links[:num_results]:
                    href = a.get('href')
                    if href and href.startswith('http'):
                        results.append(href)
                break
        
        # If no results, try to find any links
        if not results:
            all_links = soup.find_all('a', href=True)
            for link in all_links[:num_results*2]:
                href = link.get('href')
                if href and href.startswith('http') and 'fashion' in href.lower():
                    results.append(href)
                    if len(results) >= num_results:
                        break
        
        return results[:num_results]
    except Exception as e:
        print(f"Search error: {e}")
        return []

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
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                page = requests.get(url, timeout=8, headers=headers)
                page.raise_for_status()
                soup = BeautifulSoup(page.text, 'html.parser')
                
                # Try to extract main content: headlines, paragraphs
                title = soup.title.string if soup.title else "Fashion Article"
                title = title.strip()[:100]  # Limit title length
                
                # Get text from paragraphs, focusing on content
                paragraphs = []
                for p in soup.find_all('p'):
                    text = p.get_text().strip()
                    if len(text) > 50 and 'fashion' in text.lower():  # Focus on fashion content
                        paragraphs.append(text)
                
                if paragraphs:
                    content = ' '.join(paragraphs[:2])[:300]  # Limit content length
                    summaries.append(f"**{title}**\n\n{content}...\n\n🔗 [Source]({url})\n")
                
            except Exception as e:
                print(f"Error processing {url}: {e}")
                continue
        
        if summaries:
            return f"## 🔥 Latest Fashion Trends\n\n" + '\n---\n'.join(summaries)
        else:
            # Provide fallback fashion trend information when search fails
            return """## 🌟 Current Fashion Trends (2025)

**🌱 Sustainable Fashion**: Eco-friendly materials and circular fashion continue to dominate, with brands focusing on recycled fabrics and ethical production.

**📱 Tech-Integrated Clothing**: Smart fabrics and wearable technology are becoming mainstream, including temperature-regulating materials and fitness-tracking garments.

**⚖️ Gender-Neutral Design**: Unisex clothing and gender-fluid fashion are gaining popularity across all age groups.

**🎨 Bold Colors & Patterns**: Vibrant hues, artistic prints, and statement patterns are making a comeback after years of minimalism.

**🏠 Comfort-First Fashion**: The work-from-home era continues to influence fashion with comfortable yet stylish pieces that work for multiple occasions.

*Note: This information is based on general fashion industry trends. For the most current information, I recommend checking fashion magazines and trend forecasting websites directly.*"""
            
    except Exception as e:
        return f"I encountered an issue searching for fashion trends. Here's what I can tell you about current fashion: The fashion industry is currently focused on sustainability, comfort, and technology integration. Would you like me to help you with a specific design direction or color palette instead?"
