import aiohttp
from langchain_core.tools import tool


SEARXNG_URL = "http://localhost:8080/search"


@tool
async def searxng_search(keywords: list[str], max_results: int = 10) -> list[dict]:
    """
    Search the web using the local SearXNG instance.
    Accepts a list of keywords and returns matching results with title, url, and snippet.
    Use this to discover companies, people, or any web content relevant to a lead-gen query.
    """
    query = " ".join(keywords)
    params = {
        "q": query,
        "format": "json",
        "categories": "general",
        "language": "en",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(SEARXNG_URL, params=params, timeout=aiohttp.ClientTimeout(total=15)) as response:
                response.raise_for_status()
                data = await response.json()
    except aiohttp.ClientConnectorError:
        return [{"error": "SearXNG is not running. Start it with: docker compose up -d"}]
    except aiohttp.ClientResponseError as e:
        return [{"error": f"SearXNG returned HTTP {e.status}"}]
    except Exception as e:
        return [{"error": f"Unexpected error from SearXNG: {str(e)}"}]

    results = []
    for item in data.get("results", [])[:max_results]:
        results.append({
            "title"  : item.get("title", ""),
            "url"    : item.get("url", ""),
            "snippet": item.get("content", ""),
            "engine" : item.get("engine", ""),
        })

    return results
