import logging

import aiohttp
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search"

# HN tags that indicate buying / hiring / growth signals
_SIGNAL_TAGS = ["story", "ask_hn", "show_hn"]


def _item_to_signal(hit: dict, company_name: str) -> dict | None:
    """Maps a HN Algolia hit to a SignalData-shaped dict."""
    title = hit.get("title") or hit.get("story_title", "")
    url   = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}"
    points = hit.get("points") or 0
    comments = hit.get("num_comments") or 0

    if not title:
        return None

    strength = "high" if (points > 200 or comments > 50) else "medium" if points > 50 else "low"

    return {
        "company" : company_name,
        "signal"  : title[:200],
        "source"  : "hackernews",
        "strength": strength,
        "url"     : url,
    }


@tool
async def search_hn_signals(
    company_name: str,
    keywords: list[str],
    max_results: int = 10,
) -> list[dict]:
    """
    Searches Hacker News (via Algolia API) for posts mentioning the company or keywords.
    HN is a strong signal source for tech companies: fundraising, product launches,
    hiring announcements, and founder discussions surface here early.
    Returns SignalData-shaped dicts with source='hackernews'.
    Returns [] if the API is unreachable or no results found.
    """
    query = f"{company_name} {' '.join(keywords)}"
    params = {
        "query"      : query,
        "tags"       : "story",
        "hitsPerPage": max_results,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                HN_SEARCH_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                response.raise_for_status()
                data = await response.json()
    except Exception as e:
        logger.warning("HN search failed for '%s': %s", company_name, e)
        return []

    signals = []
    for hit in data.get("hits", [])[:max_results]:
        signal = _item_to_signal(hit, company_name)
        if signal:
            signals.append(signal)

    logger.info("hn_tool: found %d signal(s) for '%s'.", len(signals), company_name)
    return signals
