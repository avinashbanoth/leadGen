import json
import logging
import os
import re

import aiohttp
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

SEARXNG_URL = "http://localhost:8080/search"

# ---------------------------------------------------------------------------
# Lazy Groq LLM — used only for location expansion (8b-instant, cheap + fast)
# ---------------------------------------------------------------------------

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        from langchain_groq import ChatGroq
        _llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
            max_tokens=50,
        )
    return _llm


# ---------------------------------------------------------------------------
# Location expansion — classifies city / state / country and returns cities
# ---------------------------------------------------------------------------

_EXPAND_PROMPT = """You are a geography classifier.

Given a location name, classify it and respond accordingly:

If it is a CITY:
  Return a JSON array with just that city.
  Example: ["Visakhapatnam"]

If it is a STATE or REGION:
  Return a JSON array of the 5 most commercially important cities in that state/region.
  Example: ["Visakhapatnam", "Vijayawada", "Guntur", "Nellore", "Tirupati"]

If it is a COUNTRY:
  Return a JSON array of the 5 largest commercial cities in that country.
  Example: ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad"]

Location: {location}

Rules:
- Return ONLY a valid JSON array of city name strings
- No explanation, no markdown, no extra text
- Always return between 1 and 6 cities
- Never return an empty array"""


async def _expand_location(location: str) -> list[str]:
    """
    Calls Groq 8b-instant to classify the location (city / state / country)
    and return the 1–5 most commercially important cities within it.
    Falls back to [location] on any error so the caller always gets a usable list.
    """
    prompt = _EXPAND_PROMPT.format(location=location)
    try:
        response = await _get_llm().ainvoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        # Strip any accidental markdown fences
        raw = re.sub(r"^```[a-z]*\n?", "", raw).strip().rstrip("```").strip()
        cities = json.loads(raw)
        if isinstance(cities, list) and cities:
            logger.info(
                "searxng_tool: expanded '%s' → %s", location, cities
            )
            return [str(c) for c in cities]
    except Exception as e:
        logger.warning(
            "searxng_tool: location expansion failed for '%s' — %s. "
            "Falling back to single query.",
            location, e,
        )
    return [location]


# ---------------------------------------------------------------------------
# URL deduplication — pure Python, zero tokens
# ---------------------------------------------------------------------------

def _deduplicate_by_url(results: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for r in results:
        url = r.get("url", "")
        if url not in seen:
            seen.add(url)
            unique.append(r)
    return unique


# ---------------------------------------------------------------------------
# SearXNG search tool
# ---------------------------------------------------------------------------

@tool
async def searxng_search(
    keywords   : list[str],
    max_results: int = 10,
    location   : str | None = None,
) -> list[dict]:
    """
    Search the web using the local SearXNG instance.

    When location is provided:
      - Uses categories=map, which routes to map-indexed business listings
        instead of news articles and government portals.
      - Expands state/country locations to major cities via Groq 8b, then
        runs one query per city and merges deduplicated results.

    When location is None:
      - Uses categories=general (original behavior, unchanged).

    Returns results with title, url, snippet, and engine fields.
    """
    base_query = " ".join(keywords)

    # ── No location: original single-query general search ───────────────────
    if not location:
        params = {
            "q"         : base_query,
            "format"    : "json",
            "categories": "general",
            "language"  : "en",
        }
        return await _run_searxng_query(params, max_results)

    # ── Location present: expand → per-city map queries → deduplicate ───────
    cities = await _expand_location(location)

    all_results: list[dict] = []
    for city in cities:
        # Replace the bare location name in the query with the specific city,
        # or append the city when it's not already in the keyword list.
        if location.lower() in base_query.lower():
            city_query = re.sub(re.escape(location), city, base_query, flags=re.IGNORECASE)
        else:
            city_query = f"{base_query} {city}"

        params = {
            "q"         : city_query,
            "format"    : "json",
            "categories": "map",
            "language"  : "en",
        }
        logger.info("searxng_tool: map query for city '%s' — %s", city, city_query)
        city_results = await _run_searxng_query(params, max_results)
        all_results.extend(city_results)

    unique = _deduplicate_by_url(all_results)
    logger.info(
        "searxng_tool: %d raw results across %d city/cities → %d unique.",
        len(all_results), len(cities), len(unique),
    )
    return unique[:max_results]


# ---------------------------------------------------------------------------
# Internal HTTP helper — one SearXNG request, returns list[dict]
# ---------------------------------------------------------------------------

async def _run_searxng_query(params: dict, max_results: int) -> list[dict]:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                SEARXNG_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as response:
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
