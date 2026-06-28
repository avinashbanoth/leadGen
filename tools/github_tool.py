import logging
import os

import aiohttp
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"


def _get_headers() -> dict:
    token = os.getenv("GITHUB_TOKEN", "")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _repo_to_signal(repo: dict, company_name: str) -> dict:
    stars = repo.get("stargazers_count", 0)
    strength = "high" if stars > 500 else "medium" if stars > 50 else "low"
    return {
        "company" : company_name,
        "signal"  : f"GitHub repo: {repo.get('full_name', '')} — {repo.get('description', '')[:150]}",
        "source"  : "github",
        "strength": strength,
        "url"     : repo.get("html_url", ""),
    }


@tool
async def search_github_signals(
    company_name: str,
    keywords: list[str],
    max_results: int = 5,
) -> list[dict]:
    """
    Searches GitHub for repositories belonging to or mentioning the company.
    Open-source activity is a strong signal: it reveals tech stack, engineering
    maturity, and active product areas. High-star repos indicate market traction.
    Returns SignalData-shaped dicts with source='github'.
    Works without a token (60 req/hr); set GITHUB_TOKEN for 5000 req/hr.
    Returns [] if the API fails.
    """
    query = f"org:{company_name.lower().replace(' ', '-')} {' '.join(keywords)}"
    params = {"q": query, "sort": "stars", "order": "desc", "per_page": max_results}

    try:
        async with aiohttp.ClientSession(headers=_get_headers()) as session:
            async with session.get(
                f"{GITHUB_API}/search/repositories",
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status == 422:
                    # org: qualifier may not match — fall back to plain keyword search
                    params["q"] = f"{company_name} {' '.join(keywords)}"
                    async with session.get(
                        f"{GITHUB_API}/search/repositories",
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as r2:
                        r2.raise_for_status()
                        data = await r2.json()
                else:
                    response.raise_for_status()
                    data = await response.json()
    except Exception as e:
        logger.warning("GitHub search failed for '%s': %s", company_name, e)
        return []

    signals = [_repo_to_signal(r, company_name) for r in data.get("items", [])[:max_results]]
    logger.info("github_tool: found %d signal(s) for '%s'.", len(signals), company_name)
    return signals
