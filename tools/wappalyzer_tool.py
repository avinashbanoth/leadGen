import logging

import aiohttp
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Wappalyzer public API — no auth required, limited to ~50 req/day on free tier
WAPPALYZER_API = "https://api.wappalyzer.com/v2/lookup/"


@tool
async def detect_tech_stack(website_url: str, company_name: str) -> dict:
    """
    Detects the technology stack of a company website using the Wappalyzer API.
    Returns a dict with 'tech_stack' (list of detected technology names) and
    'categories' (e.g. CRM, Analytics, CDN) — used to filter companies by tech.
    Falls back to an empty tech_stack if the API is unreachable.
    This tool does NOT require WAPPALYZER_API_KEY for basic usage.
    """
    params = {"urls": website_url}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                WAPPALYZER_API,
                params=params,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as response:
                if response.status in (401, 403, 429):
                    logger.warning("Wappalyzer API limit/auth issue (status %d) — returning empty.", response.status)
                    return {"company": company_name, "website": website_url, "tech_stack": [], "categories": []}
                response.raise_for_status()
                data = await response.json()
    except Exception as e:
        logger.warning("Wappalyzer failed for '%s': %s", website_url, e)
        return {"company": company_name, "website": website_url, "tech_stack": [], "categories": []}

    technologies = []
    categories   = []

    # Wappalyzer v2 response: list of result objects per URL
    for result in data if isinstance(data, list) else [data]:
        for tech in result.get("technologies", []):
            name = tech.get("name", "")
            if name and name not in technologies:
                technologies.append(name)
            for cat in tech.get("categories", []):
                cat_name = cat.get("name", "")
                if cat_name and cat_name not in categories:
                    categories.append(cat_name)

    logger.info("wappalyzer_tool: detected %d technologies for '%s'.", len(technologies), website_url)

    return {
        "company"   : company_name,
        "website"   : website_url,
        "tech_stack": technologies,
        "categories": categories,
    }
