import logging
import os

import aiohttp
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_APOLLO_BASE  = "https://api.apollo.io/v1"
_PER_PAGE     = 10
_TIMEOUT      = 20.0


def _api_key() -> str:
    return os.getenv("APOLLO_API_KEY", "")


def _headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Cache-Control": "no-cache",
        "X-Api-Key"   : _api_key(),
    }


async def _post(session: aiohttp.ClientSession, path: str, payload: dict) -> dict:
    url = f"{_APOLLO_BASE}{path}"
    async with session.post(
        url,
        json=payload,
        timeout=aiohttp.ClientTimeout(total=_TIMEOUT),
    ) as resp:
        if resp.status == 429:
            logger.warning("apollo_tool: rate-limited (429) on %s", path)
            return {}
        if resp.status == 401:
            logger.warning("apollo_tool: invalid API key (401)")
            return {}
        resp.raise_for_status()
        return await resp.json()


# ---------------------------------------------------------------------------
# People Search — Layer A of people_finder cascade
# ---------------------------------------------------------------------------

@tool
async def search_apollo_people(
    company_name   : str,
    target_titles  : list[str],
    max_results    : int = 5,
) -> list[dict]:
    """
    Layer A — Apollo.io people search by company name and target job titles.
    Searches Apollo's 270M+ professional database without consuming export credits.
    Returns name, title, LinkedIn URL, and (when Apollo provides it) a verified email.
    Use this as the first layer before website team scraping or Google dorks.
    Returns [] if the API key is missing, rate-limited, or no matches found.
    """
    if not _api_key():
        logger.warning("apollo_tool: APOLLO_API_KEY not set — skipping.")
        return []

    payload = {
        "person_titles"     : target_titles,
        "q_organization_name": company_name,
        "page"              : 1,
        "per_page"          : min(max_results * 2, 25),  # fetch extra to allow dedup
    }

    try:
        async with aiohttp.ClientSession(headers=_headers()) as session:
            data = await _post(session, "/mixed_people/search", payload)
    except Exception as e:
        logger.warning("apollo_tool: people search failed for '%s' — %s", company_name, e)
        return []

    raw_people = data.get("people") or []
    seen: set[str] = set()
    results: list[dict] = []

    for p in raw_people:
        if len(results) >= max_results:
            break

        name = (p.get("name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)

        # Apollo returns email as None when locked (free tier search)
        # Include it only when Apollo marks it as verified
        email      = p.get("email")
        email_status = p.get("email_status", "")
        verified_email = email if (email and email_status == "verified") else None

        org = p.get("organization") or {}
        results.append({
            "name"        : name,
            "title"       : p.get("title") or "",
            "company"     : org.get("name") or company_name,
            "linkedin_url": p.get("linkedin_url") or "",
            "email"       : verified_email,
            "phone"       : None,
            "source"      : "apollo",
            "title_score" : 0.0,   # scored by people_finder after return
            "title_tier"  : 1,     # default; people_finder sets the real value
        })

    logger.info(
        "apollo_tool: %d result(s) for '%s' (titles=%s)",
        len(results), company_name, target_titles[:2],
    )
    return results


# ---------------------------------------------------------------------------
# Company Search — supplements company_search.py for named-company lookups
# ---------------------------------------------------------------------------

@tool
async def search_apollo_company(
    company_name: str,
) -> dict:
    """
    Looks up a named company on Apollo to get its website domain, industry,
    and employee count. Used by company_search when the query names a company
    directly so we can skip the full SearXNG discovery phase.
    Returns a partial CompanyData dict (empty dict on failure).
    """
    if not _api_key():
        return {}

    payload = {
        "q_organization_name": company_name,
        "page"               : 1,
        "per_page"           : 1,
    }

    try:
        async with aiohttp.ClientSession(headers=_headers()) as session:
            data = await _post(session, "/mixed_companies/search", payload)
    except Exception as e:
        logger.warning("apollo_tool: company search failed for '%s' — %s", company_name, e)
        return {}

    orgs = data.get("organizations") or []
    if not orgs:
        return {}

    org = orgs[0]
    domain   = org.get("primary_domain") or ""
    website  = f"https://{domain}" if domain and not domain.startswith("http") else domain
    industry = org.get("industry") or ""
    size     = str(org.get("estimated_num_employees") or "")

    logger.info("apollo_tool: company found — %s @ %s", company_name, website)
    return {
        "name"      : org.get("name") or company_name,
        "website"   : website,
        "industry"  : industry,
        "revenue"   : org.get("annual_revenue_printed") or "",
        "confidence": 0.85,
        "tech_stack": [],
        "source"    : "apollo",
        "size"      : size,
    }
