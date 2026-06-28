import asyncio
import logging
import os
import re

from langchain_core.tools import tool

from utils.rate_limiter import rate_limiter
from utils.human_behavior import random_delay

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy client init — linkedin-api is synchronous; credentials loaded at runtime
# ---------------------------------------------------------------------------

_client = None


def _get_client():
    global _client
    if _client is None:
        from linkedin_api import Linkedin
        username = os.getenv("LI_USERNAME")
        password = os.getenv("LI_PASSWORD")
        if not username or not password:
            raise RuntimeError("LI_USERNAME and LI_PASSWORD must be set in .env")
        _client = Linkedin(username, password, authenticate=True)
        logger.info("linkedin-api client authenticated as %s", username)
    return _client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_public_id(linkedin_url: str) -> str | None:
    """Extracts the public profile slug from a LinkedIn URL."""
    match = re.search(r"linkedin\.com/in/([^/?#]+)", linkedin_url)
    return match.group(1) if match else None


def _parse_person(result: dict, source: str = "linkedin_api") -> dict:
    """Maps a raw linkedin-api search result to our PersonData shape."""
    name_parts = result.get("firstName", ""), result.get("lastName", "")
    name = " ".join(p for p in name_parts if p).strip()

    headline = result.get("headline", "")
    public_id = result.get("publicIdentifier", "")
    linkedin_url = f"https://www.linkedin.com/in/{public_id}" if public_id else ""

    return {
        "name"        : name,
        "title"       : headline,
        "title_score" : 0.0,        # scored by People Finder Agent after retrieval
        "company"     : result.get("company", ""),
        "linkedin_url": linkedin_url,
        "email"       : None,
        "phone"       : None,
        "source"      : source,
    }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
async def search_linkedin_people(
    company_name: str,
    target_titles: list[str],
    max_results: int = 5,
) -> list[dict]:
    """
    Searches LinkedIn for people at a specific company matching any of the given titles.
    Tries each title variant in order and returns as soon as results are found.
    Uses Voyager HTTP API — no browser required.
    Returns a list of PersonData-shaped dicts (title_score set to 0.0 for caller to fill).
    """
    client = _get_client()

    for title in target_titles:
        await rate_limiter.check_search()
        await random_delay(1000, 3000)

        try:
            raw_results = await asyncio.to_thread(
                client.search_people,
                keyword_company=company_name,
                keyword_title=title,
                limit=max_results,
            )
        except Exception as e:
            logger.warning("linkedin-api search failed (company=%s, title=%s): %s", company_name, title, e)
            continue

        if raw_results:
            logger.info(
                "Layer 1 found %d results for '%s' @ '%s'",
                len(raw_results), title, company_name,
            )
            return [_parse_person(r) for r in raw_results[:max_results]]

    logger.info("Layer 1 returned no results for '%s' — escalating to Layer 2.", company_name)
    return []


@tool
async def get_linkedin_contact_info(linkedin_url: str) -> dict:
    """
    Fetches contact information (email, phone, websites) for a LinkedIn profile.
    Accepts a full LinkedIn profile URL (e.g. https://www.linkedin.com/in/johndoe).
    Returns a dict with email, phone, and twitter fields where available.
    """
    public_id = _extract_public_id(linkedin_url)
    if not public_id:
        return {"error": f"Could not extract public ID from URL: {linkedin_url}"}

    await rate_limiter.check_profile()
    await random_delay(2000, 5000)

    client = _get_client()

    try:
        contact = await asyncio.to_thread(
            client.get_profile_contact_info,
            public_id,
        )
    except Exception as e:
        logger.warning("get_profile_contact_info failed (%s): %s", public_id, e)
        return {"error": str(e)}

    emails = contact.get("email_address") or ""
    phone_numbers = contact.get("phone_numbers") or []
    phone = phone_numbers[0].get("number") if phone_numbers else None

    return {
        "email"  : emails if emails else None,
        "phone"  : phone,
        "twitter": contact.get("twitter", [None])[0] if contact.get("twitter") else None,
    }
