import logging
import re

import aiohttp
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

SEARXNG_URL = "http://localhost:8080/search"

# Email permutation patterns — tried in order by Contact Enricher
_PATTERNS = [
    "{first}.{last}",
    "{first}{last}",
    "{f}{last}",
    "{first}_{last}",
    "{first}",
    "{f}.{last}",
]

# LinkedIn result titles: "Name - Title at Company | LinkedIn"
# Handles accented chars (María), hyphenated names (Mary-Jane), initials (John A. Smith)
_NAME_RE = re.compile(
    r"^([\w][\w'\-\.]*(?:\s+[\w][\w'\-\.]*)+)\s*[-–|]",
    re.UNICODE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _permute_emails(first: str, last: str, domain: str) -> list[str]:
    """Generates common email address patterns for a given name and domain."""
    f = first[0].lower() if first else ""
    subs = {
        "first": first.lower(),
        "last" : last.lower(),
        "f"    : f,
    }
    seen, emails = set(), []
    for pattern in _PATTERNS:
        try:
            addr = f"{pattern.format(**subs)}@{domain}"
            if addr not in seen:
                seen.add(addr)
                emails.append(addr)
        except KeyError:
            continue
    return emails


def _extract_name(title: str) -> tuple[str, str] | tuple[None, None]:
    """
    Pulls first + last name from a LinkedIn search result title string.
    Returns (first, last) or (None, None) if no match.
    """
    match = _NAME_RE.match(title.strip())
    if not match:
        return None, None
    parts = match.group(1).strip().split()
    if len(parts) < 2:
        return None, None
    return parts[0], parts[-1]


async def _dork_search(company_name: str, title: str, max_results: int) -> list[dict]:
    """
    Queries SearXNG with a LinkedIn site: dork and returns raw result dicts.
    Query format: site:linkedin.com/in "title" "company"
    """
    query = f'site:linkedin.com/in "{title}" "{company_name}"'
    params = {
        "q"         : query,
        "format"    : "json",
        "categories": "general",
        "language"  : "en",
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                SEARXNG_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as response:
                response.raise_for_status()
                data = await response.json()
        return data.get("results", [])[:max_results]
    except Exception as e:
        logger.warning("Crosslinked dork search failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@tool
async def search_crosslinked_people(
    company_name: str,
    company_domain: str,
    target_titles: list[str],
    max_results: int = 5,
) -> list[dict]:
    """
    Layer 4 LinkedIn enumeration using Google-dork-style SearXNG queries — no LinkedIn login required.
    Searches site:linkedin.com/in for people matching target titles at the company.
    Extracts names from search result titles and generates email permutation candidates.
    Use this when Layers 1, 2, and 3 all return empty results.
    Returns PersonData-shaped dicts with source='crosslinked' and email set to the
    most likely permutation (unverified — Contact Enricher validates these).
    Returns [] if SearXNG is unreachable or no names can be extracted.
    """
    seen_names: set[str] = set()
    people: list[dict] = []

    for title in target_titles:
        if len(people) >= max_results:
            break

        raw_results = await _dork_search(company_name, title, max_results * 2)

        for result in raw_results:
            if len(people) >= max_results:
                break

            result_title = result.get("title", "")
            linkedin_url = result.get("url", "")

            first, last = _extract_name(result_title)
            if not first or not last:
                continue

            full_name = f"{first} {last}"
            if full_name in seen_names:
                continue
            seen_names.add(full_name)

            emails = _permute_emails(first, last, company_domain)
            # Surface the most likely candidate; Contact Enricher validates the list
            primary_email = emails[0] if emails else None

            people.append({
                "name"        : full_name,
                "title"       : title,   # known from the dork query — not scraped from profile
                "title_score" : 0.0,
                "company"     : company_name,
                "linkedin_url": linkedin_url if "linkedin.com/in/" in linkedin_url else "",
                "email"       : primary_email,
                "phone"       : None,
                "source"      : "crosslinked",
            })

        if people:
            logger.info(
                "Layer 4 found %d result(s) for '%s' @ '%s'",
                len(people), title, company_name,
            )
            break

    if not people:
        logger.info("Layer 4 found nothing for '%s' — all layers exhausted.", company_name)

    return people


@tool
async def get_email_permutations(
    first_name: str,
    last_name: str,
    company_domain: str,
) -> list[str]:
    """
    Generates all common email address patterns for a given name and company domain.
    Use this when you have a confirmed name but no email — returns candidates for
    the Contact Enricher to validate via Hunter or SMTP check.
    Example: ("john", "doe", "acme.com") → ["john.doe@acme.com", "johndoe@acme.com", ...]
    """
    return _permute_emails(first_name, last_name, company_domain)
