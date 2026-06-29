"""
Kompass India executive contact tool — Layer B of the people-finder cascade.

Kompass (in.kompass.com) indexes B2B companies worldwide and exposes executive
names, titles, and sometimes email addresses on public company profile pages.
For India-listed companies this covers manufacturing, healthcare, logistics,
and many other sectors that Apollo.io misses.

Flow:
  1. Search Kompass for the company by name → get the company profile URL.
  2. Fetch the profile page → extract executives (name, title, email).
  3. Return PersonData-shaped dicts. No login / no paywall hit needed for names.
"""

import asyncio
import logging
import re

import httpx
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_TIMEOUT = 15.0
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

_BASE = "https://in.kompass.com"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

async def _get(url: str, params: dict | None = None) -> str:
    """Fetch a Kompass page. Returns HTML or '' on any error."""
    try:
        async with httpx.AsyncClient(
            headers=_HEADERS,
            follow_redirects=True,
            timeout=_TIMEOUT,
        ) as client:
            resp = await client.get(url, params=params)
            if resp.status_code == 200:
                return resp.text
            logger.debug("kompass_tool: %s → HTTP %d", url, resp.status_code)
    except Exception as exc:
        logger.debug("kompass_tool: fetch failed %s — %s", url, exc)
    return ""


# ---------------------------------------------------------------------------
# Step 1 — find the company's Kompass profile URL
# ---------------------------------------------------------------------------

async def _find_profile_url(company_name: str) -> str | None:
    """
    Searches Kompass for the company and returns the first matching profile URL.
    Returns None when the company is not found.
    """
    # Kompass company search endpoint
    search_url = f"{_BASE}/searchresult/company/{_url_slug(company_name)}/"
    html = await _get(search_url)

    if not html:
        # Try free-text search as a fallback
        html = await _get(f"{_BASE}/searchresult/", params={"q": company_name})

    if not html:
        return None

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        # The first result link to a company profile — /c/{slug}/ pattern
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if re.search(r"/c/[a-z0-9\-]+/", href):
                full = href if href.startswith("http") else _BASE + href
                logger.debug("kompass_tool: profile URL found — %s", full)
                return full
    except ImportError:
        # Regex fallback
        m = re.search(r'href=["\']([^"\']*?/c/[a-z0-9\-]+/[^"\']*?)["\']', html)
        if m:
            href = m.group(1)
            return href if href.startswith("http") else _BASE + href

    return None


def _url_slug(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s-]+", "-", text)
    return text.strip("-")


# ---------------------------------------------------------------------------
# Step 2 — extract executives from the company profile page
# ---------------------------------------------------------------------------

def _extract_executives(html: str, company_name: str) -> list[dict]:
    """
    Parses a Kompass company profile page and returns a list of executive dicts.
    Extracts: name, title, email (when visible), phone.
    """
    people: list[dict] = []
    seen: set[str] = set()

    def _add(name: str, title: str = "", email: str = "", phone: str = "") -> None:
        name = name.strip()
        if not name or len(name) < 4 or name in seen:
            return
        # Filter out nav/UI text fragments
        if name.lower() in {"contact us", "view more", "see all", "show all"}:
            return
        seen.add(name)
        people.append({
            "name"        : name,
            "title"       : title.strip(),
            "title_score" : 0.0,   # scored by people_finder
            "company"     : company_name,
            "email"       : email.strip() or None,
            "phone"       : phone.strip() or None,
            "linkedin_url": "",
            "source"      : "kompass",
        })

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        # Strategy 1 — Kompass executive/contact section
        # Kompass wraps executive cards in containers with class variations
        exec_containers = soup.select(
            ".executive-item, .contact-item, .person-item, "
            ".executive-card, .team-member, [class*='executive'], [class*='contact-person']"
        )
        for container in exec_containers:
            name_tag = container.select_one(
                ".executive-name, .contact-name, .person-name, strong, b, h3, h4"
            )
            name = name_tag.get_text(strip=True) if name_tag else ""
            title_tag = container.select_one(
                ".executive-function, .executive-title, .contact-title, .function, em, i, span.title"
            )
            title = title_tag.get_text(strip=True) if title_tag else ""
            # Email may be in a mailto link or obfuscated span
            email = ""
            email_tag = container.select_one("a[href^='mailto:']")
            if email_tag:
                email = email_tag["href"].replace("mailto:", "").strip()
            # Some Kompass pages encode email in data attributes
            if not email:
                em_span = container.select_one("[data-email], [data-mail]")
                if em_span:
                    email = (em_span.get("data-email") or em_span.get("data-mail") or "").strip()
            phone_tag = container.select_one("a[href^='tel:'], .phone, .telephone")
            phone = ""
            if phone_tag:
                phone = phone_tag.get_text(strip=True)
            if name:
                _add(name, title, email, phone)

        # Strategy 2 — JSON-LD Person schema (structured data in script tags)
        if not people:
            import json
            for script in soup.find_all("script", {"type": "application/ld+json"}):
                raw = script.string or ""
                try:
                    data = json.loads(raw)
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        if item.get("@type") in ("Person", "ContactPoint"):
                            name  = item.get("name", "")
                            title = item.get("jobTitle", "")
                            email = item.get("email", "").replace("mailto:", "")
                            phone = item.get("telephone", "")
                            if name:
                                _add(name, title, email, phone)
                except (json.JSONDecodeError, AttributeError):
                    pass

        # Strategy 3 — regex over raw HTML for email + nearby name
        if not people:
            # Find email addresses and try to attribute a name from context
            for m in re.finditer(
                r"([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})",
                html,
            ):
                email = m.group(1)
                if "kompass" in email or "example" in email:
                    continue
                # Look for a name within 200 chars before the email
                context = html[max(0, m.start() - 200) : m.start()]
                nm = re.search(r"([A-Z][a-z]+ [A-Z][a-z]+)\s*(?:<[^>]+>|\s)*$", context)
                name = nm.group(1) if nm else email.split("@")[0].replace(".", " ").title()
                _add(name, "", email)

    except ImportError:
        # No BeautifulSoup — regex only
        for m in re.finditer(
            r"([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})", html
        ):
            email = m.group(1)
            if "kompass" not in email and "example" not in email:
                name = email.split("@")[0].replace(".", " ").title()
                _add(name, "", email)

    return people


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@tool
async def search_kompass_executives(
    company_name: str,
    target_titles: list[str],
    max_results: int = 5,
) -> list[dict]:
    """
    Finds executive contacts for an Indian company on Kompass (in.kompass.com).
    Good coverage for manufacturing, healthcare, logistics, and B2B companies
    that may not appear in Apollo.

    Returns PersonData-shaped dicts:
      [{"name": str, "title": str, "email": str|None, "phone": str|None,
        "company": str, "source": "kompass", "title_score": 0.0}, ...]

    title_score is left at 0.0 — people_finder.py scores it against target_role.
    Returns [] when the company has no Kompass profile or no executives are listed.

    Use as Layer B (after Apollo) in the people-finder cascade.
    target_titles is accepted but not used for filtering here — all executives
    from the profile are returned and scored upstream by people_finder.
    """
    if not company_name.strip():
        logger.warning("kompass_tool: company_name is empty — skipping.")
        return []

    profile_url = await _find_profile_url(company_name)
    if not profile_url:
        logger.info("kompass_tool: no profile found for '%s'", company_name)
        return []

    logger.info("kompass_tool: fetching profile %s", profile_url)
    html = await _get(profile_url)
    if not html:
        logger.info("kompass_tool: profile page empty for '%s'", company_name)
        return []

    people = _extract_executives(html, company_name)
    if people:
        logger.info(
            "kompass_tool: found %d executive(s) for '%s'", len(people), company_name
        )
    else:
        logger.info("kompass_tool: no executives extracted for '%s'", company_name)

    return people[:max_results]
