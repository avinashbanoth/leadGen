"""
Zaubacorp MCA director tool — Layer C of the people-finder cascade.

Zaubacorp (zaubacorp.com) aggregates Ministry of Corporate Affairs (MCA)
government data for all companies registered in India. Director names and
DINs (Director Identification Numbers) are official public records and are
almost always correct even when no email is available.

Flow:
  1. Search Zaubacorp for the company by name → get CIN + profile URL.
  2. Fetch the company profile → extract director names, DINs, and any
     available contact details.
  3. Return PersonData-shaped dicts (title="Director", email=None by default).
     Contact Enricher or permutator can find emails for these directors.

No authentication required — all data is publicly mandated by Indian law.
"""

import logging
import re

import httpx
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_TIMEOUT = 15.0
_BASE = "https://www.zaubacorp.com"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

async def _get(url: str, params: dict | None = None) -> str:
    try:
        async with httpx.AsyncClient(
            headers=_HEADERS,
            follow_redirects=True,
            timeout=_TIMEOUT,
        ) as client:
            resp = await client.get(url, params=params)
            if resp.status_code == 200:
                return resp.text
            logger.debug("zaubacorp_tool: %s → HTTP %d", url, resp.status_code)
    except Exception as exc:
        logger.debug("zaubacorp_tool: fetch failed %s — %s", url, exc)
    return ""


# ---------------------------------------------------------------------------
# Step 1 — find company profile URL via search
# ---------------------------------------------------------------------------

async def _find_profile_url(company_name: str) -> str | None:
    """
    Searches Zaubacorp for the company name and returns the first matching
    company profile URL (/company/{name}/{CIN} pattern).
    """
    html = await _get(f"{_BASE}/companysearch/", params={"search": company_name})
    if not html:
        return None

    # Profile URLs follow the pattern: /company/{Company-Name}/{CIN}
    # CIN format: [A-Z]{1}[0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6}
    profile_re = re.compile(
        r'href=["\'](/company/[^"\']+/[A-Z0-9]{21}[^"\']*)["\']',
        re.IGNORECASE,
    )

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if re.search(r"/company/[^/]+/[A-Z0-9]{15,25}", href, re.IGNORECASE):
                full = href if href.startswith("http") else _BASE + href
                logger.debug("zaubacorp_tool: profile URL — %s", full)
                return full
    except ImportError:
        m = profile_re.search(html)
        if m:
            href = m.group(1)
            return href if href.startswith("http") else _BASE + href

    return None


# ---------------------------------------------------------------------------
# Step 2 — extract directors from the company profile
# ---------------------------------------------------------------------------

def _extract_directors(html: str, company_name: str) -> list[dict]:
    """
    Parses a Zaubacorp company profile page and extracts director information.
    Returns PersonData-shaped dicts with title='Director' (or actual designation
    when available), email=None (MCA data rarely includes emails).
    """
    people: list[dict] = []
    seen: set[str] = set()

    def _add(name: str, designation: str = "Director", din: str = "") -> None:
        name = name.strip()
        # Filter out junk: short strings, UI labels, CIN/address fragments
        if (not name or len(name) < 5 or name in seen
                or re.match(r"^[0-9\-,/ ]+$", name)
                or name.lower() in {"view details", "show more", "contact us", "director"}):
            return
        seen.add(name)
        people.append({
            "name"        : name,
            "title"       : designation.strip() or "Director",
            "title_score" : 0.0,
            "company"     : company_name,
            "email"       : None,
            "phone"       : None,
            "linkedin_url": "",
            "din"         : din,       # Extra field: DIN for cross-referencing
            "source"      : "zaubacorp",
        })

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        # Strategy 1 — Zaubacorp director table
        # Directors are listed in a table under a "Directors" heading
        for table in soup.find_all("table"):
            headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
            if not any(h in headers for h in ("director", "din", "name", "designation")):
                continue
            # Identify column indices
            name_col = next(
                (i for i, h in enumerate(headers) if "name" in h or "director" in h), 0
            )
            desg_col = next(
                (i for i, h in enumerate(headers) if "designation" in h or "type" in h), -1
            )
            din_col = next(
                (i for i, h in enumerate(headers) if "din" in h), -1
            )
            for row in table.find_all("tr")[1:]:  # skip header row
                cells = row.find_all(["td", "th"])
                if not cells or len(cells) <= name_col:
                    continue
                name  = cells[name_col].get_text(strip=True)
                desg  = cells[desg_col].get_text(strip=True) if desg_col >= 0 and len(cells) > desg_col else "Director"
                din   = cells[din_col].get_text(strip=True) if din_col >= 0 and len(cells) > din_col else ""
                _add(name, desg, din)

        # Strategy 2 — named sections: look for "Director" heading followed by names
        if not people:
            for heading in soup.find_all(["h2", "h3", "h4", "strong", "b"]):
                if "director" not in heading.get_text(strip=True).lower():
                    continue
                # Walk siblings until next heading
                for sib in heading.find_next_siblings():
                    tag_name = getattr(sib, "name", None)
                    if tag_name in ("h2", "h3", "h4") and sib != heading:
                        break
                    if tag_name in ("table", "ul", "ol", "div"):
                        for item in sib.find_all(["td", "li", "p", "span"]):
                            text = item.get_text(strip=True)
                            # Director names typically 2+ words, Title-case
                            if re.match(r"[A-Z][a-z]+(?: [A-Z][a-z]+)+", text):
                                _add(text)

        # Strategy 3 — regex over raw HTML for DIN + adjacent name
        if not people:
            # DIN is always 8 digits; look for it and capture nearby name
            for m in re.finditer(r"\b([0-9]{8})\b", html):
                context_start = max(0, m.start() - 300)
                context       = html[context_start : m.start()]
                nm = re.search(
                    r"([A-Z][A-Z ]{2,}[A-Z])\s*(?:</[^>]+>|\s)*$", context
                )
                if nm:
                    name = nm.group(1).title()
                    _add(name, "Director", m.group(1))

    except ImportError:
        # No BeautifulSoup — regex only
        for m in re.finditer(
            r"([A-Z][A-Z ]{3,})\s*(?:[^A-Za-z]{0,20})\s*([0-9]{8})",
            html,
        ):
            _add(m.group(1).title(), "Director", m.group(2))

    return people


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@tool
async def search_zaubacorp_directors(
    company_name: str,
    target_titles: list[str],
    max_results: int = 5,
) -> list[dict]:
    """
    Retrieves official directors for an Indian registered company from Zaubacorp
    (MCA government registry data). Covers all companies incorporated in India —
    Private Limited, Public Limited, LLP — even those absent from Apollo or LinkedIn.

    Returns PersonData-shaped dicts:
      [{"name": str, "title": "Director"|designation, "email": None,
        "din": str, "company": str, "source": "zaubacorp", "title_score": 0.0}, ...]

    Email is None — Contact Enricher will generate permutation candidates.
    DIN (Director Identification Number) is included for cross-referencing.
    title_score is left at 0.0 — people_finder.py scores it against target_role.

    Use as Layer C (after Apollo and Kompass) in the people-finder cascade.
    target_titles is accepted but not used for filtering here — all directors
    from the MCA record are returned and scored upstream by people_finder.
    Returns [] when the company is not found in the MCA registry or the
    profile page has no director data.
    """
    if not company_name.strip():
        logger.warning("zaubacorp_tool: company_name is empty — skipping.")
        return []

    profile_url = await _find_profile_url(company_name)
    if not profile_url:
        logger.info("zaubacorp_tool: no MCA profile found for '%s'", company_name)
        return []

    logger.info("zaubacorp_tool: fetching profile %s", profile_url)
    html = await _get(profile_url)
    if not html:
        logger.info("zaubacorp_tool: profile page empty for '%s'", company_name)
        return []

    directors = _extract_directors(html, company_name)
    if directors:
        logger.info(
            "zaubacorp_tool: found %d director(s) for '%s'",
            len(directors), company_name,
        )
    else:
        logger.info("zaubacorp_tool: no directors extracted for '%s'", company_name)

    return directors[:max_results]
