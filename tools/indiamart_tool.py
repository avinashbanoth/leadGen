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
    "Connection": "keep-alive",
}


# ---------------------------------------------------------------------------
# URL builders
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s-]+", "-", text)
    return text.strip("-")


def _city_slug(city: str) -> str:
    # IndiaMart uses Title-Case city names: "Warangal", "Greater-Noida"
    return city.strip().title().replace(" ", "-")


def _candidate_urls(city: str, industry: str) -> list[str]:
    """
    Returns IndiaMart directory URL variants to try in order.
    Confirmed pattern: dir.indiamart.com/{City}/{category}.html
    Also tries common suffixes in case the plain slug returns a 404.
    """
    c = _city_slug(city)
    i = _slugify(industry)
    base = "https://dir.indiamart.com"
    return [
        f"{base}/{c}/{i}.html",
        f"{base}/{c}/{i}-companies.html",
        f"{base}/{c}/{i}-suppliers.html",
        f"{base}/{c}/{i}-products.html",
    ]


# ---------------------------------------------------------------------------
# HTML parser
# ---------------------------------------------------------------------------

def _extract_company_names(html: str, max_results: int) -> list[dict]:
    """
    Parses IndiaMart directory HTML and extracts company names.
    Tries BeautifulSoup CSS selectors first; falls back to regex on import error.
    """
    names: list[str] = []
    seen: set[str] = set()

    def _add(name: str) -> bool:
        name = name.strip()
        if name and name not in seen and 3 <= len(name) <= 120:
            names.append(name)
            seen.add(name)
            return True
        return False

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        # Primary selectors — IndiaMart uses several class names across page versions
        for selector in (
            ".company-name",
            ".list-company-name",
            ".bname",
            ".comp-name",
            ".org-name",
            "span.company-name",
            "a.company-name",
            "h3.company-info",
        ):
            for tag in soup.select(selector):
                _add(tag.get_text(strip=True))
                if len(names) >= max_results:
                    break
            if len(names) >= max_results:
                break

        # Fallback: anchor tags whose href points to an IndiaMART company subdomain
        # e.g. https://acme-corp.indiamart.com/
        if not names:
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if re.search(r"https?://[^/]+\.indiamart\.com", href) and \
                        "dir.indiamart.com" not in href:
                    _add(a.get_text(strip=True))
                    if len(names) >= max_results:
                        break

        # Last resort: JSON-LD schema "name" fields embedded in the page
        if not names:
            for m in re.finditer(r'"name"\s*:\s*"([^"]{3,120})"', html):
                _add(m.group(1))
                if len(names) >= max_results:
                    break

    except ImportError:
        # BeautifulSoup not installed — pure regex path
        for m in re.finditer(r'"name"\s*:\s*"([^"]{3,120})"', html):
            _add(m.group(1))
            if len(names) >= max_results:
                break

    return [{"name": n, "source": "indiamart"} for n in names]


# ---------------------------------------------------------------------------
# HTTP fetch
# ---------------------------------------------------------------------------

async def _fetch(url: str) -> str:
    """Returns page HTML or empty string on any error."""
    try:
        async with httpx.AsyncClient(
            headers=_HEADERS,
            follow_redirects=True,
            timeout=_TIMEOUT,
        ) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.text
            logger.debug("indiamart_tool: %s → HTTP %d", url, resp.status_code)
    except Exception as exc:
        logger.debug("indiamart_tool: fetch failed for %s — %s", url, exc)
    return ""


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@tool
async def search_indiamart_companies(
    city: str,
    industry: str,
    max_results: int = 10,
) -> list[dict]:
    """
    Searches IndiaMart's business directory for companies in a specific Indian city
    and industry sector. Uses static HTML scraping — no JavaScript, no login required.
    Works for both tier-1 cities (Hyderabad, Mumbai) and tier-2 cities (Warangal, Nashik).

    Returns a list of dicts: [{"name": "Company Name", "source": "indiamart"}, ...]
    Returns [] when city or industry is blank, the page returns no listings,
    or IndiaMart does not have a directory page for that city+industry combination.

    Use this as Phase 1 of company discovery for any Indian location query.
    Results should be passed to the company verification step (homepage finder)
    before being written to GraphState.
    """
    if not city.strip() or not industry.strip():
        logger.warning("indiamart_tool: city or industry is empty — skipping.")
        return []

    for url in _candidate_urls(city, industry):
        logger.info("indiamart_tool: trying %s", url)
        html = await _fetch(url)
        if not html:
            continue
        results = _extract_company_names(html, max_results)
        if results:
            logger.info(
                "indiamart_tool: found %d companies for city='%s' industry='%s'",
                len(results), city, industry,
            )
            return results

    logger.info(
        "indiamart_tool: no listings found for city='%s' industry='%s'",
        city, industry,
    )
    return []
