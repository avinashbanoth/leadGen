import asyncio
import logging
import re

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_CRAWL_TIMEOUT = 30.0   # JustDial pages are JS-heavy; give them extra time
_MAX_TEXT = 8000        # chars of rendered text to parse


# ---------------------------------------------------------------------------
# URL builders
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    text = text.strip().title()
    text = re.sub(r"[^a-zA-Z0-9\s-]", "", text)
    text = re.sub(r"[\s-]+", "-", text)
    return text.strip("-")


def _candidate_urls(city: str, category: str) -> list[str]:
    """
    JustDial directory URL variants in order of preference.
    The plain /{City}/{Category} pattern works for most cities.
    The /nsh- variant disables "near me" fallback — better for tier-2 searches.
    """
    city_s    = _slugify(city)
    cat_s     = _slugify(category)
    cat_comp  = f"{cat_s}-Companies" if "compan" not in cat_s.lower() else cat_s
    base      = "https://www.justdial.com"
    return [
        f"{base}/{city_s}/{cat_s}",
        f"{base}/{city_s}/{cat_comp}",
        f"{base}/{city_s}/{cat_s}/nsh-{city_s}",
    ]


# ---------------------------------------------------------------------------
# HTML parser — runs on rendered page content from Crawl4AI
# ---------------------------------------------------------------------------

def _extract_businesses(html_or_text: str, max_results: int) -> list[dict]:
    """
    Extracts business names (and phone numbers when visible) from JustDial
    rendered HTML. Tries BeautifulSoup CSS selectors first, falls back to
    regex on the raw text.
    """
    names: list[str] = []
    phones: dict[str, str] = {}   # name → phone
    seen: set[str] = set()

    def _add(name: str, phone: str = "") -> None:
        name = name.strip()
        # Drop entries that look like UI labels rather than business names
        if (name and name not in seen
                and 3 <= len(name) <= 120
                and not name.lower().startswith(("search", "home", "top ", "best "))):
            names.append(name)
            seen.add(name)
            if phone:
                phones[name] = phone

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_or_text, "html.parser")

        # Strategy 1 — known JustDial listing element selectors
        # JustDial has changed class names over the years; try several
        for selector in (
            "span.lng",               # primary business-name span (most versions)
            ".store-name",
            "p.store-name",
            "span.jcn",               # JustDial Company Name
            "a.store-name",
            "h2.store-name",
            ".companyname",
            "span[class*='company']",
            "div[class*='title'] span",
        ):
            for tag in soup.select(selector):
                text = tag.get_text(strip=True)
                if text:
                    # Try to find adjacent phone in the same listing container
                    parent = tag.find_parent(["li", "div", "article"])
                    phone = ""
                    if parent:
                        ph_tag = parent.select_one(
                            "a[href^='tel:'], span[class*='phone'], .jd-phone, .contact-btn"
                        )
                        if ph_tag:
                            phone = ph_tag.get_text(strip=True).replace(" ", "")
                    _add(text, phone)
                if len(names) >= max_results:
                    break
            if len(names) >= max_results:
                break

        # Strategy 2 — JSON-LD structured data (most reliable when present)
        if not names:
            for script in soup.find_all("script", {"type": "application/ld+json"}):
                raw = script.string or ""
                for m in re.finditer(r'"name"\s*:\s*"([^"]{3,120})"', raw):
                    _add(m.group(1))
                if len(names) >= max_results:
                    break

    except ImportError:
        pass

    # Strategy 3 — plain-text regex fallback (works on markdown/text output)
    if not names:
        # JustDial listings in text form often have business names followed by
        # ratings like "4.2★" or phone numbers
        for m in re.finditer(
            r'\n([A-Z][A-Za-z0-9&\-\' ]{2,80})\s*(?:\n|[\d★✩])',
            html_or_text,
        ):
            _add(m.group(1).strip())
            if len(names) >= max_results:
                break

    return [
        {"name": n, "phone": phones.get(n, ""), "source": "justdial"}
        for n in names[:max_results]
    ]


# ---------------------------------------------------------------------------
# Crawl4AI fetch — JS-rendered
# ---------------------------------------------------------------------------

async def _render_page(url: str) -> str:
    """
    Fetches and JS-renders the JustDial listing page using Crawl4AI.
    Returns the rendered HTML. Returns "" on timeout or error.
    """
    try:
        from crawl4ai import AsyncWebCrawler
    except ImportError:
        logger.error("justdial_tool: crawl4ai not installed.")
        return ""

    try:
        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await asyncio.wait_for(
                crawler.arun(url=url),
                timeout=_CRAWL_TIMEOUT,
            )
        # Prefer raw HTML (has class info for selectors); fall back to markdown
        return result.html or result.markdown or result.extracted_content or ""
    except asyncio.TimeoutError:
        logger.warning("justdial_tool: timed out on %s", url)
    except Exception as exc:
        logger.debug("justdial_tool: crawl error for %s — %s", url, exc)
    return ""


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@tool
async def search_justdial_businesses(
    city: str,
    category: str,
    max_results: int = 10,
) -> list[dict]:
    """
    Searches JustDial's business directory for companies in a specific Indian
    city and category. Uses Crawl4AI with Playwright to render JavaScript.
    Covers 250+ Indian cities including tier-2 cities like Warangal, Nashik, Coimbatore.

    Returns a list of dicts:
      [{"name": "Business Name", "phone": "+91-...", "source": "justdial"}, ...]
    Phone is included when visible on the listing page (may be masked by JustDial).
    Returns [] when city or category is blank, the page fails to load,
    or JustDial has no listings for that combination.

    Use this as Phase 2 of Indian company discovery after IndiaMart returns nothing.
    Results should be passed to the company verification step before GraphState.
    """
    if not city.strip() or not category.strip():
        logger.warning("justdial_tool: city or category is empty — skipping.")
        return []

    for url in _candidate_urls(city, category):
        logger.info("justdial_tool: trying %s", url)
        content = await _render_page(url)
        if not content:
            continue
        results = _extract_businesses(content, max_results)
        if results:
            logger.info(
                "justdial_tool: found %d businesses for city='%s' category='%s'",
                len(results), city, category,
            )
            return results

    logger.info(
        "justdial_tool: no results for city='%s' category='%s'",
        city, category,
    )
    return []
