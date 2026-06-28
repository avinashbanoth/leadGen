import logging
import re

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Confidence weights — how much we trust each data source
_CONFIDENCE = {
    "about_page"   : 0.85,
    "homepage"     : 0.65,
    "linkedin_page": 0.80,
    "crunchbase"   : 0.75,
    "fallback"     : 0.40,
}

# Revenue strings LinkedIn / Crunchbase commonly use
_REVENUE_RE = re.compile(
    r'\$[\d,.]+\s*(?:million|billion|M|B|K)?|\b\d+\s*(?:million|billion)\b',
    re.IGNORECASE,
)

# Employee count patterns: "500-1000 employees", "~200 employees"
_EMP_RE = re.compile(r'(\d[\d,]*)\s*[-–]\s*(\d[\d,]*)\s*employees?|~?(\d[\d,]+)\s*employees?', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_source_confidence(url: str) -> float:
    """Maps a URL to a confidence weight based on how reliable the source is."""
    if "/about" in url or "about." in url:
        return _CONFIDENCE["about_page"]
    if "linkedin.com/company" in url:
        return _CONFIDENCE["linkedin_page"]
    if "crunchbase.com" in url:
        return _CONFIDENCE["crunchbase"]
    return _CONFIDENCE["homepage"]


def _extract_revenue(text: str) -> str | None:
    match = _REVENUE_RE.search(text)
    return match.group(0).strip() if match else None


def _extract_employees(text: str) -> str | None:
    match = _EMP_RE.search(text)
    if not match:
        return None
    if match.group(1) and match.group(2):
        return f"{match.group(1)}-{match.group(2)}"
    return match.group(3)


def _extract_tech_stack(text: str) -> list[str]:
    """
    Looks for common SaaS / tech stack keywords mentioned on the page.
    Returns a list of detected technology names.
    """
    known_tech = [
        "React", "Angular", "Vue", "Next.js", "Node.js", "Django", "Rails",
        "Kubernetes", "Docker", "AWS", "GCP", "Azure", "Salesforce", "HubSpot",
        "Stripe", "Twilio", "Segment", "Snowflake", "dbt", "Kafka", "Redis",
        "PostgreSQL", "MongoDB", "Elasticsearch", "GraphQL", "REST",
    ]
    found = []
    text_lower = text.lower()
    for tech in known_tech:
        if tech.lower() in text_lower:
            found.append(tech)
    return found


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
async def scrape_company_website(url: str, company_name: str) -> dict:
    """
    Scrapes a company website using Crawl4AI and extracts structured company data.
    Returns a CompanyData-shaped dict with a confidence score (0.0–1.0) that reflects
    how much to trust the extracted fields — never asserts scraped revenue as fact.
    On failure returns a partial dict with confidence=0.0 and an error key.
    """
    try:
        from crawl4ai import AsyncWebCrawler
        from crawl4ai.extraction_strategy import NoExtractionStrategy
    except ImportError:
        logger.error("crawl4ai not installed — run: pip install crawl4ai")
        return {
            "name"      : company_name,
            "website"   : url,
            "industry"  : "",
            "revenue"   : "",
            "confidence": 0.0,
            "tech_stack": [],
            "source"    : "crawl4ai",
            "error"     : "crawl4ai not installed",
        }

    confidence = _detect_source_confidence(url)

    try:
        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(url=url)
    except Exception as e:
        logger.warning("crawl4ai failed for %s: %s", url, e)
        return {
            "name"      : company_name,
            "website"   : url,
            "industry"  : "",
            "revenue"   : "",
            "confidence": 0.0,
            "tech_stack": [],
            "source"    : "crawl4ai",
            "error"     : str(e),
        }

    # result.markdown is the page content converted to clean markdown
    text = result.markdown or result.extracted_content or ""

    revenue   = _extract_revenue(text)
    employees = _extract_employees(text)
    tech      = _extract_tech_stack(text)

    # Revenue is never asserted as fact — confidence communicates uncertainty
    if not revenue:
        confidence = max(0.0, confidence - 0.15)

    return {
        "name"         : company_name,
        "website"      : url,
        "industry"     : "",        # filled by Company Search Agent using LLM reasoning
        "revenue"      : revenue or "",
        "employee_count": employees or "",
        "confidence"   : round(confidence, 2),
        "tech_stack"   : tech,
        "source"       : "crawl4ai",
        "raw_text"     : text[:3000],   # first 3k chars passed to LLM for industry inference
    }


@tool
async def scrape_multiple_urls(urls: list[str], company_name: str) -> list[dict]:
    """
    Scrapes multiple URLs for a single company and returns the highest-confidence result.
    Useful when you have both a homepage and an /about page — tries all and picks the best.
    Returns a list sorted by confidence descending so the caller can pick the top result.
    """
    import asyncio

    tasks = [scrape_company_website.ainvoke({"url": u, "company_name": company_name}) for u in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    valid = []
    for r in results:
        if isinstance(r, Exception):
            logger.warning("scrape_multiple_urls: one URL failed — %s", r)
            continue
        if isinstance(r, dict) and "error" not in r:
            valid.append(r)

    valid.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
    return valid
