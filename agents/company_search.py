import asyncio
import logging
import os

from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import GraphState
from tools.searxng_tool import searxng_search
from tools.crawl4ai_tool import scrape_company_website

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Two Groq LLMs:
#   heavy (70b) — search query generation (needs creativity + accuracy)
#   light (8b)  — industry inference (high-volume, simple classification)
# ---------------------------------------------------------------------------

_llm_heavy = None
_llm_light = None


def _get_heavy_llm():
    global _llm_heavy
    if _llm_heavy is None:
        from langchain_groq import ChatGroq
        _llm_heavy = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )
    return _llm_heavy


def _get_llm():
    global _llm_light
    if _llm_light is None:
        from langchain_groq import ChatGroq
        _llm_light = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )
    return _llm_light


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SEARCH_QUERY_PROMPT = """You are a B2B lead generation researcher.
Generate 3 SearXNG search queries to find REAL COMPANY WEBSITES matching the criteria below.

Rules:
- Each query must find an actual company's own website (product page, about page, homepage)
- Do NOT generate queries that return blog posts, news articles, Wikipedia, lists, or dictionaries
- Include the exact industry keywords from the criteria in every query
- Add "software" or "company" or "vendor" to industry terms to find B2B company sites
- If a location (country, city, region) is in the criteria, include it in EVERY query —
  location is a hard filter, not optional
- Prefer queries that surface company homepages, not article/resource pages

Return ONLY a JSON array of exactly 3 query strings. No explanation, no markdown.

Original user query: {original_query}
Criteria:
{criteria}"""

_INDUSTRY_PROMPT = """You are a strict B2B analyst. Evaluate this web page and return JSON.

STEP 1 — Is this a real company website?
A real company website promotes its OWN products or services to customers.
IMMEDIATELY set matches=false for ANY of these:
- Wikipedia, dictionary, encyclopedia pages
- News articles or press releases (Times of India, Economic Times, TechCrunch, etc.)
- Tutorial or educational pages ("What is X?", "Understanding X", "Guide to X")
- Government authority pages (unless the entity sells commercial products)
- Startup/company directories, aggregator sites (Crunchbase, Tracxn, AngelList)
- List articles ("Top 10...", "Best X companies...")
- Generic resource or blog pages within a company website
  (URL path hints: /blog/, /resources/, /guides/, /learn/, /insights/, /news/)
- Freelancer or job marketplaces

STEP 2 — If it IS a real company website, does its industry relate to the target?
Check only industry relevance. Do NOT check location or size.

Target criteria: {criteria}
Website content (first 2000 chars):
{raw_text}

Return ONLY this JSON (no explanation):
{{"industry": "1-3 word industry label", "matches": true/false, "reason": "one sentence"}}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Generic words that make useless company names when extracted from a domain
_GENERIC_DOMAIN_WORDS = {
    "fintech", "saas", "software", "tech", "technology", "digital", "cloud",
    "solutions", "services", "company", "enterprise", "business", "startup",
    "news", "blog", "media", "info", "portal", "online", "web", "app",
    "product", "industry", "industrial", "market", "global", "group",
}

# Common subdomain prefixes that indicate a deep subdomain URL, not the company homepage
_SUBDOMAIN_PREFIXES = {
    "assets", "api", "cdn", "static", "img", "images", "files", "media",
    "docs", "help", "support", "status", "blog", "shop", "store",
    "mail", "ftp", "dev", "staging", "test", "sandbox", "demo", "app",
    "portal", "partnerfinder", "partners", "community", "forum", "careers",
    "jobs", "login", "auth", "account", "secure", "pay", "checkout",
    "m", "mobile", "amp", "wap", "www2", "go", "news", "resource", "resources",
}


def _clean_company_name(domain: str, fallback_title: str) -> str:
    """
    Derives a clean company name from the domain (e.g. 'prolim.com' → 'PROLIM').
    Rejects generic-word domains; falls back to trimming the SearXNG page title.
    """
    import re as _re
    bare = domain.replace("www.", "").split(".")[0].lower()
    if len(bare) >= 2 and bare not in _GENERIC_DOMAIN_WORDS:
        return bare.upper()
    # domain is a generic word or too short — clean the title instead
    cleaned = _re.split(r'[|\-–—:]', fallback_title)[0].strip()
    return cleaned or fallback_title


def _build_criteria_str(company_filters: dict) -> str:
    parts = []
    if company_filters.get("industry"):
        parts.append(f"Industry: {company_filters['industry']}")
    if company_filters.get("keywords"):
        parts.append(f"Keywords: {', '.join(company_filters['keywords'])}")
    if company_filters.get("revenue_min"):
        parts.append(f"Minimum revenue: {company_filters['revenue_min']}")
    if company_filters.get("location"):
        parts.append(f"Location: {company_filters['location']}")
    if company_filters.get("company_size"):
        parts.append(f"Company size: {company_filters['company_size']}")
    if company_filters.get("tech_stack"):
        parts.append(f"Tech stack: {', '.join(company_filters['tech_stack'])}")
    return "\n".join(parts) or "No specific filters — find relevant companies."


async def _generate_search_queries(criteria_str: str, original_query: str = "") -> list[str]:
    """Asks the LLM to produce 3 SearXNG queries for the given criteria."""
    import json, re
    prompt = _SEARCH_QUERY_PROMPT.format(criteria=criteria_str, original_query=original_query)
    try:
        response = await _get_heavy_llm().ainvoke([HumanMessage(content=prompt)])
        match = re.search(r'\[.*?\]', response.content, re.DOTALL)
        if match:
            queries = json.loads(match.group())
            logger.info("company_search: generated queries: %s", queries)
            return queries
    except Exception as e:
        logger.warning("Query generation failed: %s", e)
    return []


async def _infer_industry(raw_text: str, criteria_str: str) -> tuple[str, bool]:
    """Uses Cerebras to infer industry and check if company matches criteria."""
    import json, re
    prompt = _INDUSTRY_PROMPT.format(criteria=criteria_str, raw_text=raw_text[:2000])
    try:
        response = await _get_llm().ainvoke([
            SystemMessage(content="You are a precise B2B analyst. Return only valid JSON."),
            HumanMessage(content=prompt),
        ])
        match = re.search(r'\{.*?\}', response.content, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return data.get("industry", ""), data.get("matches", True)
    except Exception as e:
        logger.warning("Industry inference failed: %s", e)
    return "", False   # default: exclude on LLM failure — garbage-in protection


async def _search_and_scrape(query: str, criteria_str: str, seen_domains: set) -> list[dict]:
    """Runs one SearXNG query and scrapes the top unique company URLs."""
    results = await searxng_search.ainvoke({"keywords": query.split(), "max_results": 8})
    companies = []

    for r in results:
        if "error" in r:
            continue
        url = r.get("url", "")
        # Skip duplicates and non-company pages
        domain = url.split("/")[2] if url.startswith("http") else ""
        if not domain or domain in seen_domains:
            continue
        if any(skip in domain for skip in [
            "linkedin.com", "crunchbase.com", "wikipedia.org", "youtube.com",
            "dictionary.com", "merriam-webster.com", "investopedia.com", "shopify.com",
            "medium.com", "substack.com", "reddit.com", "quora.com", "cgaa.org",
            "timesofindia.com", "economictimes.com", "livemint.com", "businessinsider.com",
            "techcrunch.com", "venturebeat.com", "forbes.com", "businesswire.com",
            "prnewswire.com", "globenewswire.com", "startupindia.gov.in", "tracxn.com",
            "fintech.com", "thebalance.com", "nerdwallet.com",
        ]):
            continue
        # Skip deep subdomain URLs — they are asset/portal/partner pages, not company homepages.
        # e.g. assets.new.siemens.com (4 parts) or partnerfinder.plm.automation.siemens.com (5 parts)
        bare_host = domain[4:] if domain.startswith("www.") else domain
        host_parts = bare_host.split(".")
        if len(host_parts) > 3:
            logger.debug("company_search: skipping deep-subdomain URL %s", domain)
            continue
        if len(host_parts) == 3 and host_parts[0] in _SUBDOMAIN_PREFIXES:
            logger.debug("company_search: skipping subdomain prefix URL %s", domain)
            continue
        # Skip deep resource/blog pages — company homepages rarely have 3+ path segments
        url_path = url.split("/", 3)[3] if url.count("/") >= 3 else ""
        path_depth = len([p for p in url_path.split("/") if p])
        if path_depth >= 3:
            logger.debug("company_search: skipping deep URL path %s", url)
            continue
        seen_domains.add(domain)

        clean_name = _clean_company_name(domain, r.get("title", domain))
        scraped = await scrape_company_website.ainvoke({"url": url, "company_name": clean_name})
        if scraped.get("confidence", 0) == 0.0:
            continue

        raw_text = scraped.pop("raw_text", "")
        industry, matches = await _infer_industry(raw_text, criteria_str)
        if not matches:
            continue

        scraped["name"] = clean_name   # overwrite crawl4ai's copy to keep it consistent
        scraped["industry"] = industry
        scraped.pop("employee_count", None)  # not in CompanyData TypedDict
        scraped.pop("error", None)
        companies.append(scraped)

    return companies


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

async def company_search(state: GraphState) -> dict:
    """
    Company Search Agent — reads query_plan.company_filters, finds matching companies
    via SearXNG + Crawl4AI, infers industry with Cerebras, and writes list[CompanyData]
    to GraphState. Deduplicates by domain. Never raises — writes to errors on failure.
    """
    errors = list(state.get("errors", []))
    original_query = state.get("query", "")
    query_plan = state.get("query_plan", {})
    company_filters = query_plan.get("company_filters") or {}
    criteria_str = _build_criteria_str(company_filters)

    try:
        queries = await _generate_search_queries(criteria_str, original_query=original_query)
    except Exception as e:
        errors.append(f"company_search: query generation failed — {e}")
        return {"companies": [], "errors": errors}

    if not queries:
        errors.append("company_search: LLM returned no search queries.")
        return {"companies": [], "errors": errors}

    seen_domains: set[str] = set()
    all_companies: list[dict] = []

    # Run queries concurrently — each independently searches + scrapes
    tasks = [_search_and_scrape(q, criteria_str, seen_domains) for q in queries]
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        errors.append(f"company_search: gather failed — {e}")
        results = []

    for batch in results:
        if isinstance(batch, Exception):
            errors.append(f"company_search: one query batch failed — {batch}")
            continue
        all_companies.extend(batch)

    # Sort by confidence descending, cap at 10 companies
    all_companies.sort(key=lambda c: c.get("confidence", 0.0), reverse=True)
    all_companies = all_companies[:10]

    logger.info("company_search: found %d companies.", len(all_companies))

    return {
        "companies": all_companies,
        "errors"   : errors,
        "status"   : "companies_found" if all_companies else "no_companies_found",
    }
