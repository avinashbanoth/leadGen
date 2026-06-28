import asyncio
import logging
import os

from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import GraphState
from tools.searxng_tool import searxng_search
from tools.crawl4ai_tool import scrape_company_website

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy Cerebras LLM — used for industry inference + relevance reasoning
# ---------------------------------------------------------------------------

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        from langchain_openai import ChatOpenAI
        _llm = ChatOpenAI(
            model="llama3.1-70b",
            base_url="https://api.cerebras.ai/v1",
            api_key=os.getenv("CEREBRAS_API_KEY"),
            temperature=0,
        )
    return _llm


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SEARCH_QUERY_PROMPT = """You are a B2B lead generation researcher.
Given the company search criteria below, generate 3 concise SearXNG search queries
that will find real company websites matching the criteria.
Return ONLY a JSON array of 3 query strings. No explanation.
Example: ["fintech startups UK 2024", "Series B SaaS company data analytics", ...]

Criteria:
{criteria}"""

_INDUSTRY_PROMPT = """You are a B2B analyst. Given this website content, infer:
1. The company's industry (1-3 words, e.g. "B2B SaaS", "Fintech", "Healthcare IT")
2. Whether this company matches the target criteria

Criteria: {criteria}
Website content (first 2000 chars):
{raw_text}

Return JSON: {{"industry": "...", "matches": true/false, "reason": "..."}}
No explanation outside the JSON."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


async def _generate_search_queries(criteria_str: str) -> list[str]:
    """Asks the LLM to produce 3 SearXNG queries for the given criteria."""
    import json, re
    prompt = _SEARCH_QUERY_PROMPT.format(criteria=criteria_str)
    try:
        response = await _get_llm().ainvoke([HumanMessage(content=prompt)])
        match = re.search(r'\[.*?\]', response.content, re.DOTALL)
        if match:
            return json.loads(match.group())
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
    return "", True   # default: include with empty industry


async def _search_and_scrape(query: str, criteria_str: str, seen_domains: set) -> list[dict]:
    """Runs one SearXNG query and scrapes the top unique company URLs."""
    results = await searxng_search.ainvoke({"keywords": query.split(), "max_results": 5})
    companies = []

    for r in results:
        if "error" in r:
            continue
        url = r.get("url", "")
        # Skip duplicates and non-company pages
        domain = url.split("/")[2] if url.startswith("http") else ""
        if not domain or domain in seen_domains:
            continue
        if any(skip in domain for skip in ["linkedin.com", "crunchbase.com", "wikipedia.org", "youtube.com"]):
            continue
        seen_domains.add(domain)

        scraped = await scrape_company_website.ainvoke({"url": url, "company_name": r.get("title", domain)})
        if scraped.get("confidence", 0) == 0.0:
            continue

        raw_text = scraped.pop("raw_text", "")
        industry, matches = await _infer_industry(raw_text, criteria_str)
        if not matches:
            continue

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
    query_plan = state.get("query_plan", {})
    company_filters = query_plan.get("company_filters") or {}
    criteria_str = _build_criteria_str(company_filters)

    try:
        queries = await _generate_search_queries(criteria_str)
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
