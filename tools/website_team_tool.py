import asyncio
import json
import logging
import os
import re

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

_TEAM_PATHS = [
    "/team", "/about", "/leadership", "/management",
    "/about-us", "/our-team", "/people", "/company/team",
]
_CRAWL_TIMEOUT = 15.0
_MAX_TEXT_CHARS = 4000

_EXTRACT_PROMPT = """You are given raw text scraped from a company's team or about page.
Extract all people listed with their names and job titles.

Return a JSON array only. Each element must have exactly these fields:
  "name": full name (string)
  "title": job title (string)

Rules:
- Only include people who have BOTH a name and a title.
- Do not invent names or titles not present in the text.
- If no people are found, return an empty array [].
- Return ONLY the JSON array, no explanation.

Text:
{text}"""

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )
    return _llm


def _looks_like_team_page(text: str) -> bool:
    """Quick heuristic — does this page contain leadership content?"""
    keywords = ["ceo", "cto", "founder", "director", "head of", "vice president",
                "vp ", "chief", "manager", "president", "partner", "co-founder"]
    lower = text.lower()
    return sum(1 for kw in keywords if kw in lower) >= 2


async def _crawl_url(url: str) -> str:
    """Returns raw text from a URL using Crawl4AI, empty string on failure."""
    try:
        from crawl4ai import AsyncWebCrawler
        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await asyncio.wait_for(
                crawler.arun(url=url),
                timeout=_CRAWL_TIMEOUT,
            )
        return (result.markdown or result.extracted_content or "")[:_MAX_TEXT_CHARS]
    except Exception as e:
        logger.debug("website_team_tool: crawl failed for %s — %s", url, e)
        return ""


async def _extract_people_from_text(text: str, company_name: str, domain: str) -> list[dict]:
    """Uses Groq 8b to pull structured name+title pairs from raw page text."""
    prompt = _EXTRACT_PROMPT.format(text=text)
    try:
        response = await _get_llm().ainvoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        # Try direct parse first, then regex extraction
        try:
            people = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r'\[.*?\]', raw, re.DOTALL)
            people = json.loads(match.group()) if match else []

        if not isinstance(people, list):
            return []

        results = []
        for p in people:
            name  = str(p.get("name", "")).strip()
            title = str(p.get("title", "")).strip()
            if name and title and len(name) >= 3:
                results.append({
                    "name"        : name,
                    "title"       : title,
                    "company"     : company_name,
                    "linkedin_url": "",
                    "email"       : None,
                    "phone"       : None,
                    "source"      : "website_team",
                    "title_score" : 0.0,
                    "title_tier"  : 1,
                })
        return results

    except Exception as e:
        logger.warning("website_team_tool: LLM extraction failed — %s", e)
        return []


@tool
async def search_website_team(
    company_name   : str,
    company_domain : str,
    target_titles  : list[str],
    max_results    : int = 5,
) -> list[dict]:
    """
    Layer B — Website team page scraper. Tries /team, /about, /leadership and
    similar paths on the company's domain. Uses Groq 8b to extract names and
    titles from the raw page text. Works for any company that publishes its
    leadership online — especially SMEs and European companies.
    Returns [] if no team page is found or no people can be extracted.
    """
    if not company_domain:
        logger.warning("website_team_tool: no domain provided for '%s'", company_name)
        return []

    base = f"https://{company_domain}".rstrip("/")

    for path in _TEAM_PATHS:
        url  = f"{base}{path}"
        text = await _crawl_url(url)

        if not text or not _looks_like_team_page(text):
            continue

        logger.info("website_team_tool: team page found at %s", url)
        people = await _extract_people_from_text(text, company_name, company_domain)

        if people:
            # Filter by title relevance — keep only people whose title loosely
            # matches at least one of the target titles
            title_kws = {t.lower().split()[0] for t in target_titles if t}
            filtered = [
                p for p in people
                if any(kw in p["title"].lower() for kw in title_kws)
            ]
            # Fall back to all extracted people if the filter is too strict
            final = filtered if filtered else people
            logger.info(
                "website_team_tool: %d/%d people matched titles at '%s'",
                len(final), len(people), company_name,
            )
            return final[:max_results]

    logger.info("website_team_tool: no usable team page found for '%s'", company_name)
    return []
