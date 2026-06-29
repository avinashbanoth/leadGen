import os
import re
import aiohttp
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080/search")

# ---------------------------------------------------------------------------
# Groq LLM — used for all role expansions
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Role expansion
# ---------------------------------------------------------------------------


async def _expand_via_llm(role: str) -> list[str]:
    prompt = (
        f'List 6 equivalent LinkedIn job titles for "{role}". '
        f'Return only a JSON array of strings. No explanation.'
    )
    try:
        response = await _get_llm().ainvoke([
            SystemMessage(content="You are a LinkedIn recruiter who knows all equivalent job titles."),
            HumanMessage(content=prompt),
        ])
        raw = response.content.strip()
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            import json
            titles = json.loads(match.group())
            return [str(t) for t in titles if t]
    except Exception:
        pass
    return [role]


async def expand_role(target_role: str) -> list[str]:
    """
    Expands any target role into equivalent LinkedIn job titles via LLM.
    Works for any role in any industry — not limited to a static list.
    Returns the original role string as a single-item list if all else fails.
    """
    return await _expand_via_llm(target_role)


# ---------------------------------------------------------------------------
# Company name normalization
# ---------------------------------------------------------------------------

async def normalize_company_name(company_name: str) -> str:
    """
    Searches SearXNG for the official LinkedIn company page and extracts
    the canonical company name from the page title.
    Falls back to the original name if SearXNG is unavailable or returns nothing.
    """
    keywords = [company_name, "site:linkedin.com/company"]
    params = {
        "q"         : " ".join(keywords),
        "format"    : "json",
        "categories": "general",
        "language"  : "en",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                SEARXNG_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                response.raise_for_status()
                data = await response.json()

        results = data.get("results", [])
        for result in results[:3]:
            url = result.get("url", "")
            title = result.get("title", "")
            if "linkedin.com/company/" in url and title:
                canonical = title.split("|")[0].strip()
                if canonical:
                    return canonical

    except Exception:
        pass

    return company_name
