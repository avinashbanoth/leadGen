import os
import re
import aiohttp
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


# ---------------------------------------------------------------------------
# Static role expansion dictionary — covers the most common B2B target roles.
# Tried in order; LinkedIn search stops when results are found.
# ---------------------------------------------------------------------------

ROLE_EXPANSIONS: dict[str, list[str]] = {
    "cto": [
        "CTO",
        "Chief Technology Officer",
        "VP Engineering",
        "VP of Engineering",
        "Head of Technology",
        "Head of Engineering",
        "Co-founder & CTO",
        "Director of Engineering",
    ],
    "ceo": [
        "CEO",
        "Chief Executive Officer",
        "Founder & CEO",
        "Co-founder & CEO",
        "Managing Director",
        "President",
    ],
    "cfo": [
        "CFO",
        "Chief Financial Officer",
        "VP Finance",
        "Head of Finance",
        "Finance Director",
    ],
    "coo": [
        "COO",
        "Chief Operating Officer",
        "VP Operations",
        "Head of Operations",
        "Director of Operations",
    ],
    "hr head": [
        "HR Head",
        "Head of HR",
        "CHRO",
        "Chief Human Resources Officer",
        "VP People",
        "VP Human Resources",
        "Head of People",
        "HR Director",
        "Director of Human Resources",
    ],
    "vp engineering": [
        "VP Engineering",
        "VP of Engineering",
        "Vice President Engineering",
        "Head of Engineering",
        "Director of Engineering",
        "Engineering Director",
    ],
    "vp sales": [
        "VP Sales",
        "VP of Sales",
        "Vice President Sales",
        "Head of Sales",
        "Sales Director",
        "Director of Sales",
        "Chief Revenue Officer",
        "CRO",
    ],
    "founder": [
        "Founder",
        "Co-founder",
        "CEO",
        "Founder & CEO",
        "Co-founder & CEO",
        "Managing Director",
    ],
    "ciso": [
        "CISO",
        "Chief Information Security Officer",
        "VP Security",
        "Head of Security",
        "Head of Cybersecurity",
        "Director of Security",
    ],
    "devops": [
        "DevOps Engineer",
        "DevOps Lead",
        "Head of DevOps",
        "VP Infrastructure",
        "Director of DevOps",
        "Platform Engineering Lead",
        "SRE Lead",
    ],
    "product manager": [
        "Product Manager",
        "Senior Product Manager",
        "VP Product",
        "Head of Product",
        "Chief Product Officer",
        "CPO",
        "Director of Product",
    ],
}

SEARXNG_URL = "http://localhost:8080/search"

# ---------------------------------------------------------------------------
# Lazy Gemini init — only used for roles not in the static dictionary
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

def _normalize_key(role: str) -> str:
    return role.lower().strip()


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
    Expands a target role into a list of equivalent LinkedIn titles.
    Checks the static dictionary first; falls back to Gemini for unknown roles.
    Returns the original role string as a single-item list if all else fails.
    """
    key = _normalize_key(target_role)

    for dict_key, expansions in ROLE_EXPANSIONS.items():
        if dict_key in key or key in dict_key:
            return expansions

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
