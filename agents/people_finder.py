import asyncio
import logging
import os

from langchain_core.messages import HumanMessage

from graph.state import GraphState
from utils.role_normalizer import expand_role
from tools.linkedin_api_tool import search_linkedin_people
from tools.linkedin_scraper_tool import search_linkedin_people_browser
from tools.browser_use_tool import search_linkedin_people_agent
from tools.crosslinked_tool import search_crosslinked_people

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy Groq LLM — scores how well a found title matches the target role
# ---------------------------------------------------------------------------

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        from langchain_groq import ChatGroq
        _llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )
    return _llm


# ---------------------------------------------------------------------------
# Title scoring
# ---------------------------------------------------------------------------

_SCORE_PROMPT = """You are a B2B lead qualification assistant.
Rate how well the found job title matches the target role on a scale of 0.0 to 1.0.
1.0 = exact or near-exact match (e.g. target "CTO", found "Chief Technology Officer")
0.7 = strong partial match (e.g. target "CTO", found "VP Engineering")
0.4 = weak / indirect match (e.g. target "CTO", found "Senior Software Engineer")
0.0 = no match at all

Target role: {target_role}
Found title: {found_title}

Return ONLY a single decimal number between 0.0 and 1.0. Nothing else."""


async def _score_title(found_title: str, target_role: str) -> float:
    """Returns a 0.0–1.0 relevance score for a found title against the target role."""
    if not found_title or not target_role:
        return 0.0
    prompt = _SCORE_PROMPT.format(target_role=target_role, found_title=found_title)
    try:
        response = await _get_llm().ainvoke([HumanMessage(content=prompt)])
        return round(float(response.content.strip()), 2)
    except Exception:
        return 0.5   # default to mid-score on LLM failure


# ---------------------------------------------------------------------------
# Level 2 fallback role map — Director/Manager level titles
# Used when all 4 layers find zero results for Level 1 C-suite titles.
# "Something is better than nothing."
# ---------------------------------------------------------------------------

_LEVEL2_FALLBACKS: dict[str, list[str]] = {
    "cto": [
        "Director of Engineering", "Engineering Manager", "Head of Software Development",
        "Platform Lead", "Senior Engineering Manager", "VP of Technology",
    ],
    "ceo": [
        "Managing Director", "General Manager", "Board Member",
        "Director", "President", "Partner",
    ],
    "cfo": [
        "Finance Manager", "Controller", "Head of Accounting",
        "Finance Director", "Financial Controller",
    ],
    "coo": [
        "Operations Manager", "Director of Operations", "Head of Operations",
        "Senior Operations Manager",
    ],
    "vp engineering": [
        "Director of Engineering", "Engineering Manager", "Head of Software",
        "Senior Engineering Manager", "Software Development Manager",
    ],
    "vp sales": [
        "Sales Manager", "Head of Business Development", "Sales Director",
        "Regional Sales Manager", "Account Director",
    ],
    "vp product": [
        "Product Manager", "Senior Product Manager", "Director of Product",
        "Head of Product Management",
    ],
    "founder": [
        "Managing Director", "Co-Founder", "President",
        "Partner", "Director",
    ],
    "hr head": [
        "HR Manager", "Talent Acquisition Lead", "People Operations Manager",
        "HR Business Partner", "Recruitment Manager",
    ],
    "ciso": [
        "Security Manager", "Head of Cybersecurity", "InfoSec Manager",
        "IT Security Manager",
    ],
    "devops": [
        "DevOps Engineer", "SRE", "Platform Engineer",
        "Infrastructure Engineer", "Site Reliability Engineer",
    ],
}


def _get_level2_fallback(target_role: str) -> list[str]:
    """Returns Director/Manager-level fallback titles for a given role."""
    key = target_role.lower().strip()
    for dict_key, fallbacks in _LEVEL2_FALLBACKS.items():
        if dict_key in key or key in dict_key:
            return fallbacks
    # Generic fallback for unknown roles — strip "OR" alternatives and use base
    base = key.split(" or ")[0].strip()
    return [
        f"Director of {base.title()}",
        f"Head of {base.title()}",
        f"Manager, {base.title()}",
        f"Senior {base.title()} Manager",
    ]


# ---------------------------------------------------------------------------
# 4-layer cascade (internal) — shared by Level 1 and Level 2
# ---------------------------------------------------------------------------

async def _run_4_layers(
    company_name: str,
    company_domain: str,
    title_variants: list[str],
    max_results: int,
) -> list[dict]:
    """
    Runs all 4 LinkedIn layers in order for one company + title set.
    Returns as soon as any layer yields results (empty list if all fail).
    """
    # Layer 1 — Voyager HTTP API (fastest, no browser)
    try:
        people = await search_linkedin_people.ainvoke({
            "company_name"  : company_name,
            "target_titles" : title_variants,
            "max_results"   : max_results,
        })
    except Exception as e:
        logger.info("people_finder: Layer 1 failed for '%s' (%s) — trying Layer 2", company_name, type(e).__name__)
        people = []

    # Layer 2 — Camoufox stealth browser (hard-capped at 20s)
    if not people:
        try:
            people = await asyncio.wait_for(
                search_linkedin_people_browser.ainvoke({
                    "company_name"  : company_name,
                    "target_titles" : title_variants,
                    "max_results"   : max_results,
                }),
                timeout=20.0,
            )
        except asyncio.TimeoutError:
            logger.info("people_finder: Layer 2 timed out for '%s' — trying Layer 3", company_name)
            people = []
        except Exception as e:
            logger.info("people_finder: Layer 2 failed for '%s' (%s) — trying Layer 3", company_name, type(e).__name__)
            people = []

    # Layer 3 — browser-use LLM-driven browser
    if not people:
        try:
            people = await search_linkedin_people_agent.ainvoke({
                "company_name"  : company_name,
                "target_titles" : title_variants,
                "max_results"   : max_results,
            })
        except Exception as e:
            logger.info("people_finder: Layer 3 failed for '%s' (%s) — trying Layer 4", company_name, type(e).__name__)
            people = []

    # Layer 4 — Crosslinked Google dorks, no login
    if not people:
        try:
            people = await search_crosslinked_people.ainvoke({
                "company_name"  : company_name,
                "company_domain": company_domain,
                "target_titles" : title_variants,
                "max_results"   : max_results,
            })
        except Exception as e:
            logger.info("people_finder: Layer 4 failed for '%s' (%s)", company_name, type(e).__name__)
            people = []

    return people or []


# ---------------------------------------------------------------------------
# Per-company search with two-tier role priority
# ---------------------------------------------------------------------------

async def _find_people_for_company(
    company_name: str,
    company_domain: str,
    title_variants: list[str],
    fallback_variants: list[str],
    target_role: str,
    max_results: int,
) -> list[dict]:
    """
    Level 1: tries all 4 layers with C-suite/VP titles.
    Level 2: if Level 1 yields zero, retries all 4 layers with Director/Manager titles.
    Scores each person's title; marks title_tier=1 or 2 on each result.
    """
    # --- Level 1 ---
    people = await _run_4_layers(company_name, company_domain, title_variants, max_results)
    tier = 1

    # --- Level 2 fallback ---
    if not people and fallback_variants:
        logger.info(
            "people_finder: Level 1 exhausted for '%s' — falling back to Level 2 (Director/Manager).",
            company_name,
        )
        people = await _run_4_layers(company_name, company_domain, fallback_variants, max_results)
        tier = 2

    if not people:
        logger.info("people_finder: both tiers exhausted for '%s'.", company_name)
        return []

    # Score each person's title against the original target role concurrently
    score_tasks = [_score_title(p.get("title", ""), target_role) for p in people]
    scores = await asyncio.gather(*score_tasks, return_exceptions=True)

    for person, score in zip(people, scores):
        person["title_score"] = score if isinstance(score, float) else 0.5
        person["company"]     = person.get("company") or company_name
        person["title_tier"]  = tier  # 1 = C-suite, 2 = Director/Manager fallback

    # Level 2 is a "something is better than nothing" fallback — keep even weak matches
    min_score = 0.1 if tier == 2 else 0.3
    people = [p for p in people if p.get("title_score", 0) >= min_score]

    logger.info(
        "people_finder: %d qualified person(s) found for '%s' (tier %d).",
        len(people), company_name, tier,
    )
    return people


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

async def people_finder(state: GraphState) -> dict:
    """
    People Finder Agent — for each company in GraphState, cascades through all
    4 LinkedIn layers with two-tier role priority:
      Tier 1: C-suite/VP titles from expand_role()
      Tier 2: Director/Manager fallback if Tier 1 finds nothing (per company)
    Scores title relevance with Groq (0.0–1.0). Writes list[PersonData] to GraphState.
    Runs companies concurrently. Never raises — writes to errors on failure.
    """
    errors    = list(state.get("errors", []))
    companies = state.get("companies", [])
    query_plan = state.get("query_plan", {})
    target_role = query_plan.get("target_role") or ""

    if not companies:
        errors.append("people_finder: no companies in state — skipping.")
        return {"people": [], "errors": errors}

    if not target_role:
        errors.append("people_finder: no target_role in query_plan — skipping.")
        return {"people": [], "errors": errors}

    # Expand the role into Level 1 title variants once; reused for every company
    try:
        title_variants = await expand_role(target_role)
    except Exception as e:
        errors.append(f"people_finder: role expansion failed — {e}")
        title_variants = [target_role]

    # Level 2 fallback variants — Director/Manager level
    fallback_variants = _get_level2_fallback(target_role)
    logger.info(
        "people_finder: target_role='%s' | L1=%s | L2=%s",
        target_role, title_variants, fallback_variants,
    )

    async def _safe_find(company: dict) -> list[dict]:
        company_name = company.get("name", "")
        website = company.get("website", "")
        domain = website.split("/")[2].replace("www.", "") if website.startswith("http") else ""
        try:
            return await _find_people_for_company(
                company_name     = company_name,
                company_domain   = domain,
                title_variants   = title_variants,
                fallback_variants= fallback_variants,
                target_role      = target_role,
                max_results      = 5,
            )
        except Exception as e:
            errors.append(f"people_finder: company '{company_name}' failed — {e}")
            return []

    # Run all companies concurrently
    tasks   = [_safe_find(c) for c in companies]
    batches = await asyncio.gather(*tasks, return_exceptions=True)

    all_people: list[dict] = []
    for batch in batches:
        if isinstance(batch, Exception):
            errors.append(f"people_finder: batch exception — {batch}")
            continue
        all_people.extend(batch)

    # Sort by tier first (Level 1 preferred), then by title_score descending
    all_people.sort(key=lambda p: (p.get("title_tier", 1), -p.get("title_score", 0.0)))

    logger.info(
        "people_finder: total %d qualified people across all companies "
        "(%d tier-1, %d tier-2).",
        len(all_people),
        sum(1 for p in all_people if p.get("title_tier", 1) == 1),
        sum(1 for p in all_people if p.get("title_tier", 1) == 2),
    )

    return {
        "people": all_people,
        "errors": errors,
        "status": "people_found" if all_people else "no_people_found",
    }
