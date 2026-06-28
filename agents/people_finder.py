import asyncio
import logging
import os

from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import GraphState
from utils.role_normalizer import expand_role
from tools.linkedin_api_tool import search_linkedin_people
from tools.linkedin_scraper_tool import search_linkedin_people_browser
from tools.browser_use_tool import search_linkedin_people_agent
from tools.crosslinked_tool import search_crosslinked_people

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy Cerebras LLM — scores how well a found title matches the target role
# ---------------------------------------------------------------------------

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        from langchain_openai import ChatOpenAI
        _llm = ChatOpenAI(
            model="gpt-oss-120b",
            base_url="https://api.cerebras.ai/v1",
            api_key=os.getenv("CEREBRAS_API_KEY"),
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
# 4-layer cascade for a single company
# ---------------------------------------------------------------------------

async def _find_people_for_company(
    company_name: str,
    company_domain: str,
    title_variants: list[str],
    target_role: str,
    max_results: int,
) -> list[dict]:
    """
    Tries all 4 LinkedIn layers in order for one company.
    Returns as soon as any layer yields results.
    Each person's title_score is filled before returning.
    """

    # Layer 1 — Voyager HTTP API (fastest, no browser)
    people = await search_linkedin_people.ainvoke({
        "company_name"  : company_name,
        "target_titles" : title_variants,
        "max_results"   : max_results,
    })

    # Layer 2 — Camoufox stealth browser
    if not people:
        people = await search_linkedin_people_browser.ainvoke({
            "company_name"  : company_name,
            "target_titles" : title_variants,
            "max_results"   : max_results,
        })

    # Layer 3 — browser-use LLM-driven browser
    if not people:
        people = await search_linkedin_people_agent.ainvoke({
            "company_name"  : company_name,
            "target_titles" : title_variants,
            "max_results"   : max_results,
        })

    # Layer 4 — Crosslinked Google dorks, no login
    if not people:
        people = await search_crosslinked_people.ainvoke({
            "company_name"  : company_name,
            "company_domain": company_domain,
            "target_titles" : title_variants,
            "max_results"   : max_results,
        })

    if not people:
        logger.info("people_finder: all 4 layers exhausted for '%s'.", company_name)
        return []

    # Score each person's title against the target role concurrently
    score_tasks = [_score_title(p.get("title", ""), target_role) for p in people]
    scores = await asyncio.gather(*score_tasks, return_exceptions=True)

    for person, score in zip(people, scores):
        person["title_score"] = score if isinstance(score, float) else 0.5
        person["company"] = person.get("company") or company_name

    # Drop anyone with title_score below 0.3 — clearly wrong role
    people = [p for p in people if p.get("title_score", 0) >= 0.3]

    logger.info(
        "people_finder: %d qualified person(s) found for '%s'.",
        len(people), company_name,
    )
    return people


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

async def people_finder(state: GraphState) -> dict:
    """
    People Finder Agent — for each company in GraphState, cascades through all
    4 LinkedIn layers to find decision makers matching the target role.
    Scores title relevance with Cerebras (0.0–1.0). Drops results below 0.3.
    Writes list[PersonData] with title_score filled to GraphState.
    Runs companies concurrently. Never raises — writes to errors on failure.
    """
    errors  = list(state.get("errors", []))
    companies = state.get("companies", [])
    query_plan = state.get("query_plan", {})
    target_role = query_plan.get("target_role") or ""

    if not companies:
        errors.append("people_finder: no companies in state — skipping.")
        return {"people": [], "errors": errors}

    if not target_role:
        errors.append("people_finder: no target_role in query_plan — skipping.")
        return {"people": [], "errors": errors}

    # Expand the role into title variants once; reused for every company
    try:
        title_variants = await expand_role(target_role)
    except Exception as e:
        errors.append(f"people_finder: role expansion failed — {e}")
        title_variants = [target_role]

    async def _safe_find(company: dict) -> list[dict]:
        company_name = company.get("name", "")
        # Extract domain from website URL for Layer 4
        website = company.get("website", "")
        domain = website.split("/")[2].replace("www.", "") if website.startswith("http") else ""
        try:
            return await _find_people_for_company(
                company_name  = company_name,
                company_domain= domain,
                title_variants= title_variants,
                target_role   = target_role,
                max_results   = 5,
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

    # Sort by title_score descending
    all_people.sort(key=lambda p: p.get("title_score", 0.0), reverse=True)

    logger.info("people_finder: total %d qualified people across all companies.", len(all_people))

    return {
        "people": all_people,
        "errors": errors,
        "status": "people_found" if all_people else "no_people_found",
    }
