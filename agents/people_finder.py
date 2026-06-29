import asyncio
import logging
import os

from langchain_core.messages import HumanMessage

from graph.state import GraphState
from utils.role_normalizer import expand_role
from tools.apollo_tool import search_apollo_people
from tools.kompass_tool import search_kompass_executives
from tools.zaubacorp_tool import search_zaubacorp_directors
from tools.website_team_tool import search_website_team
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

_SCORE_PROMPT = """Rate how well the found job title matches the target role (0.0–1.0).
1.0 = exact match (target "CTO", found "Chief Technology Officer")
0.7 = strong partial (target "CTO", found "VP Engineering")
0.4 = weak/indirect (target "CTO", found "Senior Software Engineer")
0.0 = no match

Target role: {target_role}
Found title: {found_title}

Return ONLY a single decimal between 0.0 and 1.0."""


async def _score_title(found_title: str, target_role: str) -> float:
    if not found_title or not target_role:
        return 0.0
    prompt = _SCORE_PROMPT.format(target_role=target_role, found_title=found_title)
    try:
        response = await _get_llm().ainvoke([HumanMessage(content=prompt)])
        return round(float(response.content.strip()), 2)
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# Level 2 fallback titles — LLM generates Director/Manager equivalents
# ---------------------------------------------------------------------------

async def _get_level2_fallback(target_role: str) -> list[str]:
    """Ask the LLM for Director/Manager-level equivalents of any role."""
    base_role = target_role.split(" OR ")[0].strip()
    prompt = (
        f'List 5 Director or Manager level job titles that report to or are equivalent to "{base_role}". '
        f'These should be mid-senior titles (Director, Manager, Head of, Lead), not C-suite. '
        f'Return only a JSON array of strings. No explanation.'
    )
    try:
        response = await _get_llm().ainvoke([HumanMessage(content=prompt)])
        import re, json
        match = re.search(r'\[.*?\]', response.content, re.DOTALL)
        if match:
            titles = json.loads(match.group())
            if titles:
                return [str(t) for t in titles if t]
    except Exception as e:
        logger.warning("people_finder: LLM fallback title generation failed — %s", e)
    # Last resort: generic template
    base = base_role.split(" or ")[0].strip()
    return [f"Director of {base}", f"Head of {base}", f"Senior {base} Manager"]


# ---------------------------------------------------------------------------
# 5-layer cascade: Apollo → Kompass → Zaubacorp → Website team → Google dorks
# ---------------------------------------------------------------------------

async def _run_3_layers(
    company_name   : str,
    company_domain : str,
    title_variants : list[str],
    max_results    : int,
) -> list[dict]:
    """
    Layer A: Apollo.io people search (primary — real B2B database).
    Layer B: Kompass India executive profiles (manufacturing, healthcare, B2B).
    Layer C: Zaubacorp MCA directors (official Indian government registry).
    Layer D: Website team page scraper (generic fallback — no login needed).
    Layer E: Google dorks via SearXNG (last resort — large public companies).
    Returns on the first layer that yields results.
    """
    # Layer A — Apollo
    try:
        people = await search_apollo_people.ainvoke({
            "company_name" : company_name,
            "target_titles": title_variants,
            "max_results"  : max_results,
        })
    except Exception as e:
        logger.info("people_finder: Layer A (Apollo) failed for '%s' — %s", company_name, e)
        people = []

    # Layer B — Kompass executive profiles
    if not people:
        try:
            people = await search_kompass_executives.ainvoke({
                "company_name" : company_name,
                "target_titles": title_variants,
                "max_results"  : max_results,
            })
        except Exception as e:
            logger.info("people_finder: Layer B (Kompass) failed for '%s' — %s", company_name, e)
            people = []

    # Layer C — Zaubacorp MCA directors
    if not people:
        try:
            people = await search_zaubacorp_directors.ainvoke({
                "company_name" : company_name,
                "target_titles": title_variants,
                "max_results"  : max_results,
            })
        except Exception as e:
            logger.info("people_finder: Layer C (Zaubacorp) failed for '%s' — %s", company_name, e)
            people = []

    # Layer D — Website team pages
    if not people:
        try:
            people = await asyncio.wait_for(
                search_website_team.ainvoke({
                    "company_name"  : company_name,
                    "company_domain": company_domain,
                    "target_titles" : title_variants,
                    "max_results"   : max_results,
                }),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.info("people_finder: Layer D (website team) timed out for '%s'", company_name)
            people = []
        except Exception as e:
            logger.info("people_finder: Layer D (website team) failed for '%s' — %s", company_name, e)
            people = []

    # Layer E — Google dorks (crosslinked)
    if not people:
        try:
            people = await search_crosslinked_people.ainvoke({
                "company_name"  : company_name,
                "company_domain": company_domain,
                "target_titles" : title_variants,
                "max_results"   : max_results,
            })
        except Exception as e:
            logger.info("people_finder: Layer E (dorks) failed for '%s' — %s", company_name, e)
            people = []

    return people or []


# ---------------------------------------------------------------------------
# Per-company search with two-tier role priority
# ---------------------------------------------------------------------------

async def _find_people_for_company(
    company_name     : str,
    company_domain   : str,
    title_variants   : list[str],
    fallback_variants: list[str],
    target_role      : str,
    max_results      : int,
) -> list[dict]:
    """
    Tier 1: runs 3-layer cascade with C-suite/VP titles.
    Tier 2: if Tier 1 yields zero, retries with Director/Manager fallback titles.
    Scores each result and stamps title_tier=1 or 2.
    """
    people = await _run_3_layers(company_name, company_domain, title_variants, max_results)
    tier = 1

    if not people and fallback_variants:
        logger.info(
            "people_finder: Tier 1 exhausted for '%s' — trying Tier 2 (Director/Manager).",
            company_name,
        )
        people = await _run_3_layers(company_name, company_domain, fallback_variants, max_results)
        tier = 2

    if not people:
        logger.info("people_finder: both tiers exhausted for '%s'.", company_name)
        return []

    score_tasks = [_score_title(p.get("title", ""), target_role) for p in people]
    scores = await asyncio.gather(*score_tasks, return_exceptions=True)

    for person, score in zip(people, scores):
        person["title_score"] = score if isinstance(score, float) else 0.5
        person["company"]     = person.get("company") or company_name
        person["title_tier"]  = tier

    min_score = 0.1 if tier == 2 else 0.3
    people = [p for p in people if p.get("title_score", 0) >= min_score]

    logger.info(
        "people_finder: %d qualified person(s) for '%s' (tier %d).",
        len(people), company_name, tier,
    )
    return people


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

async def people_finder(state: GraphState) -> dict:
    """
    People Finder Agent — for each company in GraphState, runs the 3-layer
    cascade (Apollo → website team → Google dorks) with two-tier role priority
    (C-suite first, Director/Manager fallback if nothing found).
    Writes list[PersonData] with title_score and title_tier to GraphState.
    Runs companies concurrently. Never raises — writes to errors on failure.
    """
    errors      = list(state.get("errors", []))
    companies   = state.get("companies", [])
    query_plan  = state.get("query_plan", {})
    target_role = query_plan.get("target_role") or ""

    if not companies:
        named = query_plan.get("named_company")
        if query_plan.get("company_named_directly") and named:
            companies = [{"name": named, "website": "", "confidence": 1.0}]
        else:
            errors.append("people_finder: no companies in state — skipping.")
            return {"people": [], "errors": errors}

    if not target_role:
        errors.append("people_finder: no target_role in query_plan — skipping.")
        return {"people": [], "errors": errors}

    try:
        title_variants = await expand_role(target_role)
    except Exception as e:
        errors.append(f"people_finder: role expansion failed — {e}")
        title_variants = [target_role]

    fallback_variants = await _get_level2_fallback(target_role)
    logger.info(
        "people_finder: target='%s' | L1 titles=%s | L2 fallback=%s",
        target_role, title_variants[:3], fallback_variants[:2],
    )

    async def _safe_find(company: dict) -> list[dict]:
        name    = company.get("name", "")
        website = company.get("website", "")
        domain  = website.split("/")[2].replace("www.", "") if website.startswith("http") else ""
        try:
            return await _find_people_for_company(
                company_name     = name,
                company_domain   = domain,
                title_variants   = title_variants,
                fallback_variants= fallback_variants,
                target_role      = target_role,
                max_results      = 5,
            )
        except Exception as e:
            errors.append(f"people_finder: '{name}' failed — {e}")
            return []

    tasks   = [_safe_find(c) for c in companies]
    batches = await asyncio.gather(*tasks, return_exceptions=True)

    all_people: list[dict] = []
    for batch in batches:
        if isinstance(batch, Exception):
            errors.append(f"people_finder: batch exception — {batch}")
            continue
        all_people.extend(batch)

    # Tier 1 before Tier 2, then by score descending within each tier
    all_people.sort(key=lambda p: (p.get("title_tier", 1), -p.get("title_score", 0.0)))

    logger.info(
        "people_finder: %d total people (%d tier-1, %d tier-2).",
        len(all_people),
        sum(1 for p in all_people if p.get("title_tier", 1) == 1),
        sum(1 for p in all_people if p.get("title_tier", 1) == 2),
    )

    return {
        "people": all_people,
        "errors": errors,
        "status": "people_found" if all_people else "no_people_found",
    }
