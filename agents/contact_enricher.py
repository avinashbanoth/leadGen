import asyncio
import logging

from graph.state import GraphState
from providers.hunter_provider import HunterProvider
from providers.permutator_provider import PermutatorProvider
from providers.harvester_provider import HarvesterProvider
from providers.google_dork_provider import GoogleDorkProvider
from providers.website_contact_provider import WebsiteContactProvider

logger = logging.getLogger(__name__)

_MIN_CONFIDENCE = 30


def _make_providers() -> list:
    """
    Returns a fresh list of provider instances per enrichment call.
    Avoids shared-singleton state corruption when multiple people are
    enriched concurrently via asyncio.gather.
    """
    return [
        HunterProvider(),
        PermutatorProvider(),
        HarvesterProvider(),
        GoogleDorkProvider(),
        WebsiteContactProvider(),
    ]


def _domain_from_company(companies: list[dict], company_name: str) -> str:
    for c in companies:
        if c.get("name", "").lower() == company_name.lower():
            website = c.get("website", "")
            if website.startswith("http"):
                return website.split("/")[2].replace("www.", "")
    return ""


def _split_name(full_name: str) -> tuple[str, str]:
    parts = full_name.strip().split()
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return full_name, ""


async def _enrich_person(person: dict, companies: list[dict], lead_scores: list[dict]) -> dict:
    """
    Runs the 5-level provider chain for a single person.
    Fresh providers are created per call — no shared mutable state.
    Returns a ContactData-shaped dict — never None, never silent-fail.
    """
    full_name = person.get("name", "")
    title     = person.get("title", "")
    company   = person.get("company", "")
    linkedin  = person.get("linkedin_url", "")

    first, last = _split_name(full_name)
    domain      = _domain_from_company(companies, company)

    score = 0
    for ls in lead_scores:
        if ls.get("person", "").lower() == full_name.lower():
            score = ls.get("score", 0)
            break

    tried: list[str] = []
    email: str | None = None
    confidence: int   = 0
    source: str       = ""

    # If Apollo already returned a verified email, use it immediately
    apollo_email = person.get("email")
    if apollo_email:
        email      = apollo_email
        confidence = 80
        source     = "apollo"
        tried      = ["apollo"]
    else:
        for provider in _make_providers():
            tried.append(provider.name)
            try:
                result = await provider.find(first, last, domain)
            except Exception as e:
                logger.warning("contact_enricher: provider '%s' raised — %s", provider.name, e)
                continue

            if result.get("email") and result.get("confidence", 0) >= _MIN_CONFIDENCE:
                email      = result["email"]
                confidence = result["confidence"]
                source     = result["source"]
                logger.info(
                    "contact_enricher: email for '%s' via %s (confidence=%d).",
                    full_name, provider.name, confidence,
                )
                break

    if email and confidence >= 70:
        status = "verified"
    elif email:
        status = "partial"
    else:
        status = "not_found"

    suggestion = linkedin if not email and linkedin else None

    return {
        "name"       : full_name,
        "title"      : title,
        "company"    : company,
        "email"      : email,
        "confidence" : confidence,
        "linkedin"   : linkedin,
        "phone"      : person.get("phone"),
        "score"      : score,
        "status"     : status,
        "tried"      : tried,
        "suggestion" : suggestion,
        "title_tier" : person.get("title_tier", 1),
    }


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

async def contact_enricher(state: GraphState) -> dict:
    """
    Contact Enricher Agent — for every person in GraphState, walks the 5-level
    EmailProvider chain (Hunter → Permutator → Harvester → GoogleDork → Website)
    and stops at the first result with confidence ≥ 30.
    If Apollo already provided a verified email, skips the chain entirely.
    Status: 'verified' (≥70%), 'partial' (<70%), or 'not_found'.
    Partial results always include a suggestion (LinkedIn URL).
    Writes list[ContactData] to GraphState. Never raises.
    """
    errors      = list(state.get("errors", []))
    people      = state.get("people", [])
    companies   = state.get("companies", [])
    lead_scores = state.get("lead_score", [])

    if not people:
        errors.append("contact_enricher: no people in state — skipping.")
        return {"contacts": [], "errors": errors}

    _MAX_ENRICH = 5
    if lead_scores:
        score_map = {ls.get("person", "").lower(): ls.get("score", 0) for ls in lead_scores}
        people = sorted(people, key=lambda p: score_map.get(p.get("name", "").lower(), 0), reverse=True)
    people = people[:_MAX_ENRICH]

    if len(people) < len(state.get("people", [])):
        logger.info(
            "contact_enricher: capped to top %d of %d people (quota conservation).",
            len(people), len(state.get("people", [])),
        )

    _PER_PERSON_TIMEOUT = 30.0

    async def _safe_enrich(person: dict) -> dict:
        name = person.get("name", "unknown")
        try:
            return await asyncio.wait_for(
                _enrich_person(person, companies, lead_scores),
                timeout=_PER_PERSON_TIMEOUT,
            )
        except asyncio.TimeoutError:
            errors.append(f"contact_enricher: timed out for '{name}'")
        except Exception as e:
            errors.append(f"contact_enricher: failed for '{name}' — {e}")
        return {
            "name"       : name,
            "title"      : person.get("title", ""),
            "company"    : person.get("company", ""),
            "email"      : None,
            "confidence" : 0,
            "linkedin"   : person.get("linkedin_url", ""),
            "phone"      : None,
            "score"      : 0,
            "status"     : "partial",
            "tried"      : [],
            "suggestion" : person.get("linkedin_url") or None,
            "title_tier" : person.get("title_tier", 1),
        }

    contacts: list[dict] = await asyncio.gather(*[_safe_enrich(p) for p in people])

    verified = sum(1 for c in contacts if c["status"] == "verified")
    partial  = sum(1 for c in contacts if c["status"] == "partial")
    logger.info(
        "contact_enricher: %d verified, %d partial, %d not_found.",
        verified, partial, len(contacts) - verified - partial,
    )

    return {
        "contacts": list(contacts),
        "errors"  : errors,
        "status"  : "enrichment_complete",
    }
