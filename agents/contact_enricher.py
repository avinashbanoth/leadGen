import asyncio
import logging

from graph.state import GraphState
from providers.hunter_provider import HunterProvider
from providers.permutator_provider import PermutatorProvider
from providers.harvester_provider import HarvesterProvider
from providers.google_dork_provider import GoogleDorkProvider
from providers.linkedin_contact_provider import LinkedInContactProvider
from providers.website_contact_provider import WebsiteContactProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider chain — tried in order, stops at the first verified result
# ---------------------------------------------------------------------------

_PROVIDERS = [
    HunterProvider(),
    PermutatorProvider(),
    HarvesterProvider(),
    GoogleDorkProvider(),
    LinkedInContactProvider(),
    WebsiteContactProvider(),
]

_MIN_CONFIDENCE = 30   # below this, keep trying next provider


def _domain_from_company(companies: list[dict], company_name: str) -> str:
    """Finds the company domain from the companies list by matching name."""
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
    Runs the 6-level provider chain for a single person.
    Returns a ContactData-shaped dict — never returns None, never silent-fails.
    """
    full_name = person.get("name", "")
    title     = person.get("title", "")
    company   = person.get("company", "")
    linkedin  = person.get("linkedin_url", "")

    first, last = _split_name(full_name)
    domain      = _domain_from_company(companies, company)

    # Find the lead score for this person
    score = 0
    for ls in lead_scores:
        if ls.get("person", "").lower() == full_name.lower():
            score = ls.get("score", 0)
            break

    tried: list[str] = []
    email: str | None = None
    confidence: int   = 0
    source: str       = ""

    for provider in _PROVIDERS:
        tried.append(provider.name)

        # LinkedInContactProvider needs the profile URL injected
        if hasattr(provider, "_linkedin_url"):
            provider._linkedin_url = linkedin

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
                "contact_enricher: email found for '%s' via %s (confidence=%d).",
                full_name, provider.name, confidence,
            )
            break

    # Determine status
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
    Contact Enricher Agent — for every person in GraphState, walks the 6-level
    EmailProvider chain (Hunter → Permutator → Harvester → GoogleDork → LinkedIn → Website)
    and stops at the first result with confidence ≥ 30.
    Status is 'verified' (≥70), 'partial' (<70), or 'not_found'.
    Partial results always include a suggestion (LinkedIn URL) rather than failing silently.
    Writes list[ContactData] to GraphState. Never raises.
    """
    errors     = list(state.get("errors", []))
    people     = state.get("people", [])
    companies  = state.get("companies", [])
    lead_scores = state.get("lead_score", [])

    if not people:
        errors.append("contact_enricher: no people in state — skipping.")
        return {"contacts": [], "errors": errors}

    # Sort people by lead score (highest first), then cap to conserve quota.
    # Default cap = 5; callers that need more can expand the people list before this node.
    _MAX_ENRICH = 5
    if lead_scores:
        score_map = {ls.get("person", "").lower(): ls.get("score", 0) for ls in lead_scores}
        people = sorted(people, key=lambda p: score_map.get(p.get("name", "").lower(), 0), reverse=True)
    people = people[:_MAX_ENRICH]
    if len(people) < len(state.get("people", [])):
        logger.info(
            "contact_enricher: capped enrichment to top %d of %d people (quota conservation).",
            len(people), len(state.get("people", [])),
        )

    _PER_PERSON_TIMEOUT = 25.0   # max seconds per person across all 6 providers

    async def _safe_enrich(person: dict) -> dict:
        name = person.get("name", "unknown")
        try:
            return await asyncio.wait_for(
                _enrich_person(person, companies, lead_scores),
                timeout=_PER_PERSON_TIMEOUT,
            )
        except asyncio.TimeoutError:
            errors.append(f"contact_enricher: timed out for '{name}' after {_PER_PERSON_TIMEOUT}s")
        except Exception as e:
            errors.append(f"contact_enricher: failed for '{name}' — {e}")
        # Partial result rather than silent failure (Rule 9)
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

    # Enrich all people concurrently — was sequential (10 × 90s = 15 min)
    contacts: list[dict] = await asyncio.gather(*[_safe_enrich(p) for p in people])

    verified = sum(1 for c in contacts if c["status"] == "verified")
    partial  = sum(1 for c in contacts if c["status"] == "partial")
    logger.info(
        "contact_enricher: %d verified, %d partial, %d not_found.",
        verified, partial, len(contacts) - verified - partial,
    )

    return {
        "contacts": contacts,
        "errors"  : errors,
        "status"  : "enrichment_complete",
    }
