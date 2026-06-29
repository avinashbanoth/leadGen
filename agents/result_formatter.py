import json
import logging
from collections import defaultdict

from graph.state import GraphState

logger = logging.getLogger(__name__)

_STATUS_ICON = {"verified": "✓", "partial": "~", "not_found": "✗"}


def _contact_line(contact: dict) -> str:
    icon      = _STATUS_ICON.get(contact.get("status", ""), "?")
    name      = contact.get("name", "?")
    title     = contact.get("title", "?")
    email_str = contact.get("email") or contact.get("suggestion") or "—"
    score     = contact.get("score", 0)
    linkedin  = contact.get("linkedin", "")
    conf      = contact.get("confidence", 0)
    tier      = contact.get("title_tier", 1)

    # Add a subtle indicator when this is a Level 2 (Director/Manager fallback) result
    tier_label = "  [L2 fallback]" if tier == 2 else ""

    line = f"  [{icon}] {name}  |  {title}{tier_label}\n"
    line += f"      Email: {email_str}"
    if conf:
        line += f"  (confidence: {conf}%)"
    line += f"  Score: {score}/100"
    if linkedin and not contact.get("email"):
        line += f"\n      LinkedIn: {linkedin}"
    return line


def _format_company_block(company_name: str, company_meta: dict, contacts: list[dict]) -> str:
    """Renders one company section: header + list of contacts beneath."""
    industry = company_meta.get("industry", "")
    website  = company_meta.get("website", "")
    location = company_meta.get("location", "")

    # Build a compact header: COMPANY NAME (Industry, Location) — website
    header_parts = [p for p in [industry, location] if p]
    header = f"── {company_name.upper()}"
    if header_parts:
        header += f"  ({', '.join(header_parts)})"
    if website:
        header += f"  —  {website}"
    header += " ──"

    lines = [header]
    for c in contacts:
        lines.append(_contact_line(c))
    return "\n".join(lines)


async def result_formatter(state: GraphState) -> dict:
    """
    Result Formatter — final node. Reads completed GraphState and produces
    a human-readable chat message grouped by company plus a structured JSON payload.
    Handles: verified results, partial results, clarification needed,
    rejected queries, and total failure (errors only).
    Writes to GraphState.messages and GraphState.status.
    """
    query    = state.get("query", "")
    plan     = state.get("query_plan", {})
    contacts = state.get("contacts", [])
    errors   = state.get("errors", [])
    status   = state.get("status", "")

    # ── Rejected / non-lead-gen query ───────────────────────────────────────
    if plan and not plan.get("is_lead_gen_query", True):
        reason  = plan.get("rejection_reason") or "This doesn't look like a lead generation query."
        message = f"Sorry, I can only help with B2B lead generation queries.\n\n{reason}"
        return {"messages": [message], "status": "rejected"}

    # ── Clarification needed ─────────────────────────────────────────────────
    if status == "awaiting_clarification":
        return {}   # clarification_node already wrote the message

    # ── No results ──────────────────────────────────────────────────────────
    if not contacts:
        companies = state.get("companies", [])
        people    = state.get("people", [])
        lines = [f"I couldn't find verified contacts for your query: \"{query}\""]

        if companies:
            lines.append(f"\nFound {len(companies)} matching company/companies:")
            for c in companies[:5]:
                name = c.get("name", "?")
                site = c.get("website", "")
                ind  = c.get("industry", "")
                lines.append(
                    f"  • {name}" +
                    (f" ({ind})" if ind else "") +
                    (f"  —  {site}" if site else "")
                )
        if people:
            lines.append(
                f"\nFound {len(people)} people but couldn't enrich their contact details."
            )
        if errors:
            lines.append(f"\nIssues encountered ({len(errors)}):")
            for e in errors[:5]:
                lines.append(f"  • {e}")
        lines.append(
            "\nTip: try adding a specific location or company type to your query, "
            "e.g. \"Find CTOs at fintech startups in Bangalore\"."
        )
        return {"messages": ["\n".join(lines)], "status": "no_results"}

    # ── Results found — group by company ────────────────────────────────────
    companies_state = state.get("companies", [])
    company_meta_map: dict[str, dict] = {
        c.get("name", "").upper(): c for c in companies_state
    }

    # Group contacts by company name (case-insensitive key)
    by_company: dict[str, list[dict]] = defaultdict(list)
    for c in contacts:
        key = (c.get("company") or "Unknown Company").upper()
        by_company[key].append(c)

    verified  = [c for c in contacts if c.get("status") == "verified"]
    partial   = [c for c in contacts if c.get("status") == "partial"]
    not_found = [c for c in contacts if c.get("status") == "not_found"]

    lines = [
        f"Found {len(contacts)} lead(s) across {len(by_company)} company/companies for:",
        f"  \"{query}\"\n",
        f"  ✓ {len(verified)} verified  ~  {len(partial)} partial  ✗ {len(not_found)} not found\n",
    ]

    for company_key, clist in sorted(by_company.items()):
        meta = company_meta_map.get(company_key, {})
        # Show verified + partial; omit not_found from the display
        visible = [c for c in clist if c.get("status") != "not_found"]
        if not visible:
            continue
        lines.append("")
        lines.append(_format_company_block(company_key, meta, visible))

    if errors:
        lines.append(f"\n[{len(errors)} pipeline issue(s) logged — check errors field]")

    chat_message = "\n".join(lines)

    payload = {
        "query"   : query,
        "stats"   : {
            "total"      : len(contacts),
            "verified"   : len(verified),
            "partial"    : len(partial),
            "not_found"  : len(not_found),
            "companies"  : len(by_company),
            "errors"     : len(errors),
        },
        "contacts": contacts,
        "errors"  : errors,
    }

    logger.info(
        "result_formatter: %d total leads (%d verified, %d partial) across %d companies.",
        len(contacts), len(verified), len(partial), len(by_company),
    )

    return {
        "messages"   : [chat_message],
        "status"     : "complete",
        "result_json": json.dumps(payload, indent=2),
    }
