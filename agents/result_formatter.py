import json
import logging

from graph.state import GraphState

logger = logging.getLogger(__name__)


def _format_contact_row(contact: dict) -> str:
    status_icon = {"verified": "✓", "partial": "~", "not_found": "✗"}.get(contact.get("status", ""), "?")
    email_str   = contact.get("email") or contact.get("suggestion") or "—"
    return (
        f"  [{status_icon}] {contact.get('name', '?')} | {contact.get('title', '?')} "
        f"@ {contact.get('company', '?')}\n"
        f"      Email: {email_str}  Score: {contact.get('score', 0)}/100  "
        f"Confidence: {contact.get('confidence', 0)}%\n"
        f"      LinkedIn: {contact.get('linkedin', '—')}"
    )


async def result_formatter(state: GraphState) -> dict:
    """
    Result Formatter — final node in the graph. Reads the completed GraphState and
    produces a human-readable chat message plus a structured JSON payload.
    Handles all states: verified results, partial results, clarification needed,
    rejected queries, and total failure (errors only).
    Writes to GraphState.messages and GraphState.status.
    """
    query   = state.get("query", "")
    plan    = state.get("query_plan", {})
    contacts = state.get("contacts", [])
    errors  = state.get("errors", [])
    status  = state.get("status", "")

    # ── Rejected / non-lead-gen query ───────────────────────────────────────
    if plan and not plan.get("is_lead_gen_query", True):
        reason = plan.get("rejection_reason") or "This doesn't look like a lead generation query."
        message = f"Sorry, I can only help with B2B lead generation queries.\n\n{reason}"
        return {"messages": [message], "status": "rejected"}

    # ── Clarification needed ─────────────────────────────────────────────────
    if status == "awaiting_clarification":
        # clarification_node already wrote the message — nothing to add
        return {}

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
                lines.append(f"  • {name}" + (f" ({ind})" if ind else "") + (f" — {site}" if site else ""))
        if people:
            lines.append(f"\nFound {len(people)} people but couldn't enrich their email addresses.")
        if errors:
            lines.append(f"\nIssues encountered ({len(errors)}):")
            for e in errors[:5]:
                lines.append(f"  • {e}")
        lines.append("\nTip: try a more specific industry, location, or company name.")
        return {"messages": ["\n".join(lines)], "status": "no_results"}

    # ── Results found ────────────────────────────────────────────────────────
    verified = [c for c in contacts if c.get("status") == "verified"]
    partial  = [c for c in contacts if c.get("status") == "partial"]
    not_found = [c for c in contacts if c.get("status") == "not_found"]

    lines = [f"Found {len(contacts)} lead(s) for: \"{query}\"\n"]
    lines.append(f"  ✓ {len(verified)} verified  ~  {len(partial)} partial  ✗ {len(not_found)} not found\n")

    if verified:
        lines.append("── Verified Leads ──")
        for c in verified:
            lines.append(_format_contact_row(c))

    if partial:
        lines.append("\n── Partial Leads (email unverified or unavailable) ──")
        for c in partial:
            lines.append(_format_contact_row(c))

    if errors:
        lines.append(f"\n[{len(errors)} pipeline issue(s) logged — check errors field for details]")

    chat_message = "\n".join(lines)

    # Structured JSON payload for the API response
    payload = {
        "query"   : query,
        "stats"   : {
            "total"    : len(contacts),
            "verified" : len(verified),
            "partial"  : len(partial),
            "not_found": len(not_found),
            "errors"   : len(errors),
        },
        "contacts": contacts,
        "errors"  : errors,
    }

    logger.info(
        "result_formatter: %d total leads (%d verified, %d partial).",
        len(contacts), len(verified), len(partial),
    )

    return {
        "messages": [chat_message],
        "status"  : "complete",
        "result_json": json.dumps(payload, indent=2),
    }
