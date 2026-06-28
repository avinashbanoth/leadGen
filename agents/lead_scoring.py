import json
import logging
import os
import re

from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import GraphState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy Gemini 2.5 Flash — ONLY reads GraphState, calls NO tools
# This is the hallucination guard: no new information can enter here
# ---------------------------------------------------------------------------

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        from langchain_groq import ChatGroq
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )
    return _llm


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a B2B lead qualification expert.
You will receive structured data about companies, people, and signals gathered for a sales query.
Your job is to score each (company, person) pair from 0 to 100 and explain why.

Scoring rubric:
- 80–100: Strong fit. Company matches all criteria. Person's title is a strong match. High-strength signals present.
- 60–79:  Good fit. Company mostly matches. Title is relevant. Some signals.
- 40–59:  Partial fit. Missing some criteria or title is indirect. Weak signals.
- 0–39:   Poor fit. Company or person doesn't match the query intent.

Rules:
- Do NOT look up any external information. Score ONLY based on the data provided.
- Be specific in reasons — mention the actual signal, title, or company detail.
- Return ONLY a valid JSON array. No explanation outside the JSON.

Output format:
[
  {
    "company": "Company Name",
    "person": "Full Name",
    "score": 82,
    "reasons": ["CTO title is exact match", "HN post about Series B funding", "Tech stack includes React"]
  }
]"""


def _build_context(state: GraphState) -> str:
    """Serialises the relevant GraphState fields into a compact context string for the LLM."""
    companies = state.get("companies", [])
    people    = state.get("people", [])
    signals   = state.get("signals", [])
    query     = state.get("query", "")

    lines = [f"Original query: {query}\n"]

    lines.append("=== COMPANIES ===")
    for c in companies:
        lines.append(
            f"- {c.get('name')} | industry: {c.get('industry')} | "
            f"revenue: {c.get('revenue')} | confidence: {c.get('confidence')} | "
            f"tech: {', '.join(c.get('tech_stack', []))}"
        )

    lines.append("\n=== PEOPLE ===")
    for p in people:
        lines.append(
            f"- {p.get('name')} @ {p.get('company')} | title: {p.get('title')} | "
            f"title_score: {p.get('title_score')} | source: {p.get('source')}"
        )

    lines.append("\n=== SIGNALS (high-strength first) ===")
    for s in signals[:20]:   # cap at 20 signals to keep context manageable
        lines.append(
            f"- [{s.get('strength')}] {s.get('company')}: {s.get('signal')} ({s.get('source')})"
        )

    return "\n".join(lines)


def _parse_scores(raw: str) -> list[dict]:
    """Extracts the JSON array from the LLM response and validates shape."""
    match = re.search(r'\[.*\]', raw, re.DOTALL)   # greedy — captures full outer array, not first nested one
    if not match:
        return []
    try:
        items = json.loads(match.group())
    except (json.JSONDecodeError, ValueError):
        return []

    results = []
    for item in items:
        if not isinstance(item, dict):
            continue
        results.append({
            "company": str(item.get("company", "")),
            "person" : str(item.get("person", "")),
            "score"  : int(item.get("score", 0)),
            "reasons": [str(r) for r in item.get("reasons", [])],
        })
    return results


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

async def lead_scoring(state: GraphState) -> dict:
    """
    Lead Scoring Agent — reads companies, people, and signals from GraphState
    and asks Groq llama-3.3-70b-versatile to score each (company, person) pair 0–100.
    IMPORTANT: This agent has NO tools. It cannot call any external service
    or look up new information. It reasons only over the data already in state.
    Writes list[LeadScore] to GraphState. Never raises.
    """
    errors = list(state.get("errors", []))

    if not state.get("people"):
        logger.info("lead_scoring: no people in state — skipping to avoid hallucinated scores.")
        return {"lead_score": [], "errors": errors}

    context = _build_context(state)

    try:
        response = await _get_llm().ainvoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ])
        scores = _parse_scores(response.content)
    except Exception as e:
        errors.append(f"lead_scoring: LLM call failed — {e}")
        return {"lead_score": [], "errors": errors}

    if not scores:
        errors.append("lead_scoring: could not parse scores from LLM response.")
        logger.warning("lead_scoring: raw LLM output: %s", response.content[:500] if 'response' in dir() else "N/A")

    # Sort by score descending
    scores.sort(key=lambda s: s.get("score", 0), reverse=True)

    logger.info("lead_scoring: scored %d leads.", len(scores))

    return {
        "lead_score": scores,
        "errors"    : errors,
        "status"    : "scored",
    }
