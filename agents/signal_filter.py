import asyncio
import logging

from graph.state import GraphState
from tools.reddit_tool import search_reddit_signals
from tools.hn_tool import search_hn_signals
from tools.github_tool import search_github_signals
from tools.wappalyzer_tool import detect_tech_stack

logger = logging.getLogger(__name__)


async def _gather_signals_for_company(
    company: dict,
    keywords: list[str],
) -> list[dict]:
    """Runs all 4 signal sources concurrently for a single company."""
    company_name = company.get("name", "")
    website      = company.get("website", "")

    tasks = [
        search_reddit_signals.ainvoke({"company_name": company_name, "keywords": keywords, "max_results": 5}),
        search_hn_signals.ainvoke({"company_name": company_name, "keywords": keywords, "max_results": 5}),
        search_github_signals.ainvoke({"company_name": company_name, "keywords": keywords, "max_results": 3}),
    ]

    # Wappalyzer only if we have a website
    if website:
        tasks.append(detect_tech_stack.ainvoke({"website_url": website, "company_name": company_name}))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    signals: list[dict] = []
    for result in results:
        if isinstance(result, Exception):
            logger.warning("signal_filter: one source failed for '%s' — %s", company_name, result)
            continue
        if isinstance(result, list):
            signals.extend(result)
        # Wappalyzer returns a dict — convert tech_stack to a signal entry
        elif isinstance(result, dict) and result.get("tech_stack"):
            tech_list = ", ".join(result["tech_stack"][:10])
            signals.append({
                "company" : company_name,
                "signal"  : f"Tech stack detected: {tech_list}",
                "source"  : "wappalyzer",
                "strength": "medium",
                "url"     : website,
            })

    return signals


async def signal_filter(state: GraphState) -> dict:
    """
    Signal Filter Agent — gathers buying/growth signals for all companies in GraphState
    from Reddit, Hacker News, GitHub, and Wappalyzer, running all sources concurrently.
    Writes list[SignalData] to GraphState. High-strength signals surface in Lead Scoring.
    Never raises — writes to errors on failure.
    """
    errors    = list(state.get("errors", []))
    companies = state.get("companies", [])
    query_plan = state.get("query_plan", {})

    # Use signal_hints from QueryPlan as extra keywords for signal searches
    signal_hints = query_plan.get("signal_hints", [])
    keywords = signal_hints or []

    if not companies:
        errors.append("signal_filter: no companies in state — skipping.")
        return {"signals": [], "errors": errors}

    tasks = [_gather_signals_for_company(c, keywords) for c in companies]

    try:
        batches = await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        errors.append(f"signal_filter: gather failed — {e}")
        return {"signals": [], "errors": errors}

    all_signals: list[dict] = []
    for batch in batches:
        if isinstance(batch, Exception):
            errors.append(f"signal_filter: company batch failed — {batch}")
            continue
        all_signals.extend(batch)

    # Sort: high > medium > low
    _strength_order = {"high": 0, "medium": 1, "low": 2}
    all_signals.sort(key=lambda s: _strength_order.get(s.get("strength", "low"), 2))

    logger.info("signal_filter: collected %d signals across %d companies.", len(all_signals), len(companies))

    return {
        "signals": all_signals,
        "errors" : errors,
        "status" : "signals_found" if all_signals else "no_signals_found",
    }
