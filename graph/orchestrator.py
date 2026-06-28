from langgraph.graph import StateGraph, END

from graph.state import GraphState
from graph.router import guard_router, agent_router
from agents.query_parser import query_parser
from agents.clarification import clarification_node
from agents.company_search import company_search
from agents.people_finder import people_finder
from agents.contact_enricher import contact_enricher
from agents.result_formatter import result_formatter

# Signal Filter and Lead Scoring are imported lazily inside build_graph()
# to avoid circular imports and keep startup fast.


def _after_company_search(state: GraphState) -> str:
    """
    Edge after company_search: if people_finder is also needed, go there.
    Otherwise jump straight to lead_scoring.
    """
    agents_needed = state.get("query_plan", {}).get("agents_needed", [])
    if "people_finder" in agents_needed:
        return "people_finder"
    if "signal_filter" in agents_needed:
        return "signal_filter"
    return "lead_scoring"


def _after_signal_filter(state: GraphState) -> str:
    """Edge after signal_filter: go to people_finder if needed, else lead_scoring."""
    agents_needed = state.get("query_plan", {}).get("agents_needed", [])
    if "people_finder" in agents_needed:
        return "people_finder"
    return "lead_scoring"


def build_graph() -> StateGraph:
    """
    Builds and compiles the full LangGraph StateGraph.
    Returns the compiled graph ready for .ainvoke().
    """
    from agents.signal_filter import signal_filter
    from agents.lead_scoring import lead_scoring

    graph = StateGraph(GraphState)

    # ── Register all nodes ──────────────────────────────────────────────────
    graph.add_node("query_parser",       query_parser)
    graph.add_node("clarification_node", clarification_node)
    graph.add_node("company_search",     company_search)
    graph.add_node("signal_filter",      signal_filter)
    graph.add_node("people_finder",      people_finder)
    graph.add_node("lead_scoring",       lead_scoring)
    graph.add_node("contact_enricher",   contact_enricher)
    graph.add_node("result_formatter",   result_formatter)

    # ── Entry point ─────────────────────────────────────────────────────────
    graph.set_entry_point("query_parser")

    # ── query_parser → guard_router (conditional) ───────────────────────────
    graph.add_conditional_edges(
        "query_parser",
        guard_router,
        {
            "clarification_node": "clarification_node",
            "agent_router"      : "agent_router_node",   # virtual — resolved below
            "result_formatter"  : "result_formatter",
        },
    )

    # LangGraph requires a real node for conditional targets.
    # agent_router is a function, not a node — we use a pass-through node.
    async def _agent_router_node(state: GraphState) -> dict:
        return {}   # no-op; the conditional edge below does the routing

    graph.add_node("agent_router_node", _agent_router_node)
    graph.add_conditional_edges(
        "agent_router_node",
        agent_router,
        {
            "company_search" : "company_search",
            "signal_filter"  : "signal_filter",
            "people_finder"  : "people_finder",
            "result_formatter": "result_formatter",
        },
    )

    # ── Agent chain edges ────────────────────────────────────────────────────
    graph.add_conditional_edges(
        "company_search",
        _after_company_search,
        {
            "people_finder" : "people_finder",
            "signal_filter" : "signal_filter",
            "lead_scoring"  : "lead_scoring",
        },
    )

    graph.add_conditional_edges(
        "signal_filter",
        _after_signal_filter,
        {
            "people_finder": "people_finder",
            "lead_scoring" : "lead_scoring",
        },
    )

    graph.add_edge("people_finder",    "lead_scoring")
    graph.add_edge("lead_scoring",     "contact_enricher")
    graph.add_edge("contact_enricher", "result_formatter")

    # ── Terminal edges ───────────────────────────────────────────────────────
    graph.add_edge("clarification_node", END)
    graph.add_edge("result_formatter",   END)

    return graph.compile()


# Module-level compiled graph — import this in api/main.py
app_graph = build_graph()
