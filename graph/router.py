from graph.state import GraphState


def guard_router(state: GraphState) -> str:
    """
    First conditional_edge — runs after query_parser.
    Reads query_plan and routes to one of three paths.
    """
    plan = state.get("query_plan")

    if not plan:
        return "result_formatter"

    if not plan.get("is_lead_gen_query", False):
        return "result_formatter"

    if plan.get("needs_clarification", False):
        return "clarification_node"

    return "agent_router"


def agent_router(state: GraphState) -> str:
    """
    Second conditional_edge — runs after guard passes.
    Reads agents_needed from query_plan and returns the first agent to activate.
    The orchestrator chains remaining agents after each one completes.

    Priority: company_search → signal_filter → people_finder
    """
    plan = state.get("query_plan", {})
    agents_needed = plan.get("agents_needed", [])

    if "company_search" in agents_needed:
        return "company_search"

    if "signal_filter" in agents_needed:
        return "signal_filter"

    if "people_finder" in agents_needed:
        return "people_finder"

    return "result_formatter"
