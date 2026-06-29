from graph.state import GraphState

_FALLBACK_QUESTION = (
    "What industry or type of company are you targeting? "
    "What role are you looking for (or should I find the top decision makers)? "
    "Any location preference?"
)


async def clarification_node(state: GraphState) -> dict:
    """
    Surfaces the clarification question written by query_parser into query_plan.
    No LLM call here — the question is already in query_plan.clarification_ask.
    Sets status to awaiting_clarification so the API layer knows to pause.
    """
    plan     = state.get("query_plan", {})
    question = plan.get("clarification_ask") or _FALLBACK_QUESTION

    message = (
        f"I need a bit more detail to find the right leads.\n\n"
        f"{question}\n\n"
        f"You can also just rephrase in one line, e.g.:\n"
        f"  \"Find CTOs at fintech startups in Bangalore\"\n"
        f"  \"VP Sales at SaaS companies in Germany with 200+ employees\"\n"
        f"  \"founders of e-commerce companies struggling with payment integration\""
    )

    return {
        "status"  : "awaiting_clarification",
        "messages": state.get("messages", []) + [message],
    }
