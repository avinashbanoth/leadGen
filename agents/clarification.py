
from graph.state import GraphState

FALLBACK_QUESTION = (
    "Could you give me a bit more detail? "
    "What industry or type of company are you targeting? "
    "What role are you looking for? "
    "Any location or company size preference?"
)


async def clarification_node(state: GraphState) -> dict:
    """
    Surfaces the clarification question Gemini wrote into query_plan.
    No LLM call — the question is already in query_plan.clarification_ask.
    Sets status to awaiting_clarification so the API layer knows to pause.
    """
    plan = state.get("query_plan", {})
    question = plan.get("clarification_ask") or FALLBACK_QUESTION

    message = f"I need a bit more information to find the right leads.\n\n{question}"

    return {
        "status"  : "awaiting_clarification",
        "messages": state.get("messages", []) + [message],
    }
