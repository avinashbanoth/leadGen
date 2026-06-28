import os
from pydantic import BaseModel, Field, field_validator
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import GraphState


# ---------------------------------------------------------------------------
# Pydantic models — mirrors QueryPlan TypedDict but Pydantic for structured output
# ---------------------------------------------------------------------------

class CompanyFiltersModel(BaseModel):
    industry     : str | None       = Field(None, description="Industry sector e.g. 'fintech', 'logistics', 'SaaS'")
    keywords     : list[str]        = Field(default_factory=list, description="Search keywords derived from the query")
    revenue_min  : str | None       = Field(None, description="Minimum revenue e.g. '$5M', '€10M'")
    location     : str | None       = Field(None, description="Geographic location e.g. 'Germany', 'Bangalore'")
    company_size : str | None       = Field(None, description="Company size e.g. 'startup', '200+ employees', 'enterprise'")
    tech_stack   : list[str]        = Field(default_factory=list, description="Specific technologies mentioned e.g. ['AWS', 'Kubernetes']")


class QueryPlanModel(BaseModel):
    is_lead_gen_query   : bool                  = Field(..., description="True if this is a valid business lead generation request")
    needs_clarification : bool                  = Field(..., description="True if the query is valid but too vague to act on")
    clarification_ask   : str | None            = Field(None, description="Question to ask the user when needs_clarification is True")
    rejection_reason    : str | None            = Field(None, description="Polite explanation when is_lead_gen_query is False")
    company_filters     : CompanyFiltersModel | None = Field(None, description="Structured company search criteria")
    signal_hints        : list[str]             = Field(default_factory=list, description="Behavioral signals to look for e.g. ['hiring devops', 'cloud cost issues']")
    target_role         : str | None            = Field(None, description="Decision maker role to find e.g. 'CTO', 'VP Engineering', 'HR Head'")
    agents_needed       : list[str]             = Field(default_factory=list, description="Agents to activate: any of ['company_search', 'signal_filter', 'people_finder']")

    @field_validator("signal_hints", "agents_needed", mode="before")
    @classmethod
    def coerce_null_to_list(cls, v):
        return v if isinstance(v, list) else []


# ---------------------------------------------------------------------------
# LLM — lazy init so the module can be imported without GOOGLE_API_KEY set
# ---------------------------------------------------------------------------

_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )
        _chain = llm.with_structured_output(QueryPlanModel, method="json_mode")
    return _chain

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the query parser for a B2B lead generation system.
Your only job is to extract a structured QueryPlan from the user's natural language query.

## What counts as a valid lead-gen query
A query is valid if the user wants to find business contacts — companies, decision makers,
or people at companies — for sales, outreach, or business development purposes.

Valid examples:
- "Find CTOs of fintech startups in Bangalore"
- "Get me HR heads at logistics companies in Germany with 200+ employees"
- "Find founders of e-commerce companies struggling with payment integration"
- "Who is the VP Engineering at Razorpay"

Invalid examples (set is_lead_gen_query=False):
- "Find me a Python developer job" → job search, not lead gen
- "What is the weather in Berlin" → completely unrelated
- "Write me a blog post" → content task, not lead gen
- "Find freelance designers" → freelancer search, not B2B lead gen

## When to ask for clarification (needs_clarification=True)
Ask when the query is a valid lead-gen intent but lacks enough detail to search:
- "Find me some companies" → valid intent, but what industry? what role? where?
- "Get contacts from startups" → valid intent, but too vague

Do NOT ask for clarification if the query has enough to start searching.

## agents_needed rules
When needs_clarification is false, always include BOTH of:
- "company_search" — finds and verifies companies so people_finder has domains to work with.
  Even when a company is named directly ("Razorpay"), company_search must still run to populate
  company state with domain, confidence, and metadata.
- "people_finder" — finds the decision makers. This system always finds people.
Also include "signal_filter" when the query mentions a behavioral signal (hiring, cloud costs,
recent funding, struggling with something).
Result: almost every valid query has agents_needed = ["company_search", "people_finder"] or
["company_search", "signal_filter", "people_finder"].

## signal_hints
Extract implied behavioral signals from the query:
- "struggling with Kubernetes costs" → ["kubernetes cost issues", "cloud cost optimization"]
- "looking for payment integration" → ["payment gateway problems", "seeking payment solution"]
- "Series B startups" → ["Series B", "recently funded", "growing team"]

## target_role
Expand the role to cover equivalent titles:
- "CTO" → "CTO / Chief Technology Officer / VP Engineering / Head of Technology"
- "HR head" → "HR Head / CHRO / VP People / Head of Human Resources"
- "founder" → "Founder / Co-founder / CEO / Managing Director"

If the query explicitly names a role, use that role (expanded).
If no role is mentioned but needs_clarification is false, infer the most likely decision-maker
for that industry:
- Tech / software / SaaS / PLM / ERP / cloud → "CTO OR VP Engineering OR Head of Technology OR IT Director"
- Fintech / payments / banking → "CTO OR VP Engineering OR Head of Product"
- E-commerce / retail → "CTO OR Head of Technology OR VP Product"
- Logistics / supply chain / manufacturing → "VP Operations OR Head of Supply Chain OR CTO"
- Healthcare / pharma → "CIO OR VP Technology OR Head of IT"
- General or unknown industry → "CEO OR Founder OR Managing Director OR CTO"
Only set target_role to null when needs_clarification is true (query is too vague to act on).

Return only valid JSON matching the QueryPlan schema. No explanation, no preamble, no markdown."""


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

async def query_parser(state: GraphState) -> dict:
    query = state["query"]

    try:
        result: QueryPlanModel = await _get_chain().ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=query),
        ])

        query_plan = {
            "is_lead_gen_query"  : result.is_lead_gen_query,
            "needs_clarification": result.needs_clarification,
            "clarification_ask"  : result.clarification_ask,
            "rejection_reason"   : result.rejection_reason,
            "company_filters"    : result.company_filters.model_dump() if result.company_filters else None,
            "signal_hints"       : result.signal_hints,
            "target_role"        : result.target_role,
            "agents_needed"      : result.agents_needed,
        }

    except Exception as e:
        query_plan = {
            "is_lead_gen_query"  : False,
            "needs_clarification": False,
            "clarification_ask"  : None,
            "rejection_reason"   : "Query parsing failed — please try again.",
            "company_filters"    : None,
            "signal_hints"       : [],
            "target_role"        : None,
            "agents_needed"      : [],
        }
        return {
            "query_plan": query_plan,
            "errors"    : state.get("errors", []) + [f"query_parser failed: {str(e)}"],
        }

    return {"query_plan": query_plan}
