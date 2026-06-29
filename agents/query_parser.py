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
    is_lead_gen_query      : bool                       = Field(..., description="True if this is a valid business lead generation request")
    needs_clarification    : bool                       = Field(..., description="True if the query is valid but too vague to act on")
    clarification_ask      : str | None                 = Field(None, description="Question to ask the user when needs_clarification is True")
    rejection_reason       : str | None                 = Field(None, description="Polite explanation when is_lead_gen_query is False")
    company_filters        : CompanyFiltersModel | None = Field(None, description="Structured company search criteria")
    signal_hints           : list[str]                  = Field(default_factory=list, description="Behavioral signals to look for e.g. ['hiring devops', 'cloud cost issues']")
    target_role            : str | None                 = Field(None, description="Decision maker role to find e.g. 'CTO', 'VP Engineering', 'HR Head'")
    agents_needed          : list[str]                  = Field(default_factory=list, description="Agents to activate: any of ['company_search', 'signal_filter', 'people_finder']")
    company_named_directly : bool                       = Field(False, description="True when the user names a specific company (e.g. 'Find CTO at Razorpay'). Skips company discovery.")
    named_company          : str | None                 = Field(None, description="The exact company name when company_named_directly is True")

    @field_validator("signal_hints", "agents_needed", mode="before")
    @classmethod
    def coerce_null_to_list(cls, v):
        return v if isinstance(v, list) else []


# ---------------------------------------------------------------------------
# LLMs — heavy (70b) primary; light (8b) fallback when 70b hits daily quota
# ---------------------------------------------------------------------------

_chain_heavy = None
_chain_light = None


def _get_chain_heavy():
    global _chain_heavy
    if _chain_heavy is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )
        _chain_heavy = llm.with_structured_output(QueryPlanModel, method="json_mode")
    return _chain_heavy


def _get_chain_light():
    global _chain_light
    if _chain_light is None:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )
        _chain_light = llm.with_structured_output(QueryPlanModel, method="json_mode")
    return _chain_light

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the query parser for a lead generation system.
Your only job is to extract a structured QueryPlan from the user's natural language query.

## What counts as a valid lead-gen query
A query is valid if the user wants to find business contacts — companies, decision makers,
or people at companies — for sales, outreach, or business development purposes.
This includes ANY type of company: B2B software, retail chains, hospitals, restaurants,
manufacturing, consumer goods, hospitality, media, real estate — if someone wants to
find a decision maker or company contact, it is a valid lead-gen query.

Valid examples:
- "Find CTOs of fintech startups in Bangalore"
- "Get me HR heads at logistics companies in Germany with 200+ employees"
- "Find the Head of Procurement at retail chains in Hyderabad"
- "CMO at FMCG companies in Mumbai"
- "Who manages operations at hospital chains in Telangana"
- "Find founders of e-commerce companies struggling with payment integration"
- "Who is the VP Engineering at Razorpay"

Invalid examples (set is_lead_gen_query=False):
- "Find me a Python developer job" → job search, not lead gen
- "What is the weather in Berlin" → completely unrelated
- "Write me a blog post" → content task, not lead gen
- "Find freelance designers" → freelancer search, not company contact
- "Show me restaurants near me" → consumer search, no decision-maker intent

## When to ask for clarification (needs_clarification=True)
Ask when the query is a valid lead-gen intent but lacks enough detail to search:
- "Find me some companies" → valid intent, but what industry? what role? where?
- "Get contacts from startups" → valid intent, but too vague

Do NOT ask for clarification if the query has enough to start searching.

When needs_clarification is True, set clarification_ask to a message that:
1. Echoes what you DID understand (e.g. "You're looking for startup contacts")
2. Lists ONLY the specific missing pieces (industry? role? location?)
3. Gives a concrete example of a complete query

Example: If the user says "find startup contacts", set clarification_ask to:
"You're looking for contacts at startups — here's what I still need:
- Industry (e.g. fintech, SaaS, healthcare, logistics)
- Decision-maker role (e.g. CTO, VP Sales, Founder — or leave it and I'll target the top executives)
- Location (e.g. Bangalore, Germany, US)
Try: \"Find CTOs at fintech startups in Bangalore\""

## company_named_directly
Set company_named_directly=True when the user names a SPECIFIC company (e.g. "Find CTO at Razorpay",
"Who leads engineering at Stripe", "Get me the VP Sales at SAP").
Set named_company to the exact company name in those cases.
When company_named_directly=True, set agents_needed=["people_finder"] only — company_search is skipped.

## agents_needed rules
When needs_clarification is false:
- If company_named_directly=True: agents_needed = ["people_finder"] (company_search skipped)
- Otherwise: always include BOTH "company_search" AND "people_finder"
Also include "signal_filter" when the query mentions a behavioral signal (hiring, cloud costs,
recent funding, struggling with something).
Result: named-company queries → ["people_finder"]; category queries → ["company_search", "people_finder"].

## signal_hints
Extract implied behavioral signals from the query:
- "struggling with Kubernetes costs" → ["kubernetes cost issues", "cloud cost optimization"]
- "looking for payment integration" → ["payment gateway problems", "seeking payment solution"]
- "Series B startups" → ["Series B", "recently funded", "growing team"]

## company_filters
ALWAYS fill these fields from the query (never leave all of them null):
- industry: the industry or sector (e.g. "fintech", "logistics", "SaaS", "PLM")
- location: the geographic location mentioned (e.g. "Bangalore", "Germany", "USA")
- company_size: size if stated (e.g. "startup", "enterprise", "200+ employees")
- keywords: 2–4 key search terms derived from the query

Examples:
- "Find CTOs of fintech startups in Bangalore"
  → {industry: "fintech", location: "Bangalore", company_size: "startup", keywords: ["fintech", "startup", "Bangalore"]}
- "HR heads at logistics companies in Germany with 200+ employees"
  → {industry: "logistics", location: "Germany", company_size: "200+ employees", keywords: ["logistics", "supply chain"]}
- "who is the VP Engineering at Razorpay"
  → {industry: "fintech", location: "India", keywords: ["Razorpay", "fintech", "payments"]}

## target_role
Expand the role to cover equivalent titles:
- "CTO" → "CTO / Chief Technology Officer / VP Engineering / Head of Technology"
- "HR head" → "HR Head / CHRO / VP People / Head of Human Resources"
- "founder" → "Founder / Co-founder / CEO / Managing Director"

If the query explicitly names a role, use that role (expanded).
If no role is mentioned but needs_clarification is false, infer the most relevant decision-maker
for the specific industry and context in the query. Think about who actually makes purchasing
decisions in that sector — it varies by industry. Use your knowledge across all industries,
not a fixed formula. Format as "Title1 OR Title2 OR Title3".
Only set target_role to null when needs_clarification is true (query is too vague to act on).

Return only valid JSON matching the QueryPlan schema. No explanation, no preamble, no markdown."""


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

import logging as _logging
_qp_logger = _logging.getLogger(__name__)


async def query_parser(state: GraphState) -> dict:
    query = state["query"]
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=query)]

    try:
        # Primary: 70b for best JSON accuracy
        try:
            result: QueryPlanModel = await _get_chain_heavy().ainvoke(messages)
        except Exception as primary_err:
            err_str = str(primary_err)
            if "429" in err_str or "rate_limit" in err_str.lower() or "TPD" in err_str or "tokens per day" in err_str.lower():
                _qp_logger.warning("query_parser: 70b quota hit — falling back to 8b-instant")
                result = await _get_chain_light().ainvoke(messages)
            else:
                raise

        query_plan = {
            "is_lead_gen_query"      : result.is_lead_gen_query,
            "needs_clarification"    : result.needs_clarification,
            "clarification_ask"      : result.clarification_ask,
            "rejection_reason"       : result.rejection_reason,
            "company_filters"        : result.company_filters.model_dump() if result.company_filters else None,
            "signal_hints"           : result.signal_hints,
            "target_role"            : result.target_role,
            "agents_needed"          : result.agents_needed,
            "company_named_directly" : result.company_named_directly,
            "named_company"          : result.named_company,
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
