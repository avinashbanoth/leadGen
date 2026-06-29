"""
Tests for graph/router.py — guard_router and agent_router.

Both functions are pure Python (no I/O, no LLM, no GraphState mutation),
so every test is synchronous and needs no mocking.
"""
import pytest

from graph.router import agent_router, guard_router


# ---------------------------------------------------------------------------
# Helpers — build minimal valid dicts without importing TypedDicts
# ---------------------------------------------------------------------------

def _plan(**overrides) -> dict:
    """Return a minimal valid QueryPlan dict with the given overrides applied."""
    base = {
        "is_lead_gen_query"     : True,
        "needs_clarification"   : False,
        "clarification_ask"     : None,
        "rejection_reason"      : None,
        "company_filters"       : None,
        "signal_hints"          : [],
        "target_role"           : None,
        "agents_needed"         : ["company_search", "people_finder"],
        "company_named_directly": False,
        "named_company"         : None,
    }
    base.update(overrides)
    return base


def _state(query_plan=None, **overrides) -> dict:
    """Return a minimal valid GraphState dict."""
    base = {
        "query"      : "find CTOs at fintech startups in Bangalore",
        "query_plan" : query_plan,
        "companies"  : [],
        "people"     : [],
        "signals"    : [],
        "lead_score" : [],
        "contacts"   : [],
        "errors"     : [],
        "messages"   : [],
        "status"     : "",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# guard_router
# ---------------------------------------------------------------------------

class TestGuardRouter:
    def test_missing_plan_routes_to_formatter(self):
        """No query_plan key at all → reject path."""
        assert guard_router(_state(query_plan=None)) == "result_formatter"

    def test_empty_plan_routes_to_formatter(self):
        """Empty dict — is_lead_gen_query absent → defaults to False → reject."""
        assert guard_router(_state(query_plan={})) == "result_formatter"

    def test_non_lead_gen_routes_to_formatter(self):
        """Explicit rejection flag → result_formatter."""
        plan = _plan(is_lead_gen_query=False, rejection_reason="This is a job search query.")
        assert guard_router(_state(plan)) == "result_formatter"

    def test_rejection_takes_priority_over_clarification(self):
        """is_lead_gen_query=False wins even when needs_clarification=True."""
        plan = _plan(is_lead_gen_query=False, needs_clarification=True)
        assert guard_router(_state(plan)) == "result_formatter"

    def test_needs_clarification_routes_to_clarification_node(self):
        """Valid lead-gen query but missing detail → ask the user."""
        plan = _plan(needs_clarification=True, clarification_ask="What industry?")
        assert guard_router(_state(plan)) == "clarification_node"

    def test_valid_query_routes_to_agent_router(self):
        """Fully specified lead-gen query → proceed to agents."""
        assert guard_router(_state(_plan())) == "agent_router"

    def test_valid_query_with_all_fields_populated(self):
        """Realistic complete plan still routes to agent_router."""
        plan = _plan(
            company_filters={
                "industry": "fintech", "location": "Bangalore",
                "keywords": ["payments", "banking"], "revenue_min": None,
                "company_size": "startup", "tech_stack": ["AWS"],
            },
            signal_hints=["hiring engineers", "Series A"],
            target_role="CTO",
            agents_needed=["company_search", "people_finder"],
        )
        assert guard_router(_state(plan)) == "agent_router"

    def test_weather_query_rejected(self):
        """Non-lead-gen query (guard router sets is_lead_gen_query=False)."""
        plan = _plan(
            is_lead_gen_query=False,
            rejection_reason="Weather questions are not lead generation queries.",
        )
        assert guard_router(_state(plan)) == "result_formatter"


# ---------------------------------------------------------------------------
# agent_router
# ---------------------------------------------------------------------------

class TestAgentRouter:
    def test_company_search_first_in_priority(self):
        """company_search always wins when present, regardless of list order."""
        plan = _plan(agents_needed=["company_search", "people_finder"])
        assert agent_router(_state(plan)) == "company_search"

    def test_company_search_priority_over_signal_filter(self):
        """company_search beats signal_filter even when signal_filter is listed first."""
        plan = _plan(agents_needed=["signal_filter", "company_search", "people_finder"])
        assert agent_router(_state(plan)) == "company_search"

    def test_signal_filter_second_priority(self):
        """With no company_search, signal_filter wins."""
        plan = _plan(agents_needed=["signal_filter", "people_finder"])
        assert agent_router(_state(plan)) == "signal_filter"

    def test_signal_filter_priority_over_people_finder(self):
        plan = _plan(agents_needed=["people_finder", "signal_filter"])
        assert agent_router(_state(plan)) == "signal_filter"

    def test_people_finder_alone(self):
        """Named-company short-circuit path — only people_finder needed."""
        plan = _plan(
            agents_needed=["people_finder"],
            company_named_directly=True,
            named_company="Infosys",
        )
        assert agent_router(_state(plan)) == "people_finder"

    def test_empty_agents_routes_to_formatter(self):
        """Nothing to run → skip to output."""
        plan = _plan(agents_needed=[])
        assert agent_router(_state(plan)) == "result_formatter"

    def test_unknown_agent_routes_to_formatter(self):
        """Unrecognised agent name → safe fallback to formatter."""
        plan = _plan(agents_needed=["unknown_future_agent"])
        assert agent_router(_state(plan)) == "result_formatter"

    def test_missing_agents_needed_key_routes_to_formatter(self):
        """agents_needed missing from plan entirely → safe fallback."""
        plan = {k: v for k, v in _plan().items() if k != "agents_needed"}
        assert agent_router(_state(plan)) == "result_formatter"

    def test_all_three_agents_respects_priority(self):
        """All three listed → company_search runs first."""
        plan = _plan(agents_needed=["people_finder", "signal_filter", "company_search"])
        assert agent_router(_state(plan)) == "company_search"
