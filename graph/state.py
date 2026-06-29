from typing import TypedDict


class CompanyFilters(TypedDict):
    industry     : str | None
    keywords     : list[str]
    revenue_min  : str | None
    location     : str | None
    company_size : str | None
    tech_stack   : list[str]


class QueryPlan(TypedDict):
    is_lead_gen_query      : bool
    needs_clarification    : bool
    clarification_ask      : str | None
    rejection_reason       : str | None
    company_filters        : CompanyFilters | None
    signal_hints           : list[str]
    target_role            : str | None
    agents_needed          : list[str]
    company_named_directly : bool       # True → skip company_search, go straight to people_finder
    named_company          : str | None # the exact company name when company_named_directly=True


class CompanyData(TypedDict):
    name       : str
    website    : str
    industry   : str
    revenue    : str
    confidence : float
    tech_stack : list[str]
    source     : str


class PersonData(TypedDict):
    name         : str
    title        : str
    title_score  : float
    title_tier   : int          # 1 = C-suite/VP (Level 1), 2 = Director/Manager fallback
    company      : str
    linkedin_url : str
    email        : str | None
    phone        : str | None
    source       : str


class SignalData(TypedDict):
    company  : str
    signal   : str
    source   : str
    strength : str
    url      : str


class LeadScore(TypedDict):
    company : str
    person  : str
    score   : int
    reasons : list[str]


class ContactData(TypedDict):
    name       : str
    title      : str
    title_tier : int          # 1 = C-suite/VP (Level 1), 2 = Director/Manager fallback
    company    : str
    email      : str | None
    confidence : int
    linkedin   : str
    phone      : str | None
    score      : int
    status     : str
    tried      : list[str]
    suggestion : str | None


class GraphState(TypedDict):
    query      : str
    query_plan : QueryPlan
    companies  : list[CompanyData]
    people     : list[PersonData]
    signals    : list[SignalData]
    lead_score : list[LeadScore]
    contacts   : list[ContactData]
    errors     : list[str]
    messages   : list[str]
    status     : str
