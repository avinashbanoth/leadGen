# Lead Generation Multi-Agent System — POC Architecture

> **Status:** Active — architecture pivot complete (June 2026)
> **Author:** Avinash
> **Stack:** Python 3.11+ · LangChain · LangGraph · FastAPI · Groq · Apollo.io
> **Goal:** Generic, production-grade contact finder via natural language chat
> **People stack:** Apollo.io (Layer A) → Website team pages (Layer B) → Google dorks (Layer C)

---

## 1. Problem Statement

The goal of this system is one thing: **get contacts**. Given any natural language query about any type of company or person, the system finds verified contact details — email, LinkedIn URL, phone where available — of the right decision makers.

The system is fully generic. No industry, role, signal, or company type is hardcoded anywhere. The user's query drives everything dynamically.

### What "Contacts" Means as Output

Every run of the system aims to produce this:

```json
{
  "name"        : "John Smith",
  "title"       : "CTO",
  "company"     : "Acme Corp",
  "email"       : "j.smith@acme.com",
  "confidence"  : 94,
  "linkedin"    : "linkedin.com/in/johnsmith",
  "phone"       : "+91-XXXXXXXXXX",
  "score"       : 87,
  "reasons"     : ["Uses AWS", "Hiring DevOps engineers", "Revenue ~$12M"],
  "status"      : "verified",
  "tried"       : ["hunter", "permutator", "harvester", "google_dork"]
}
```

If a contact is partial (email not found), the system still returns what it found with a clear status — never silently fails.

### Example Queries the System Handles

| Query | What gets activated |
|---|---|
| "Find CTOs of logistics companies in Germany with 200+ employees" | Company Search + People Finder |
| "Who is struggling with Kubernetes costs and needs DevOps help" | Company Search + Signal Filter + People Finder |
| "Get me HR heads of Series B fintech startups in Bangalore" | Company Search + People Finder |
| "Find founders of e-commerce companies looking for payment integration" | Signal Filter + People Finder |
| "Find freelance Python jobs" | Guard Router rejects — not lead gen |
| "What is the weather today" | Guard Router rejects — not lead gen |

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      CHAT INTERFACE                             │
│               FastAPI backend + React frontend                  │
└────────────────────────┬────────────────────────────────────────┘
                         │ user query (natural language)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LANGGRAPH STATEGRAPH                          │
│                                                                 │
│  GraphState = {                                                 │
│    query        : str                                           │
│    query_plan   : QueryPlan   ← replaces hardcoded intent       │
│    companies    : list[CompanyData]                             │
│    people       : list[PersonData]                              │
│    signals      : list[SignalData]                              │
│    lead_score   : list[LeadScore]                               │
│    contacts     : list[ContactData]                             │
│    errors       : list[str]   ← every agent writes here        │
│    messages     : list[str]                                     │
│    status       : str                                           │
│  }                                                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │    Query Parser      │  ← Claude extracts structured
              │       Node           │    QueryPlan from raw query
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │    Guard Router      │  ← conditional_edge
              │  (conditional_edge)  │
              └──────────┬───────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
   Not lead gen     Needs more      Valid query
          │           info               │
          ▼              │               ▼
  result_formatter  clarification   ┌───────────────────┐
  (polite reject)      node         │  conditional_edge  │
                    (ask user)      │  routes by plan    │
                                    └──┬──────┬───────┬──┘
                                       │      │       │
                                       ▼      ▼       ▼
                                   Company  Signal  People
                                   Search   Filter  Finder
                                   Agent    Agent   Agent
                                       │      │       │
                                       └──────┴───────┘
                                               │
                                               ▼
                                      ┌─────────────────┐
                                      │  Lead Scoring   │  ← pure Claude
                                      │     Agent       │    reasoning only
                                      └────────┬────────┘
                                               │
                                               ▼
                                      ┌─────────────────┐
                                      │ Contact Enricher│  ← EmailProvider
                                      │     Agent       │    abstraction
                                      └────────┬────────┘
                                               │
                                               ▼
                                      ┌─────────────────┐
                                      │ Result Formatter│  ← final output
                                      └─────────────────┘
```

---

## 3. GraphState — Single Source of Truth

No agent passes data directly to another. All communication happens through GraphState only.

```python
from typing import TypedDict

class CompanyFilters(TypedDict):
    industry     : str | None      # "SaaS", "logistics", "fintech" — from query
    keywords     : list[str]       # ["kubernetes", "PLM", "payment gateway"]
    revenue_min  : str | None      # "$5M", "$10M" — extracted from query
    location     : str | None      # "India", "Germany", "Bangalore"
    company_size : str | None      # "startup", "200+ employees", "enterprise"
    tech_stack   : list[str]       # ["AWS", "Stripe"] — or empty

class QueryPlan(TypedDict):
    is_lead_gen_query      : bool          # False → guard router rejects
    needs_clarification    : bool          # True → ask user for more info
    clarification_ask      : str | None    # question to show user
    rejection_reason       : str | None    # why it was rejected
    company_filters        : CompanyFilters | None
    signal_hints           : list[str]     # ["hiring devops", "cloud cost issues"]
    target_role            : str | None    # "CTO", "HR Head", "VP Engineering"
    agents_needed          : list[str]     # ["company_search", "people_finder"]
    company_named_directly : bool          # True → skip company_search (Rule 10)
    named_company          : str | None    # exact company name when above is True

class CompanyData(TypedDict):
    name        : str
    website     : str
    industry    : str
    revenue     : str       # extracted value or "unknown"
    confidence  : float     # 0.0–1.0 — reliability of scraped data
    tech_stack  : list[str]
    source      : str       # which tool found this

class PersonData(TypedDict):
    name         : str
    title        : str
    title_score  : float    # 0.0–1.0 — how well title matches target_role
    title_tier   : int      # 1 = C-suite/VP (primary), 2 = Director/Manager (fallback)
    company      : str
    linkedin_url : str
    email        : str | None
    phone        : str | None
    source       : str      # "apollo" / "website_team" / "crosslinked"

class SignalData(TypedDict):
    company  : str
    signal   : str          # "hiring 3 FinOps engineers"
    source   : str          # "linkedin_jobs" / "reddit" / "github" / "hn"
    strength : str          # "high" / "medium" / "low"
    url      : str

class LeadScore(TypedDict):
    company : str
    person  : str
    score   : int           # 0–100
    reasons : list[str]     # ["Uses AWS", "Hiring DevOps", "Growing team"]

class ContactData(TypedDict):
    name       : str
    title      : str
    title_tier : int            # 1 = C-suite/VP, 2 = Director/Manager fallback
    company    : str
    email      : str | None     # None if not found
    confidence : int            # 0–100
    linkedin   : str
    phone      : str | None
    score      : int
    status     : str            # "verified" / "partial" / "not_found"
    tried      : list[str]      # which providers were attempted
    suggestion : str | None     # LinkedIn URL when email not found

class GraphState(TypedDict):
    query      : str
    query_plan : QueryPlan
    companies  : list[CompanyData]
    people     : list[PersonData]
    signals    : list[SignalData]
    lead_score : list[LeadScore]
    contacts   : list[ContactData]
    errors     : list[str]          # every agent writes failures here
    messages   : list[str]
    status     : str
```

---

## 4. Agent & Node Details

### 4.1 Query Parser Node

**What it does:** First node that runs. Takes the raw user query and uses Claude to extract a fully structured `QueryPlan`. This is what makes the system generic — no hardcoded industry, role, or signal anywhere.

**LangGraph role:** Node — reads `GraphState.query`, writes `GraphState.query_plan`

**Examples:**

```
Query: "Find HR heads of logistics companies in Germany with 200+ employees"
QueryPlan:
  is_lead_gen_query   = True
  needs_clarification = False
  company_filters:
    industry     = "logistics"
    location     = "Germany"
    company_size = "200+ employees"
    keywords     = ["logistics", "supply chain", "freight"]
  signal_hints  = ["hiring HR", "expanding operations"]
  target_role   = "HR Head / CHRO / VP People"
  agents_needed = ["company_search", "people_finder"]

---

Query: "Find freelance Python jobs"
QueryPlan:
  is_lead_gen_query = False
  rejection_reason  = "This is a job search query. This system finds
                       business contacts, not job listings."

---

Query: "Find companies"
QueryPlan:
  is_lead_gen_query   = True
  needs_clarification = True
  clarification_ask   = "Could you tell me more? What industry or type
                         of company? What role are you looking for?
                         Any location or size preference?"
```

**Implementation:** LangChain chain with Pydantic structured output parser

---

### 4.2 Guard Router (conditional_edge)

**What it does:** Reads `QueryPlan` and routes to one of three paths. This is not a node — it is a LangGraph `conditional_edge` function.

**LangGraph role:** `conditional_edge` after `query_parser`

```python
def guard_router(state: GraphState) -> str:
    plan = state["query_plan"]
    if not plan["is_lead_gen_query"]:
        return "result_formatter"      # polite rejection
    if plan["needs_clarification"]:
        return "clarification_node"    # ask user for more info
    return "agent_router"              # valid — proceed to agents
```

---

### 4.3 Clarification Node

**What it does:** When the query is valid but too vague, this node returns the `clarification_ask` question to the user and waits. The conversation resumes when the user replies with more detail.

**LangGraph role:** Node — reads `query_plan.clarification_ask`, writes to `status = "awaiting_clarification"`

---

### 4.4 Company Search Agent

**What it does:** Discovers companies matching `QueryPlan.company_filters`. All extracted data includes a confidence score — scraped revenue and employee counts are estimates, not verified facts.

**LangGraph role:** Node — reads `GraphState.query_plan`, writes `GraphState.companies`

**Tools:**

| Tool | Purpose |
|---|---|
| `SearXNG` | Self-hosted metasearch — `categories=map` when location present (business listings); `categories=general` otherwise; Groq 8b expands state/country to city list and runs one query per city |
| `Overpass API` | OpenStreetMap business data — Nominatim geocodes location to bbox; keyword-filtered OSM office/shop/craft query; fallback to all named offices; free, no auth |
| `Crawl4AI` | Scrapes company pages — revenue, funding, headcount with confidence score |
| `Wappalyzer` | Detects tech stack from company website |

**Phase 0 (location queries):** IndiaMart/JustDial + Overpass run concurrently; results merged and deduplicated.
**Phase 1A:** Known company directories (Kompass, Zaubacorp, etc.).
**Phase 1B:** SearXNG with map category + city expansion.
**Phase 2:** Verify each name → find homepage via Crawl4AI.

**Output shape:**
```json
{
  "name": "Acme Logistics GmbH",
  "website": "acme-logistics.de",
  "industry": "logistics",
  "revenue": "~€8M",
  "confidence": 0.68,
  "tech_stack": ["AWS", "SAP"],
  "source": "crunchbase_public"
}
```

---

### 4.5 Signal Filter Agent

**What it does:** Detects buying signals from public sources. Does not look for companies — looks for *evidence that a company has a problem your client can solve*. Signal hints come from `QueryPlan.signal_hints`, not hardcoded.

**LangGraph role:** Node — reads `GraphState.query_plan + companies`, writes `GraphState.signals`

**Tools:**

| Tool | What it detects |
|---|---|
| `HackerNews Algolia API` | "Ask HN" threads matching signal hints (no auth needed) |
| `GitHub Issues API` | Open issues matching signal keywords (GITHUB_TOKEN) |
| `Wappalyzer` | Tech stack presence as a signal |

> Reddit removed — IP blocked on most cloud/residential IPs. HN + GitHub cover equivalent signals.

---

### 4.6 People Finder Agent

**What it does:** Finds the right person at each company. Uses `QueryPlan.target_role` dynamically — never hardcodes any role. Three-layer cascade with two-tier role priority.

**LangGraph role:** Node — reads `GraphState.companies + query_plan.target_role`, writes `GraphState.people`

**Named-company short-circuit (Rule 10):** When `query_plan.company_named_directly = True`, company_search is skipped entirely and people_finder receives the company directly from a quick Apollo lookup.

**Role Normalizer (runs before search):**

```python
# target_role = "CTO"
# Level 1 (primary):  ["CTO", "Chief Technology Officer", "VP Engineering", "Head of Technology"]
# Level 2 (fallback): ["Director of Engineering", "Engineering Manager", "Head of Software Development"]
```

**Three-Layer Cascade:**

```
Layer A: Apollo.io People Search API (primary)
         search by company_name + target_titles
         returns name, title, LinkedIn URL; verified email when available
         → covers 270M+ professionals; large + mid-market companies
                │
                │ if Apollo returns nothing (small/private company)
                ▼
Layer B: Website Team Page Scraper
         crawls /team /about /leadership /management /people
         Groq 8b-instant extracts names + titles from raw page text
         → works for any company that lists staff online (SMEs, EU companies)
                │
                │ if no team page found
                ▼
Layer C: Google Dorks via SearXNG
         site:linkedin.com/in "title" "company"
         extracts names from result titles, generates email permutations
         → last resort; works for large companies with indexed profiles
```

**Two-tier role priority:**
- **Tier 1:** C-suite/VP titles from `expand_role()` — run all 3 layers
- **Tier 2:** Director/Manager fallback — if Tier 1 yields nothing, retry all 3 layers
- Results marked `title_tier=1` or `title_tier=2`; Tier 2 shown as `[L2 fallback]` in output

**Title relevance scoring (Groq 8b):**
```python
# target_role = "CTO", found title = "Director of Engineering"
# score = 0.65 — strong partial match, included
# threshold: ≥0.3 for Tier 1, ≥0.1 for Tier 2
```

---

### 4.7 Lead Scoring Agent

**What it does:** Pure reasoning node. No tools, no scraping. Claude reads all accumulated data in GraphState — companies, signals, people — and outputs a ranked scored list with explanations.

**LangGraph role:** Node — reads `GraphState.companies + signals + people`, writes `GraphState.lead_score`

**No external tools. Claude + GraphState only.**

**Output shape:**
```json
[
  {
    "company": "Acme Logistics GmbH",
    "person": "Klaus Weber, VP Engineering",
    "score": 87,
    "reasons": [
      "Uses AWS — cloud spend likely significant",
      "Hiring 3 DevOps engineers — active problem signal",
      "Engineering team grew 40% YoY",
      "Revenue ~€8M (confidence 0.68)"
    ]
  },
  {
    "company": "FastFreight AG",
    "person": "Anna Müller, CTO",
    "score": 61,
    "reasons": [
      "Uses GCP",
      "No active hiring signals found",
      "Revenue unknown"
    ]
  }
]
```

**LLM:** Groq `llama-3.3-70b-versatile` — reasoning task, not browsing.

**Hallucination guard:** Scorer only reasons over data already present in GraphState. It cannot call tools or look up new information. If a field is missing, it scores lower — never invents data.

---

### 4.8 Contact Enricher Agent

**What it does:** Takes scored leads and finds verified contact details. Uses `EmailProvider` abstraction — tries six levels before giving up. Always returns a result, even if partial.

**LangGraph role:** Node — reads `GraphState.lead_score + people`, writes `GraphState.contacts`

**EmailProvider abstraction — 5 levels (tried in order, stop at first ≥30% confidence):**

```python
# Level 1 — HunterProvider       hunter.io API (50 free/month)
# Level 2 — PermutatorProvider   name permutations + real MX/SMTP verify (dnspython)
# Level 3 — HarvesterProvider    OSINT sources (theHarvester)
# Level 4 — GoogleDorkProvider   "name" "email" site:domain.com via SearXNG
# Level 5 — WebsiteContactProvider  /team /contact pages via Crawl4AI (single crawler)

# Fresh provider instances created per person — no shared singleton state
```

**Fast-path:** If Apollo already returned a verified email for the person, the provider chain is skipped entirely.

**Partial result — never silent failure:**
```json
{
  "name": "Anna Müller",
  "title": "CTO",
  "company": "FastFreight AG",
  "email": null,
  "confidence": 0,
  "linkedin": "linkedin.com/in/annamuller",
  "status": "partial",
  "tried": ["hunter", "permutator", "harvester", "google_dork", "website"],
  "suggestion": "Reach out directly via LinkedIn: linkedin.com/in/annamuller"
}
```

---

### 4.9 Result Formatter Node

**What it does:** Formats the final contacts for chat interface output. Handles three output types in one response.

**Output:**
- Chat-friendly summary: top leads with scores and why
- Structured JSON: full contact list for frontend table
- Partial results clearly marked with suggestions
- Total stats: "Found 12 leads, 9 with verified emails, 3 partial"

---

## 5. Data Sources & API Keys

| Source | Purpose | Auth | Tier |
|---|---|---|---|
| Apollo.io | People search + company lookup | `APOLLO_API_KEY` | Free: unlimited search, 50 email exports/month |
| Hunter.io | Email enrichment Level 1 | `HUNTER_API_KEY` | Free: 50 lookups/month |
| SearXNG | Web search + Google dorks | None (self-hosted Docker) | Unlimited |
| Crawl4AI | Web scraping (company pages, team pages) | None | Unlimited |
| OpenStreetMap Overpass | Geographic business discovery | None | Free, unlimited |
| Nominatim | Location → bounding box geocoding | None (User-Agent header required) | Free, unlimited |
| GitHub API | Signal detection (issues, org members) | `GITHUB_TOKEN` (PAT) | Free: 5000 req/hr authenticated |
| HackerNews | Signal detection (Ask HN threads) | None | Unlimited |
| Groq | All LLM inference | `GROQ_API_KEY` | Free: 100K tokens/day (70b), higher limit for 8b |

```bash
# Setup
pip install -r requirements.txt
crawl4ai-setup                    # installs Playwright Chromium for Crawl4AI
docker-compose up -d              # starts SearXNG on port 8080
```

---

## 6. Folder Structure

```
lead-gen-agent/
│
├── CLAUDE.md                          # Project context for every Claude Code session
├── .env                               # LI_USERNAME, LI_PASSWORD, HUNTER_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY, CEREBRAS_API_KEY
├── requirements.txt
├── docker-compose.yml                 # SearXNG on port 8080
│
├── graph/
│   ├── state.py                       # GraphState + all TypedDicts
│   ├── orchestrator.py                # StateGraph — nodes, edges, routing
│   └── router.py                      # guard_router + agent_router functions
│
├── agents/
│   ├── query_parser.py                # extracts QueryPlan from raw query
│   ├── clarification.py               # asks user for more info when vague
│   ├── company_search.py              # Phase 0: IndiaMart/JustDial + Overpass (concurrent) → Phase 1A dirs → Phase 1B SearXNG → Phase 2 verify
│   ├── signal_filter.py               # Reddit + HN + GitHub + LinkedIn Jobs
│   ├── people_finder.py               # 4-layer fallback + role normalizer
│   ├── lead_scoring.py                # pure Gemini reasoning — no tools
│   ├── contact_enricher.py            # EmailProvider abstraction chain
│   └── result_formatter.py            # final output; domain dedup (1 contact/domain for company-search runs)
│
├── tools/
│   ├── searxng_tool.py                # @tool — categories=map + Groq 8b city expansion for location queries
│   ├── overpass_tool.py               # @tool — OpenStreetMap Overpass API; Nominatim geocode → bbox; business discovery
│   ├── crawl4ai_tool.py               # @tool — confidence-scored extraction
│   ├── linkedin_api_tool.py           # @tool — Voyager HTTP Layer 1
│   ├── linkedin_scraper_tool.py       # @tool — Camoufox + stealth Layer 2
│   ├── browser_use_tool.py            # @tool — LLM browser Layer 3 (Groq)
│   ├── crosslinked_tool.py            # @tool — Google dork Layer 4
│   ├── reddit_tool.py                 # @tool — PRAW official API
│   ├── hn_tool.py                     # @tool — HN Algolia API
│   ├── wappalyzer_tool.py             # @tool — tech stack detection
│   └── github_tool.py                 # @tool — GitHub Issues + org members API
│
├── providers/
│   ├── email_provider.py              # EmailProvider ABC
│   ├── hunter_provider.py             # Level 1 — Hunter.io
│   ├── permutator_provider.py         # Level 2 — pattern generation
│   ├── harvester_provider.py          # Level 3 — theHarvester OSINT
│   ├── google_dork_provider.py        # Level 4 — Google dork via SearXNG
│   ├── linkedin_contact_provider.py   # Level 5 — linkedin-api contact info
│   └── website_contact_provider.py   # Level 6 — /team /about via Crawl4AI
│
├── utils/
│   ├── human_behavior.py              # random_delay, human_type, human_scroll
│   ├── session_manager.py             # cookie save/load/warm
│   ├── rate_limiter.py                # LinkedIn-safe rate limiting
│   └── role_normalizer.py             # expands role + normalizes company name
│
├── api/
│   └── main.py                        # FastAPI — POST /chat
│
└── tests/
    ├── test_tools.py                   # one test per tool
    ├── test_guard_router.py            # test rejection + clarification paths
    └── test_graph.py                   # end-to-end with mock tools
```

---

## 7. Data Flow — End-to-End Example

**Query:** `"Find VP Engineering of fintech startups in Bangalore"`

```
Step 1 — Query Parser
  Input:  "Find VP Engineering of fintech startups in Bangalore"
  Output: QueryPlan = {
    is_lead_gen_query: true,
    needs_clarification: false,
    company_filters: {
      industry: "fintech",
      location: "Bangalore",
      company_size: "startup",
      keywords: ["fintech", "payments", "banking tech"]
    },
    signal_hints: ["hiring engineering", "Series A", "growing tech team"],
    target_role: "VP Engineering / Head of Engineering / VP Tech",
    agents_needed: ["company_search", "people_finder"]
  }

Step 2 — Guard Router
  is_lead_gen_query = true, needs_clarification = false
  → routes to company_search + people_finder

Step 3 — Company Search Agent
  SearXNG("fintech startups Bangalore")
  → ["Razorpay", "Slice", "Jupiter", "Setu", ...]
  Crawl4AI(crunchbase) → { revenue, funding, confidence: 0.74 }
  Wappalyzer → { tech_stack: ["AWS", "React", "PostgreSQL"] }
  → GraphState.companies updated

Step 4 — People Finder Agent
  role_normalizer("VP Engineering")
  → tries: ["VP Engineering", "VP of Engineering", "Head of Engineering",
             "Director of Engineering", "VP Tech"]
  Layer 1: linkedin-api.search_people(company="Razorpay", title="VP Engineering")
  → { name: "Rahul Mehta", linkedin_url, headline }
  title_score: 0.95 — exact match
  → GraphState.people updated

Step 5 — Lead Scoring Agent
  Gemini 2.5 Flash reads companies + people (no signals activated for this query)
  → GraphState.lead_score = [{
       company: "Razorpay",
       person: "Rahul Mehta, VP Engineering",
       score: 82,
       reasons: ["Uses AWS", "Series D funded", "Active engineering hiring"]
     }]

Step 6 — Contact Enricher
  Level 1: HunterProvider("razorpay.com", "Rahul Mehta") → null (not in DB)
  Level 2: PermutatorProvider → "r.mehta@razorpay.com" (confidence: 60)
  Level 3: TheHarvesterProvider → confirms "r.mehta@razorpay.com" (confidence: 78)
  → GraphState.contacts = [{
       name: "Rahul Mehta",
       title: "VP Engineering",
       company: "Razorpay",
       email: "r.mehta@razorpay.com",
       confidence: 78,
       linkedin: "linkedin.com/in/rahulmehta",
       status: "partial",
       tried: ["hunter", "permutator", "harvester"],
       suggestion: null
     }]

Step 7 — Result Formatter
  Chat: "Found 6 VPs of Engineering at fintech startups in Bangalore.
         Top lead: Rahul Mehta, VP Engineering at Razorpay — Score 82/100
         Email: r.mehta@razorpay.com (78% confidence)
         Reasons: Uses AWS, Series D funded, Active engineering hiring
         
         3 contacts with verified emails · 2 partial · 1 not found"
  JSON: [{ name, title, company, email, confidence, score, status, ... }]
```

---

## 8. LLM Provider Strategy

Three free providers, each assigned to the right task. No credit card needed for any of them.

### Provider Assignment

| Agent / Node | Provider | Model | Why |
|---|---|---|---|
| Query Parser | **Google Gemini** | `gemini-2.5-flash` | Best structured output, 1500 req/day free |
| Lead Scoring | **Google Gemini** | `gemini-2.5-flash` | Complex reasoning task, needs the strongest free model |
| Guard Router | **Groq** | `llama-3.1-8b-instant` | Simple classification, needs to be instant |
| Clarification Node | **Groq** | `llama-3.1-8b-instant` | Simple response generation, fast |
| Company Search Agent | **Cerebras** | `llama3.1-70b` | 1M tokens/day — agents make many LLM calls |
| Signal Filter Agent | **Cerebras** | `llama3.1-70b` | Same — high token volume per run |
| People Finder Agent | **Cerebras** | `llama3.1-70b` | Same — high token volume per run |
| browser-use (Layer 3) | **Groq** | `llama-3.1-8b-instant` | Fast page interaction, simple decisions |

### Why This Split

```
Gemini 2.5 Flash  → best reasoning quality   → query understanding + scoring
Groq Llama 8B     → fastest response time    → routing + browser control
Cerebras Llama 70B → highest daily volume    → agents (many calls per query)
```

### Rate Limits at a Glance

| Provider | Free Limit | No Credit Card |
|---|---|---|
| Google AI Studio | 1,500 req/day · 1M tokens/min | ✅ |
| Groq | 1,000 req/day · 100K tokens/day | ✅ |
| Cerebras | 1M tokens/day · 20 RPM | ✅ |

### LangChain Setup

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI  # Cerebras uses OpenAI-compatible endpoint

# Query Parser + Lead Scoring — best reasoning
llm_main = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Guard Router + Clarification + browser-use — fastest response
llm_fast = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# All three main agents — highest token volume
llm_agents = ChatOpenAI(
    model="llama3.1-70b",
    base_url="https://api.cerebras.ai/v1",
    api_key=os.getenv("CEREBRAS_API_KEY")
)
```

### Install

```bash
pip install langchain-google-genai langchain-groq langchain-openai
```

### Get Your Keys (all free, no credit card)

```
Google AI Studio : aistudio.google.com      → "Get API Key"
Groq             : console.groq.com         → "API Keys"
Cerebras         : cloud.cerebras.ai        → "API Keys"
```

Get all three before your first Claude Code session — takes under 10 minutes.

---

## 9. Tech Stack Summary

| Layer | Technology | Purpose |
|---|---|---|
| **Orchestration** | LangGraph `StateGraph` | Agent routing + state management |
| **Query Understanding** | Google Gemini 2.5 Flash (free) | Generic `QueryPlan` extraction |
| **Lead Scoring** | Google Gemini 2.5 Flash (free) | Pure reasoning over GraphState |
| **Routing / Browser** | Groq Llama 3.1 8B Instant (free) | Fast decisions, low latency |
| **Agent LLM** | Cerebras Llama 3.1 70B (free) | High token volume, 1M/day |
| **Web Search** | SearXNG (self-hosted Docker) | Dynamic queries via 70+ engines |
| **Web Scraping** | Crawl4AI | AI-native structured extraction |
| **LinkedIn Layer 1** | `linkedin-api` (Voyager HTTP) | Fast, no browser |
| **LinkedIn Layer 2** | `linkedin_scraper` + Camoufox | Stealth Firefox browser |
| **LinkedIn Layer 3** | `browser-use` + Groq | LLM-driven, selector-free |
| **LinkedIn Layer 4** | Crosslinked + public fallbacks | No LinkedIn login needed |
| **Stealth** | Camoufox + playwright-stealth + Patchright | Anti-detection |
| **Signal Sources** | Reddit PRAW · HN Algolia · GitHub API · LinkedIn Jobs | Buying signal detection |
| **Email Enrichment** | EmailProvider ABC (6 levels) | Swappable, never silent fail |
| **Data Models** | Pydantic v2 | Type-safe throughout |
| **API** | FastAPI | POST /chat endpoint |
| **Frontend** | React | Chat interface |

---

## 10. Build Order

- [ ] `docker-compose.yml` — SearXNG on port 8080, verify JSON response
- [ ] `graph/state.py` — `GraphState` + all TypedDicts including `QueryPlan`, `CompanyFilters`, `ContactData` with status/tried fields
- [ ] `tools/searxng_tool.py` — first `@tool`, dynamic query from keywords list
- [ ] `agents/query_parser.py` — Claude extracts `QueryPlan` with Pydantic output
- [ ] `graph/router.py` — `guard_router` conditional edge (reject / clarify / proceed)
- [ ] `agents/clarification.py` — returns clarification question to user
- [ ] `utils/role_normalizer.py` — role expansion + company name normalization
- [ ] `tools/linkedin_api_tool.py` — Layer 1, cookie auth + `search_people()`
- [ ] `utils/human_behavior.py` + `utils/rate_limiter.py` + `utils/session_manager.py`
- [ ] `tools/linkedin_scraper_tool.py` — Layer 2, Camoufox + stealth
- [ ] `tools/crawl4ai_tool.py` — confidence-scored extraction
- [ ] `agents/company_search.py` — reads from `QueryPlan.company_filters`
- [ ] `agents/people_finder.py` — 4-layer fallback + role normalizer wired in
- [ ] `graph/orchestrator.py` — full StateGraph wired: parser → guard → agents
- [ ] `tools/reddit_tool.py` + `tools/hn_tool.py` + `tools/github_tool.py` + `tools/wappalyzer_tool.py`
- [ ] `agents/signal_filter.py` — reads from `QueryPlan.signal_hints`
- [ ] `agents/lead_scoring.py` — pure Claude reasoning, no tools, hallucination-safe
- [ ] `providers/email_provider.py` — ABC definition
- [ ] `providers/hunter_provider.py` — Level 1
- [ ] `providers/permutator_provider.py` — Level 2
- [ ] `providers/harvester_provider.py` — Level 3
- [ ] `providers/google_dork_provider.py` — Level 4
- [ ] `providers/linkedin_contact_provider.py` — Level 5
- [ ] `providers/website_contact_provider.py` — Level 6
- [ ] `agents/contact_enricher.py` — chains all 6 providers, always returns result
- [ ] `tools/browser_use_tool.py` — Layer 3, Claude Haiku
- [ ] `agents/result_formatter.py` — chat + JSON + stats
- [ ] `api/main.py` — FastAPI POST `/chat`
- [ ] Basic React chat frontend
- [ ] `tests/test_tools.py` + `tests/test_guard_router.py` + `tests/test_graph.py`
- [ ] End-to-end: run 3 different query types, verify output shape

---

## 11. CLAUDE.md (Drop This in Project Root Before First Session)

> The full standalone `CLAUDE.md` file is provided separately.
> Copy it exactly as-is into the project root.
> Below is a summary of what it contains.

**Sections in CLAUDE.md:**
- Project Purpose — what we're building and why
- Architecture Flow — full pipeline in one diagram
- Tech Stack table — every technology and its role
- LLM Provider Assignment — which model goes to which node
- Folder Map — every file with its purpose
- GraphState Shape — quick reference for all TypedDicts
- Non-Negotiable Rules — 11 rules Claude Code must follow every session
- Session Pattern — how every prompt interaction works
- Environment Variables — all keys with where to get them
- Build Order Checklist — grouped by phase, update after each component
- Opening Prompt — exact text to paste at the start of every new session

---

## 12. Key Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| LinkedIn bans throwaway account | People Finder Layer 1+2 break | Camoufox + rate limiter. Layer 3+4 continue working without LinkedIn login |
| LinkedIn DOM changes | Layer 2 selectors break | Layer 3 (browser-use) has no selectors — adapts automatically |
| Wrong person found | Bad leads | Role normalizer + title_score — near-matches scored, not discarded |
| Company name mismatch | LinkedIn search returns nothing | Company name normalizer runs before every LinkedIn search |
| Revenue/funding data unreliable | Scoring skewed | confidence float on every CompanyData — scorer weights accordingly |
| Email not found | Partial contact | 6-level EmailProvider chain + partial result with LinkedIn suggestion |
| Query too vague | Garbage results | Guard router → clarification node — never runs agents on vague input |
| Non-lead-gen query | Agents fire uselessly | Guard router rejects instantly — no agents activated |
| browser-use token cost | High API spend | Claude Haiku only, Layer 3 only — never for bulk |
| Lead Scoring hallucinates | Wrong scores | Scorer cannot call tools — only reasons over existing GraphState data |

---

## 13. What This POC Validates

- [ ] Guard router correctly rejects non-lead-gen queries
- [ ] Guard router correctly asks clarification for vague queries
- [ ] QueryPlan correctly extracted for 5 different query types
- [ ] Company Search builds queries dynamically from QueryPlan (no hardcoding)
- [ ] Role normalizer finds person when exact title doesn't exist on LinkedIn
- [ ] All 4 People Finder layers activate in sequence when previous layer fails
- [ ] Signal Filter reads signal_hints from QueryPlan (not hardcoded signals)
- [ ] Lead Scoring outputs ranked results with reasons using only GraphState data
- [ ] EmailProvider chain tries all 6 levels before returning partial
- [ ] Partial contacts returned with status + suggestion — never empty result
- [ ] End-to-end: query → contacts in under 3 minutes

---

*Single source of truth for this POC. Update CLAUDE.md current status as each build order item is completed.*
