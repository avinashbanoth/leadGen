# Lead Generation Multi-Agent System — POC Architecture

> **Status:** Proof of Concept v4 — Final before build  
> **Author:** Avinash  
> **Stack:** Python 3.11+ · LangChain · LangGraph · FastAPI  
> **Goal:** Generic, production-grade contact finder via natural language chat  
> **Changes:** Generic QueryPlan · Guard Router · Role Normalizer · Partial Results · Extended Email Fallbacks · Multi-provider LLM (Gemini + Groq + Cerebras)

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
    is_lead_gen_query   : bool          # False → guard router rejects
    needs_clarification : bool          # True → ask user for more info
    clarification_ask   : str | None    # question to show user
    rejection_reason    : str | None    # why it was rejected
    company_filters     : CompanyFilters | None
    signal_hints        : list[str]     # ["hiring devops", "cloud cost issues"]
    target_role         : str | None    # "CTO", "HR Head", "VP Engineering"
    agents_needed       : list[str]     # ["company_search", "people_finder"]

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
    company      : str
    linkedin_url : str
    email        : str | None
    phone        : str | None
    source       : str      # which layer found this (linkedin_api / scraper / dork)

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
    company    : str
    email      : str | None     # None if not found
    confidence : int            # 0–100
    linkedin   : str
    phone      : str | None
    score      : int
    status     : str            # "verified" / "partial" / "not_found"
    tried      : list[str]      # which providers were attempted
    suggestion : str | None     # "Try LinkedIn directly: <url>" if email not found

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
| `SearXNG` | Self-hosted metasearch — builds search query dynamically from `company_filters.keywords + industry + location` |
| `Crawl4AI` | Scrapes Crunchbase public pages — revenue, funding, headcount with confidence score |
| `Google Maps Scraper` | Local business phone, address, website |
| `Wappalyzer` | Detects tech stack from company website |

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
| `LinkedIn Jobs (public, no login)` | Job postings that match signal hints |
| `Reddit PRAW` | Community posts matching signal hints |
| `HackerNews Algolia API` | "Ask HN" threads matching signal hints |
| `GitHub Issues API` | Open issues matching signal keywords |
| `Wappalyzer` | Tech stack presence as a signal |

---

### 4.6 People Finder Agent

**What it does:** Finds the right person at each company. Uses `QueryPlan.target_role` dynamically — never hardcodes "CTO" or any role. Includes a role normalizer and a four-layer fallback.

**LangGraph role:** Node — reads `GraphState.companies + query_plan.target_role`, writes `GraphState.people`

**Role Normalizer (runs before search):**

LinkedIn rarely has exactly the title in the query. The normalizer expands the role:

```python
# target_role from query: "CTO"
# normalizer expands to:
search_titles = [
    "CTO",
    "Chief Technology Officer",
    "VP Engineering",
    "Head of Technology",
    "Co-founder & CTO",
    "VP of Engineering"
]
# tries each until results found
```

Also normalizes company name:
```python
# query says "Siemens PLM" → SearXNG finds official LinkedIn name
# → "Siemens Digital Industries Software"
# searches with correct name
```

**Four-Layer Fallback:**

```
Layer 1: linkedin-api (Voyager HTTP)
         search_people(company, title)
         get_profile_contact_info(profile_id)
         → fastest, no browser, direct HTTP with session cookie
                │
                │ if rate-limited / blocked
                ▼
Layer 2: linkedin_scraper + Camoufox
         Browser automation, Firefox engine
         Stealth patches + human behavior simulation
         → slower, more resilient, survives DOM changes less well
                │
                │ if LinkedIn DOM changes / selectors break
                ▼
Layer 3: browser-use Agent (Claude Haiku)
         LLM reads page, decides what to click
         No CSS selectors — adapts to any LinkedIn UI
         → most resilient, most expensive (tokens), last resort
                │
                │ if LinkedIn fully inaccessible
                ▼
Layer 4: Public fallbacks (no LinkedIn needed)
         • Crosslinked  — Google dorks: site:linkedin.com/in "CTO" "company"
         • Crunchbase   — /people section via Crawl4AI
         • GitHub API   — org members (for tech companies)
         • Company /team or /about page via Crawl4AI
         • Google dork  — "company name" "CTO" email site:company.com
```

Every layer writes its source into `PersonData.source` so you know which layer succeeded.

**Title relevance scoring:**
```python
# target_role = "CTO", found title = "Co-founder & Tech Lead"
# Claude scores: relevance = 0.85
# reason: "Co-founder with tech focus = likely decision maker"
# included in results with title_score, not discarded
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

**LLM:** Claude Sonnet 4.6 — reasoning task, not browsing.

**Hallucination guard:** Scorer only reasons over data already present in GraphState. It cannot call tools or look up new information. If a field is missing, it scores lower — never invents data.

---

### 4.8 Contact Enricher Agent

**What it does:** Takes scored leads and finds verified contact details. Uses `EmailProvider` abstraction — tries six levels before giving up. Always returns a result, even if partial.

**LangGraph role:** Node — reads `GraphState.lead_score + people`, writes `GraphState.contacts`

**EmailProvider abstraction:**

```python
from abc import ABC, abstractmethod

class EmailProvider(ABC):
    @abstractmethod
    async def find_email(self, name: str, domain: str) -> dict:
        """Returns { email, confidence, source }"""
        ...

    @abstractmethod
    async def verify_email(self, email: str) -> dict:
        """Returns { valid: bool, confidence, reason }"""
        ...

# Implementations (tried in order):
class HunterProvider(EmailProvider):      # Level 1 — 25 free/month
class PermutatorProvider(EmailProvider):  # Level 2 — f.lastname@domain.com patterns
class TheHarvesterProvider(EmailProvider) # Level 3 — OSINT sources
class GoogleDorkProvider(EmailProvider):  # Level 4 — "name" email site:domain.com
class LinkedInContactProvider(EmailProvider) # Level 5 — linkedin-api contact info
class WebsiteContactProvider(EmailProvider)  # Level 6 — /team /about via Crawl4AI
```

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

## 5. Camouflage & Anti-Detection Stack

```
Detection Layer     Our Solution
─────────────────── ────────────────────────────────────────────
Browser Fingerprint Camoufox (Firefox engine, not Chrome)
                    + playwright-stealth (patches navigator.webdriver,
                    HeadlessChrome UA, WebGL, plugin array, canvas)

TLS / Network       Patchright (closes CDP Runtime.Enable leak)
                    + realistic HTTP headers for linkedin-api

Behavioral          HumanBehavior utility:
                    • random_delay(800ms–3000ms) between actions
                    • human_type() — variable keystroke speed
                    • human_scroll() — gradual, not instant
                    • random_mouse_move() before clicks

Session             SessionManager:
                    • Warm session before any search
                      (feed → own profile → notifications → then search)
                    • Persist cookies to session.json
                    • Reuse warmed session — never fresh login per run

Rate Limiting       RateLimiter:
                    • Max 30 profiles/hour
                    • Max 15 searches/day
                    • 8–20s random delay between profiles
                    • 30–90s random delay between searches

IP Reputation       POC: local machine IP (fine for testing)
                    Production: residential IP (not datacenter)
```

```bash
# Install
pip install camoufox[geoip] playwright-stealth patchright browser-use
python -m camoufox fetch
python -m patchright install chromium
```

**Playwright codegen** (built-in, no extra install) — use during development to record LinkedIn selectors:
```bash
python -m playwright codegen https://www.linkedin.com
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
│   ├── company_search.py              # SearXNG + Crawl4AI + Wappalyzer
│   ├── signal_filter.py               # Reddit + HN + GitHub + LinkedIn Jobs
│   ├── people_finder.py               # 4-layer fallback + role normalizer
│   ├── lead_scoring.py                # pure Gemini reasoning — no tools
│   ├── contact_enricher.py            # EmailProvider abstraction chain
│   └── result_formatter.py            # final output formatting
│
├── tools/
│   ├── searxng_tool.py                # @tool — dynamic query from QueryPlan
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
