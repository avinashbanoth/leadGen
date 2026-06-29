# Lead Generation Multi-Agent POC — Project Context

> Read this file completely before doing anything in every session.
> This is the single source of truth for how we work together.

---

## Project Purpose

Build a generic multi-agent AI system that takes any natural language query
and returns verified contact details (email, LinkedIn, phone) of the right
decision makers at the right companies.

NOT hardcoded to any industry, role, or signal.
The user's query drives everything dynamically via QueryPlan.

---

## Architecture Flow

```
User Query
    │
    ▼
Query Parser        → extracts QueryPlan (industry, role, signals, agents needed)
    │
    ▼
Guard Router        → rejects non-lead-gen · asks clarification if vague · proceeds if valid
    │
    ▼
[Company Search | Signal Filter | People Finder]   → run based on QueryPlan
    │
    ▼
Lead Scoring        → Groq reasons over GraphState, ranks leads 0–100 with reasons
    │
    ▼
Contact Enricher    → 5-level EmailProvider chain, never silent fail
    │
    ▼
Result Formatter    → chat response + JSON + stats
```

---

## Tech Stack

| What | Technology |
|---|---|
| Orchestration | LangGraph StateGraph |
| Framework | LangChain |
| API | FastAPI |
| Data Models | Pydantic v2 |
| Language | Python 3.11+ |
| LLM (all nodes + agents) | Groq — `langchain-groq` (free, no credit card) |
| Web Search | SearXNG — self-hosted Docker at `localhost:8080` |
| Web Scraping | Crawl4AI |
| Indian Directory Sources | JustDial · IndiaMart · Kompass · Zaubacorp (MCA data) |
| People Layer A | Apollo.io People Search API (global, structured — free search) |
| People Layer B | Kompass director emails (extracted directly from listing page) |
| People Layer C | Zaubacorp board of directors (MCA government data, DIN + email) |
| People Layer D | Website team page scraper via Crawl4AI (/team /about /leadership) |
| People Layer E | Google dorks via SearXNG (last resort, large companies) |
| Signal Sources | HN Algolia API (no auth) · GitHub API · LinkedIn Jobs public pages · SearXNG dorks |
| Email Enrichment | EmailProvider ABC → 5 levels (Hunter → Permutator → Harvester → GoogleDork → Website) |
| Frontend | React (minimal chat UI) |

---

## LLM Provider Assignment

| Node / Agent | Provider | Model |
|---|---|---|
| Query Parser | Groq | `llama-3.3-70b-versatile` |
| Lead Scoring | Groq | `llama-3.3-70b-versatile` |
| Guard Router | Groq | `llama-3.1-8b-instant` |
| Clarification Node | Groq | `llama-3.1-8b-instant` |
| Company Search Agent | Groq | `llama-3.3-70b-versatile` |
| Signal Filter Agent | Groq | `llama-3.3-70b-versatile` |
| People Finder Agent | Groq | `llama-3.3-70b-versatile` |

---

## Folder Map

```
lead-gen-agent/
├── CLAUDE.md                    ← you are here
├── lead-gen-poc-architecture.md ← full architecture reference
├── .env                         ← all API keys
├── docker-compose.yml           ← SearXNG
├── requirements.txt
│
├── graph/
│   ├── state.py                 ← GraphState + all TypedDicts
│   ├── orchestrator.py          ← StateGraph nodes + edges
│   └── router.py                ← guard_router + agent_router
│
├── agents/
│   ├── query_parser.py
│   ├── clarification.py
│   ├── company_search.py
│   ├── signal_filter.py
│   ├── people_finder.py
│   ├── lead_scoring.py
│   ├── contact_enricher.py
│   └── result_formatter.py
│
├── tools/
│   ├── searxng_tool.py
│   ├── crawl4ai_tool.py
│   ├── apollo_tool.py           ← Layer A: People Search (names + titles + LinkedIn URLs)
│   ├── website_team_tool.py     ← Layer D: /team /about scraper
│   ├── crosslinked_tool.py      ← Layer E: Google dorks
│   ├── indiamart_tool.py        ← Company discovery: Indian B2B marketplace (static HTML)
│   ├── justdial_tool.py         ← Company discovery: local business search, 250+ cities (Playwright)
│   ├── kompass_tool.py          ← Company discovery + Layer B: executive emails on listing page
│   ├── zaubacorp_tool.py        ← Company discovery + Layer C: MCA government directors + DIN
│   ├── hn_tool.py               ← HN Algolia API (no auth needed)
│   ├── github_tool.py           ← GitHub Issues + org members
│   └── wappalyzer_tool.py       ← tech stack detection
│
├── providers/
│   ├── email_provider.py        ← ABC
│   ├── hunter_provider.py       ← Level 1 (50 free/month)
│   ├── permutator_provider.py   ← Level 2 (SMTP MX fix via dnspython)
│   ├── harvester_provider.py    ← Level 3 (theHarvester OSINT)
│   ├── google_dork_provider.py  ← Level 4 (SearXNG dork)
│   └── website_contact_provider.py ← Level 5 (Crawl4AI /team /contact)
│
├── utils/
│   ├── human_behavior.py        ← kept for future use
│   ├── session_manager.py       ← kept for future use
│   ├── rate_limiter.py          ← kept for future use
│   ├── role_normalizer.py
│   └── api_tracker.py           ← singleton: Groq key hint, Hunter credits, Apollo status
│
├── api/
│   └── main.py
│
└── tests/
    ├── test_tools.py
    ├── test_guard_router.py
    └── test_graph.py
```

---

## GraphState Shape (reference)

```python
class GraphState(TypedDict):
    query      : str
    query_plan : QueryPlan         # extracted by query_parser — drives all agents
    companies  : list[CompanyData] # always includes confidence: float
    people     : list[PersonData]  # always includes title_score: float
    signals    : list[SignalData]
    lead_score : list[LeadScore]   # score 0–100 + reasons list
    contacts   : list[ContactData] # always includes status + tried + suggestion
    errors     : list[str]         # every agent writes here on failure
    messages   : list[str]
    status     : str
```

Full TypedDict definitions are in `graph/state.py` and
`lead-gen-poc-architecture.md` Section 3.

---

## Non-Negotiable Rules — Read Before Every Component

1. **ONE component per prompt.** Never build more than one file at a time.
2. **Explain before coding — wait for approval.** Before writing any code:
   - What does this component do in plain English?
   - Where does it fit in the architecture flow?
   - What does it read from GraphState? What does it write?
   - Why are we building it this way?
   End with: "Ready to write it?"
   Only write code after explicit approval ("yes" / "ok go ahead").
   Never write code without receiving approval first.
3. **Every tool = `@tool` decorated async function.** No exceptions.
4. **GraphState is the only data channel.** No agent passes data directly to another agent ever.
5. **QueryPlan drives everything.** Never hardcode any industry, company, role, or signal.
6. **Confidence on all company data.** Every `CompanyData` must have `confidence: float` (0.0–1.0).
7. **Lead Scoring has NO tools.** Reads GraphState only — cannot look up new information.
8. **EmailProvider ABC always.** Never call Hunter.io directly from agent code.
9. **Never silently fail.** Every agent writes to `GraphState.errors` on failure.
10. **Named company short-circuit.** If `QueryPlan.company_named_directly = True`, skip company_search and go straight to people_finder.
11. **After each component:** suggest a git commit message before moving on.
12. **Block-by-block explanation after code.** Explain what each section does and why.

---

## How We Work Together (Session Pattern)

```
You explain the component → I confirm → You write code
→ You explain code block by block → I ask questions
→ We verify it works → You suggest git commit → Next component
```

Never jump ahead. Never build the next thing unless I explicitly ask.

---

## Environment Variables (.env)

```
# LLM Provider (free, no credit card)
GROQ_API_KEY=...             # console.groq.com

# Contact Data
APOLLO_API_KEY=...           # app.apollo.io → Settings → Integrations → API Keys
HUNTER_API_KEY=...           # hunter.io → Dashboard → API (50 free/month)

# Signal Sources
GITHUB_TOKEN=...             # github.com → Settings → Developer Settings → PAT (classic)
                             # Scopes: public_repo + read:org

# HackerNews — NO KEY NEEDED
# SearXNG    — NO KEY NEEDED (self-hosted)
# Reddit     — SKIPPED (IP blocked, HN + GitHub covers the same signals)
```

---

## Build Order Checklist

Update this list after every completed component.

### Foundation
- [x] `docker-compose.yml` — SearXNG on port 8080
- [x] `searxng/settings.yml` — JSON format enabled, rate limiter off, safe_search=2
- [x] `graph/state.py` — GraphState + all TypedDicts (needs `company_named_directly` added to QueryPlan)
- [x] `tools/searxng_tool.py` — categories=map for location queries + Groq 8b city expansion (state/country → cities); categories=general otherwise
- [x] `tools/overpass_tool.py` — OpenStreetMap Overpass API, free/no-auth; Nominatim geocode → bbox; keyword-filtered OSM office/shop/craft query; fallback to general office search; wired into company_search Phase 0 alongside IndiaMart/JustDial (concurrent)

### Utilities
- [x] `utils/role_normalizer.py` — role expansion + two-tier fallback map
- [x] `utils/human_behavior.py` — kept for future use
- [x] `utils/rate_limiter.py` — kept for future use
- [x] `utils/session_manager.py` — kept for future use
- [x] `utils/api_tracker.py` — singleton tracker: Groq key hint + query count, Hunter live credits, Apollo health; reads .env on every call so key changes reflect instantly

### Query Understanding
- [x] `agents/query_parser.py` — Groq extracts QueryPlan (needs `company_named_directly` field)
- [x] `graph/router.py` — guard_router conditional edge
- [x] `agents/clarification.py` — vague query handler

### People Stack
- [x] `tools/apollo_tool.py` — Layer A: Apollo People + Company Search
- [x] `tools/website_team_tool.py` — Layer D: /team /about scraper via Crawl4AI + Groq 8b extraction
- [x] `tools/crosslinked_tool.py` — Layer E: Google dorks (regex broadened for accents/hyphens)

### Indian Directory Layer (Company Discovery + People Layers B & C)
- [x] `tools/indiamart_tool.py` — static HTML scraper (dir.indiamart.com/{city}/{category}.html)
- [x] `tools/justdial_tool.py` — JS-rendered via Crawl4AI, 3-URL fallback, 3-strategy parser
- [x] `tools/kompass_tool.py` — executive contacts from Kompass India profiles (Layer B)
- [x] `tools/zaubacorp_tool.py` — MCA director names + DIN from Zaubacorp (Layer C)

### Core Agents
- [x] `tools/crawl4ai_tool.py` — confidence-scored company extraction
- [x] `agents/company_search.py` — Phase 0 IndiaMart/JustDial + Overpass (concurrent) → Phase 1A dirs → Phase 1B SearXNG (map category + city expansion) → Phase 2 verify
- [x] `agents/people_finder.py` — Apollo (A) → Kompass (B) → Zaubacorp (C) → website team (D) → dorks (E)
- [x] `graph/orchestrator.py` — full StateGraph wired

### Signal Detection
- [x] `tools/hn_tool.py` — HackerNews Algolia API
- [x] `tools/github_tool.py` — GitHub Issues + org members
- [x] `tools/wappalyzer_tool.py` — tech stack detection
- [x] `agents/signal_filter.py`

### Intelligence + Enrichment
- [x] `agents/lead_scoring.py` — Groq reasoning, no tools
- [x] `providers/email_provider.py` — ABC
- [x] `providers/hunter_provider.py` — Level 1
- [x] `providers/permutator_provider.py` — Level 2 (SMTP via dnspython MX lookup — fixed)
- [x] `providers/harvester_provider.py` — Level 3
- [x] `providers/google_dork_provider.py` — Level 4
- [x] `providers/website_contact_provider.py` — Level 5 (single shared crawler — fixed)
- [x] `agents/contact_enricher.py` — fresh providers per call, Apollo email short-circuit

### Output + API
- [x] `agents/result_formatter.py` — domain deduplication (1 contact per unique domain when company_search in agents_needed); deduplicated flag in stats payload
- [x] `api/main.py` — FastAPI POST /chat + GET /api/credits (live API key tracker)

### Frontend
- [x] React chat UI — CDN + Babel, served at `/`, dark GitHub theme
- [x] `CompanyTable` — one row per company, best contact per company, nil = `—`
- [x] `StatsPanel` sidebar — token usage bar, queries today, last query, model tags
- [x] `CreditsSection` — Groq key badge + query count + rate-limit alert, Hunter progress bar, Apollo status dot; Tailwind CDN layout, auto-refresh every 30 s

### Testing
- [ ] `tests/test_tools.py`
- [ ] `tests/test_guard_router.py`
- [ ] `tests/test_graph.py`
- [ ] End-to-end: 3 different query types verified with new people stack (Apollo + team pages)

---

## Opening Prompt for Every New Session

Paste this at the start of every Claude Code conversation:

```
Read both files before doing anything:
1. CLAUDE.md — project rules, current status, and build order
2. lead-gen-poc-architecture.md — full architecture reference

After reading both, tell me:
- What has been built so far (from the checklist above)
- What the next item in the build order is
- Explain that component to me — what it does, where it fits,
  what it reads and writes in GraphState — before any code
- Wait for my confirmation before proceeding
```

---

*Last updated: 4 quality fixes applied — (1) title hallucination: website_team_tool prompt rules + 3-step _clean_title() validator in people_finder; (2) entity variety: domain dedup in result_formatter (1 contact/domain for company-search runs); (3) SearXNG location fix: categories=map + Groq 8b city expansion for state/country queries; (4) OpenStreetMap Overpass: new tools/overpass_tool.py, wired into company_search Phase 0 alongside IndiaMart/JustDial (concurrent). All pipeline code complete. Remaining: tests + end-to-end verification.*
