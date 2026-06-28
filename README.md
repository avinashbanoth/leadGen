# Lead Gen Multi-Agent AI

A generic multi-agent system that takes any natural language B2B lead generation query and returns verified contact details of the right decision makers at the right companies. No hardcoded industries, roles, or signals — the user's query drives everything.

---

## Architecture

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
[Company Search | Signal Filter | People Finder]   → run in parallel based on QueryPlan
    │
    ▼
Lead Scoring        → ranks (company, person) pairs 0–100 with reasons
    │
    ▼
Contact Enricher    → 6-level email chain: Hunter → Permutator → Harvester → Dork → LinkedIn → Website
    │
    ▼
Result Formatter    → chat message + structured JSON + stats
```

### LinkedIn 4-Layer Fallback
| Layer | Tool | Method |
|---|---|---|
| 1 | `linkedin-api` | Voyager HTTP (cookie auth) |
| 2 | `linkedin_scraper` + Camoufox | Stealth browser |
| 3 | `browser-use` + Groq | LLM-driven browser, no selectors |
| 4 | Crosslinked | Google dorks, no login required |

Each layer falls back to the next on failure (ChallengeException, empty results, etc.).

---

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph `StateGraph` |
| Framework | LangChain |
| API | FastAPI |
| Data Models | Pydantic v2 |
| Language | Python 3.11+ |
| Web Search | SearXNG (self-hosted Docker) |
| Web Scraping | Crawl4AI |
| Signals | Reddit PRAW · HN Algolia · GitHub API · LinkedIn Jobs |
| Email Enrichment | Hunter.io → permutations → harvester → Google dork → LinkedIn → website |
| Frontend | React 18 (CDN, no build step) |

### LLM Assignment
| Task | Model | Reason |
|---|---|---|
| Query parsing | Groq `llama-3.3-70b-versatile` | Structured JSON extraction |
| Lead scoring | Groq `llama-3.3-70b-versatile` | Complex reasoning |
| Search query generation | Groq `llama-3.3-70b-versatile` | Creative, accurate |
| Guard routing | Groq `llama-3.1-8b-instant` | Fast binary decision |
| Industry inference | Groq `llama-3.1-8b-instant` | High-volume, simple |
| Title scoring | Groq `llama-3.1-8b-instant` | High-volume, simple |
| Browser agent (Layer 3) | Groq `llama-3.1-8b-instant` | Real-time browser control |

---

## Setup

### Prerequisites
- Python 3.11+
- Docker (for SearXNG)
- Git

### 1. Clone and install
```bash
git clone https://github.com/avinashbanoth/leadGen.git
cd leadGen
pip install -r requirements.txt
```

### 2. Post-install binary downloads
```bash
playwright install chromium          # for Crawl4AI
playwright install firefox           # for Camoufox (Layer 2)
python -m camoufox fetch             # Camoufox browser binary
```

### 3. Environment variables
Create `.env` in the project root:
```env
# LLM
GROQ_API_KEY=...          # console.groq.com (free tier: 100K tokens/day for 70b)

# LinkedIn (throwaway account only — never personal)
LI_USERNAME=...
LI_PASSWORD=...

# Email enrichment
HUNTER_API_KEY=...        # hunter.io (25 lookups/month free)

# Optional signals
GITHUB_TOKEN=...          # github.com/settings/tokens (increases rate limit)
```

### 4. Start SearXNG
```bash
docker-compose up -d
```
Verify: `curl http://localhost:8080/search?q=test&format=json`

### 5. Start the server
```bash
python -m uvicorn api.main:app --port 8000
```

### 6. Open the dashboard
Navigate to **http://localhost:8000/** in your browser.

---

## Usage

### Dashboard (recommended)
Open `http://localhost:8000/` — type any B2B lead gen query and hit Enter.

Example queries:
- `Find CTOs at Series B SaaS companies in the US that use React`
- `VP Engineering contacts at UK fintech startups with 50-200 employees`
- `Founders of AI startups that raised Series A in 2024`
- `CFOs at healthcare software companies in Germany with revenue over 10M`

### API directly
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Find CTOs at fintech startups in Bangalore"}'
```

Response:
```json
{
  "message": "...",
  "status": "complete | no_results | awaiting_clarification | rejected",
  "contacts": [...],
  "stats": {"total": 3, "verified": 1, "partial": 2},
  "errors": [...],
  "elapsed_seconds": 61.2,
  "estimated_tokens": 12000
}
```

### Token usage
```bash
curl http://localhost:8000/stats
```
```json
{
  "queries_today": 3,
  "estimated_tokens_today": 36000,
  "token_limit": 100000,
  "tokens_remaining": 64000,
  "reset_in_seconds": 38400
}
```

### Batch test
```bash
python test_queries.py
```
Runs 10 diverse queries sequentially and prints status, timing, and errors for each.

---

## Project Structure

```
lead-gen/
├── api/
│   └── main.py                  FastAPI app — /chat, /stats, /health, serves frontend
├── agents/
│   ├── query_parser.py          Groq 70b extracts QueryPlan (json_mode)
│   ├── clarification.py         Asks user for missing details
│   ├── company_search.py        SearXNG + Crawl4AI + Groq industry inference
│   ├── people_finder.py         4-layer LinkedIn cascade
│   ├── signal_filter.py         Reddit + HN + GitHub + Wappalyzer
│   ├── lead_scoring.py          Groq 70b ranks leads 0–100 (no tools)
│   ├── contact_enricher.py      6-provider email chain
│   └── result_formatter.py      Chat message + JSON output
├── graph/
│   ├── state.py                 GraphState + all TypedDicts
│   ├── orchestrator.py          LangGraph StateGraph wiring
│   └── router.py                Guard router + agent router edges
├── tools/
│   ├── searxng_tool.py
│   ├── crawl4ai_tool.py
│   ├── linkedin_api_tool.py     Layer 1
│   ├── linkedin_scraper_tool.py Layer 2
│   ├── browser_use_tool.py      Layer 3
│   ├── crosslinked_tool.py      Layer 4
│   ├── reddit_tool.py
│   ├── hn_tool.py
│   ├── github_tool.py
│   └── wappalyzer_tool.py
├── providers/
│   ├── email_provider.py        ABC
│   ├── hunter_provider.py       Level 1 — Hunter.io API
│   ├── permutator_provider.py   Level 2 — SMTP-verified permutations
│   ├── harvester_provider.py    Level 3 — SearXNG email harvest
│   ├── google_dork_provider.py  Level 4 — Google dork search
│   ├── linkedin_contact_provider.py  Level 5
│   └── website_contact_provider.py   Level 6 — Crawl4AI /contact page
├── utils/
│   ├── human_behavior.py        Anti-detection delays
│   ├── session_manager.py       LinkedIn cookie persistence
│   ├── rate_limiter.py          Rolling-window rate limiter
│   └── role_normalizer.py       Role expansion + company normalization
├── frontend/
│   └── index.html               React 18 dashboard (CDN, no npm)
├── tests/
│   └── (pending)
├── docker-compose.yml           SearXNG on port 8080
├── requirements.txt
└── test_queries.py              10-query evaluation script
```

---

## Known Limitations

| Issue | Cause | Status |
|---|---|---|
| LinkedIn returns 0 people | `ChallengeException` on Voyager API without warmed session | Layer 2–4 cascade fires but needs active LinkedIn session |
| Company search returns list pages | SearXNG finds blog aggregators instead of company homepages | Partially mitigated via `_infer_industry` prompt rejecting list pages |
| Groq 100K tokens/day (free tier) | `llama-3.3-70b-versatile` hard daily cap | ~8 full-pipeline queries/day; 8b model has higher limit; resets midnight UTC |
| SearXNG must be running | All web search goes through self-hosted Docker instance | `docker-compose up -d` before starting |

---

## Query Types

| Query type | Routing | Typical time |
|---|---|---|
| Non-lead-gen ("weather", "poem") | Rejected immediately | < 1s |
| Vague ("find me some leads") | Clarification asked | ~2s |
| Valid lead-gen | Full pipeline | 50–90s |
