# Lead Gen Multi-Agent AI

A generic multi-agent system that takes any natural language B2B lead generation query and returns verified contact details of the right decision makers at the right companies. No hardcoded industries, roles, or signals — the user's query drives everything.

---

## Architecture

```
User Query
    │
    ▼
Query Parser        → extracts QueryPlan (industry, role, company, signals)
    │
    ▼
Guard Router        → rejects non-lead-gen · asks clarification if vague · proceeds if valid
    │
    ▼
Company Search      → two-phase discovery (directories → SearXNG → direct domains)
    │               → short-circuits to Apollo if a specific company is named
    ▼
People Finder       → Apollo (Layer A) → website team pages (Layer B) → Google dorks (Layer C)
    │               → two-tier role priority: C-suite first, Director/Manager fallback
    ▼
Lead Scoring        → Groq 70b ranks leads 0–100 with reasons (no tools, reads GraphState only)
    │
    ▼
Contact Enricher    → Hunter → Permutator (MX verified) → Harvester → Dork → Website
    │
    ▼
Result Formatter    → chat message grouped by company + structured JSON + stats
```

### People Finding — Three Layers

| Layer | Tool | Coverage |
|---|---|---|
| A | Apollo.io People Search | 270M+ professionals — large + mid-market companies globally |
| B | Website team page scraper | Any company with a /team or /about page — SMEs, European companies |
| C | Google dorks via SearXNG | Large companies with indexed LinkedIn profiles — last resort |

---

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph `StateGraph` |
| Framework | LangChain |
| API | FastAPI |
| Data Models | Pydantic v2 |
| Language | Python 3.11+ |
| LLM (all nodes) | Groq — `llama-3.3-70b-versatile` + `llama-3.1-8b-instant` |
| Web Search | SearXNG (self-hosted Docker on port 8080) |
| Web Scraping | Crawl4AI |
| People — Layer A | Apollo.io REST API |
| People — Layer B | Crawl4AI + Groq 8b extraction |
| People — Layer C | Google dorks via SearXNG |
| Signals | HackerNews Algolia API · GitHub API · Wappalyzer |
| Email Enrichment | Hunter → Permutator (dnspython MX) → Harvester → Dork → Website |
| Frontend | React 18 (CDN, no build step) |

### LLM Assignment

| Task | Model |
|---|---|
| Query parsing, lead scoring, company search | Groq `llama-3.3-70b-versatile` |
| Guard routing, title scoring, team page extraction | Groq `llama-3.1-8b-instant` |

---

## Setup

### Prerequisites
- Python 3.11+
- Docker (for SearXNG)

### 1. Clone and install
```bash
git clone https://github.com/avinashbanoth/leadGen.git
cd leadGen
pip install -r requirements.txt
crawl4ai-setup
```

### 2. Environment variables
Create `.env` in the project root:
```env
# LLM (free — console.groq.com)
GROQ_API_KEY=...

# People search (free — app.apollo.io → Settings → Integrations → API Keys)
APOLLO_API_KEY=...

# Email enrichment (free — hunter.io, 50 lookups/month)
HUNTER_API_KEY=...

# Signal detection (free PAT — github.com/settings/tokens, scopes: public_repo + read:org)
GITHUB_TOKEN=...
```

### 3. Start SearXNG
```bash
docker-compose up -d
```
Verify: open http://localhost:8080

### 4. Start the server
```bash
python -m uvicorn api.main:app --port 8000 --reload
```

### 5. Open the dashboard
Navigate to **http://localhost:8000/** in your browser.

---

## Usage

### Dashboard
Open `http://localhost:8000/` and type any B2B lead gen query.

Example queries:
- `Find CTOs at fintech startups in Bangalore`
- `VP Engineering at PLM software companies in Germany`
- `Who is the CTO at Razorpay`
- `Find founders of SaaS companies in Singapore that recently raised Series B`
- `CFOs at healthcare software companies in Germany`

See `QUERY_GUIDE.md` for the full query reference.

### API
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Find CTOs at fintech startups in Bangalore"}'
```

Response:
```json
{
  "message": "Found 5 lead(s) across 2 companies...",
  "status": "complete",
  "contacts": [
    {
      "name": "Harshil Mathur",
      "title": "CEO",
      "company": "Razorpay",
      "email": "harshil@razorpay.com",
      "confidence": 80,
      "linkedin": "linkedin.com/in/harshilmathur",
      "score": 85,
      "status": "partial",
      "title_tier": 1
    }
  ],
  "stats": {"total": 5, "verified": 2, "partial": 3},
  "elapsed_seconds": 45.2
}
```

### Token usage
```bash
curl http://localhost:8000/stats
```

---

## Output Format

Each contact in results:

| Field | Meaning |
|---|---|
| `status: verified` | Email confirmed ≥70% confidence |
| `status: partial` | Email permutation or LinkedIn URL only |
| `status: not_found` | Person found, no email |
| `title_tier: 1` | C-suite / VP level (primary search) |
| `title_tier: 2` | Director / Manager fallback (shown as `[L2 fallback]`) |
| `confidence` | Email confidence 0–100% |
| `score` | Lead relevance score 0–100 |

---

## Project Structure

```
lead-gen/
├── api/main.py                  FastAPI — /chat, /stats, /health, serves frontend
├── agents/
│   ├── query_parser.py          Groq 70b extracts QueryPlan
│   ├── clarification.py         Asks user for missing details
│   ├── company_search.py        Two-phase discovery + named-company short-circuit
│   ├── people_finder.py         Apollo → website team → dorks, two-tier priority
│   ├── signal_filter.py         HN + GitHub + Wappalyzer signals
│   ├── lead_scoring.py          Groq 70b ranks leads 0–100 (no tools)
│   ├── contact_enricher.py      5-provider email chain
│   └── result_formatter.py      Chat message + JSON, grouped by company
├── graph/
│   ├── state.py                 GraphState + all TypedDicts
│   ├── orchestrator.py          LangGraph StateGraph wiring
│   └── router.py                Guard router + agent router
├── tools/
│   ├── searxng_tool.py          SearXNG search
│   ├── crawl4ai_tool.py         Company website scraping
│   ├── apollo_tool.py           Layer A — Apollo people + company search
│   ├── website_team_tool.py     Layer B — /team /about scraper
│   ├── crosslinked_tool.py      Layer C — Google dorks
│   ├── hn_tool.py               HackerNews Algolia API
│   ├── github_tool.py           GitHub Issues + org members
│   └── wappalyzer_tool.py       Tech stack detection
├── providers/
│   ├── email_provider.py        ABC base class
│   ├── hunter_provider.py       Level 1 — Hunter.io API
│   ├── permutator_provider.py   Level 2 — SMTP verified (dnspython MX)
│   ├── harvester_provider.py    Level 3 — OSINT
│   ├── google_dork_provider.py  Level 4 — SearXNG dork
│   └── website_contact_provider.py  Level 5 — Crawl4AI /contact /about
├── utils/
│   ├── role_normalizer.py       Role expansion + two-tier fallback map
│   ├── human_behavior.py        Delay utilities (kept for future use)
│   ├── session_manager.py       Cookie persistence (kept for future use)
│   └── rate_limiter.py          Rolling-window rate limiter (kept for future use)
├── frontend/index.html          React 18 dashboard (CDN, no npm)
├── docker-compose.yml           SearXNG on port 8080
├── requirements.txt
├── QUERY_GUIDE.md               Full query reference for end users
└── CLAUDE.md                    Project context + build checklist
```

---

## Known Limits

| Constraint | Detail |
|---|---|
| Groq free tier | 100K tokens/day for 70b model — ~6–8 full pipeline runs/day; resets midnight UTC |
| Apollo free tier | 50 contact export credits/month; search is unlimited |
| Hunter free tier | 50 email lookups/month |
| SearXNG required | All web search routes through Docker instance — run `docker-compose up -d` first |
| SMTP on port 25 | Many ISPs block outbound port 25 — Permutator falls back to 30% confidence |
