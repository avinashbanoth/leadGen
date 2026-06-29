import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()   # must run before any LLM or tool imports that read .env

from graph.orchestrator import app_graph    # noqa: E402 — load_dotenv must precede this
from utils.api_tracker import tracker       # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_FRONTEND = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")

# ---------------------------------------------------------------------------
# Daily usage tracker — resets automatically at UTC midnight
# ---------------------------------------------------------------------------

_usage: dict = {"date": "", "queries": 0, "estimated_tokens": 0}

_TOKEN_EST = {
    "rejected"             : 800,
    "awaiting_clarification": 1_500,
    "no_results"           : 12_000,
    "complete"             : 16_000,
}


def _track(status: str) -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _usage["date"] != today:
        _usage.update({"date": today, "queries": 0, "estimated_tokens": 0})
    _usage["queries"] += 1
    _usage["estimated_tokens"] += _TOKEN_EST.get(status, 8_000)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    message         : str
    status          : str
    contacts        : list[dict] = []
    companies       : list[dict] = []
    stats           : dict       = {}
    errors          : list[str]  = []
    result_json     : str        = ""
    elapsed_seconds : float      = 0.0
    estimated_tokens: int        = 0


class UsageResponse(BaseModel):
    queries_today         : int
    estimated_tokens_today: int
    token_limit           : int
    tokens_remaining      : int
    reset_in_seconds      : int


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Lead Gen Agent API starting up.")
    yield
    logger.info("Lead Gen Agent API shutting down.")


app = FastAPI(
    title      ="Lead Gen Multi-Agent API",
    description="Takes any natural language query and returns verified B2B contact details.",
    version    ="0.1.0",
    lifespan   =lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins    =["*"],
    allow_credentials=True,
    allow_methods    =["*"],
    allow_headers    =["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_dashboard():
    return FileResponse(_FRONTEND)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/credits")
async def get_credits():
    """Live API key status — Groq key hint + query count, Hunter credits, Apollo health."""
    return await tracker.status()


@app.get("/stats", response_model=UsageResponse)
async def get_stats():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    queries = _usage["queries"] if _usage["date"] == today else 0
    tokens  = _usage["estimated_tokens"] if _usage["date"] == today else 0

    now           = datetime.now(timezone.utc)
    next_midnight = datetime(now.year, now.month, now.day, tzinfo=timezone.utc) + timedelta(days=1)
    reset_in      = int((next_midnight - now).total_seconds())

    limit = 100_000
    return UsageResponse(
        queries_today         =queries,
        estimated_tokens_today=tokens,
        token_limit           =limit,
        tokens_remaining      =max(0, limit - tokens),
        reset_in_seconds      =reset_in,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    logger.info("Received query: %s", request.query)
    tracker.record_query()
    t0 = time.monotonic()

    initial_state = {
        "query"      : request.query,
        "query_plan" : {},
        "companies"  : [],
        "people"     : [],
        "signals"    : [],
        "lead_score" : [],
        "contacts"   : [],
        "errors"     : [],
        "messages"   : [],
        "status"     : "started",
    }

    try:
        final_state = await app_graph.ainvoke(initial_state)
    except Exception as e:
        logger.error("Graph execution failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    elapsed  = round(time.monotonic() - t0, 2)

    # Surface any Groq rate-limit errors to the tracker
    for err in final_state.get("errors", []):
        if "429" in err or "rate_limit" in err.lower() or "tpd" in err.lower():
            tracker.record_error(err)
            break

    messages = final_state.get("messages", [])
    message  = messages[-1] if messages else "No response generated."
    status   = final_state.get("status", "unknown")
    contacts = final_state.get("contacts", [])
    companies = final_state.get("companies", [])
    errors   = final_state.get("errors", [])
    result_json = final_state.get("result_json", "")

    stats = {}
    if result_json:
        try:
            stats = json.loads(result_json).get("stats", {})
        except Exception:
            pass

    _track(status)

    return ChatResponse(
        message         =message,
        status          =status,
        contacts        =contacts,
        companies       =companies,
        stats           =stats,
        errors          =errors,
        result_json     =result_json,
        elapsed_seconds =elapsed,
        estimated_tokens=_TOKEN_EST.get(status, 8_000),
    )
