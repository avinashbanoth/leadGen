import json
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()   # must run before any LLM or tool imports that read .env

from graph.orchestrator import app_graph   # noqa: E402 — load_dotenv must precede this

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    message    : str
    status     : str
    contacts   : list[dict] = []
    stats      : dict       = {}
    errors     : list[str]  = []
    result_json: str        = ""


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

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main endpoint. Takes a natural language lead generation query and returns
    verified contacts, lead scores, signals, and a human-readable summary.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    logger.info("Received query: %s", request.query)

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

    messages   = final_state.get("messages", [])
    message    = messages[-1] if messages else "No response generated."
    status     = final_state.get("status", "unknown")
    contacts   = final_state.get("contacts", [])
    errors     = final_state.get("errors", [])
    result_json = final_state.get("result_json", "")

    # Extract stats from result_json if available
    stats = {}
    if result_json:
        try:
            stats = json.loads(result_json).get("stats", {})
        except Exception:
            pass

    return ChatResponse(
        message    =message,
        status     =status,
        contacts   =contacts,
        stats      =stats,
        errors     =errors,
        result_json=result_json,
    )
