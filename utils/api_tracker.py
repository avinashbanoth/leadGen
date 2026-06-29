"""
Singleton API usage tracker — Groq, Hunter, Apollo.

Reads .env on every status() call via load_dotenv(override=True),
so swapping the key in .env is reflected immediately without restarting
the server. Per-key query counters reset automatically when the key changes.
"""

import logging
import os

import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_HUNTER_URL = "https://api.hunter.io/v2/account"
_APOLLO_URL = "https://api.apollo.io/v1/auth/health"


class APITracker:
    def __init__(self) -> None:
        self._groq_key    : str = ""
        self._queries     : int = 0
        self._groq_errors : int = 0
        self._last_error  : str = ""

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refresh(self) -> str:
        """
        Re-reads GROQ_API_KEY from .env each call.
        Resets per-key counters whenever the key value changes.
        """
        load_dotenv(override=True)
        key = os.getenv("GROQ_API_KEY", "")
        if key != self._groq_key:
            self._groq_key    = key
            self._queries     = 0
            self._groq_errors = 0
            self._last_error  = ""
        return key

    @staticmethod
    def _hint(key: str) -> str:
        if not key:
            return "not configured"
        if len(key) <= 12:
            return key[:4] + "···"
        return key[:8] + "···" + key[-4:]

    # ------------------------------------------------------------------
    # Call-site hooks (called from api/main.py — no agent changes needed)
    # ------------------------------------------------------------------

    def record_query(self) -> None:
        """Increment per-key query count. Call once at the top of /chat."""
        self._refresh()
        self._queries += 1

    def record_error(self, msg: str) -> None:
        """Record a Groq rate-limit or API error seen in the pipeline errors list."""
        self._groq_errors += 1
        self._last_error = msg[:140]

    # ------------------------------------------------------------------
    # Remote credit checks
    # ------------------------------------------------------------------

    async def _hunter_credits(self) -> dict:
        load_dotenv(override=True)
        key = os.getenv("HUNTER_API_KEY", "")
        if not key:
            return {"configured": False}
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(_HUNTER_URL, params={"api_key": key})
                data = r.json()
            searches = data.get("data", {}).get("requests", {}).get("searches", {})
            return {
                "configured": True,
                "used"      : searches.get("used", 0),
                "available" : searches.get("available", 50),
            }
        except Exception as exc:
            logger.debug("api_tracker: Hunter fetch failed — %s", exc)
            return {"configured": True, "fetch_error": True, "used": 0, "available": 50}

    async def _apollo_status(self) -> dict:
        load_dotenv(override=True)
        key = os.getenv("APOLLO_API_KEY", "")
        if not key:
            return {"configured": False}
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(_APOLLO_URL, params={"api_key": key})
                data = r.json()
            return {
                "configured": True,
                "active"    : bool(data.get("is_logged_in", False)),
            }
        except Exception as exc:
            logger.debug("api_tracker: Apollo health failed — %s", exc)
            return {"configured": True, "active": False, "fetch_error": True}

    # ------------------------------------------------------------------
    # Status snapshot — returned by GET /api/credits
    # ------------------------------------------------------------------

    async def status(self) -> dict:
        import asyncio
        key = self._refresh()
        # Run Hunter + Apollo concurrently — each has an 8s timeout; sequential = 16s worst case
        hunter, apollo = await asyncio.gather(
            self._hunter_credits(),
            self._apollo_status(),
            return_exceptions=False,
        )
        return {
            "groq": {
                "configured" : bool(key),
                "key_hint"   : self._hint(key),
                "queries"    : self._queries,
                "errors"     : self._groq_errors,
                "last_error" : self._last_error,
            },
            "hunter": hunter,
            "apollo": apollo,
        }


# Module-level singleton — imported by api/main.py
tracker = APITracker()
