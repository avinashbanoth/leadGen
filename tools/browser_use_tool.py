import json
import logging
import os
import re

from langchain_core.tools import tool

from utils.rate_limiter import rate_limiter
from utils.human_behavior import random_delay

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy LLM init — Groq Llama 3.1 8B Instant drives the browser agent
# ---------------------------------------------------------------------------

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        from langchain_groq import ChatGroq
        _llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )
    return _llm


# ---------------------------------------------------------------------------
# Task template — the LLM receives this as its browser navigation goal
# No CSS selectors anywhere: the LLM reads the page and decides what to click
# ---------------------------------------------------------------------------

_TASK_TEMPLATE = (
    "Open https://www.linkedin.com in a browser. "
    "If the page shows a login form, enter email '{username}' and password '{password}' and submit. "
    "Once logged in, search for LinkedIn members who have the job title '{title}' "
    "and currently work at '{company}'. "
    "Collect the full name, current job title, and LinkedIn profile URL "
    "for up to {max_results} people. "
    "Return ONLY a valid JSON array — no explanation, no markdown, JSON only:\n"
    '[{{"name": "Full Name", "title": "Job Title", '
    '"linkedin_url": "https://www.linkedin.com/in/slug"}}]'
)


# ---------------------------------------------------------------------------
# Result parser
# ---------------------------------------------------------------------------

def _extract_people(raw, company_name: str) -> list[dict]:
    """
    Pulls a JSON array from the agent's raw output string and maps each item
    to a PersonData-shaped dict.  Tries JSON parse first, returns [] on any failure.
    """
    text = str(raw) if raw else ""

    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if not match:
        return []

    try:
        items = json.loads(match.group())
    except (json.JSONDecodeError, ValueError):
        return []

    people = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name or name.lower() == "linkedin member":
            continue
        people.append({
            "name"        : name,
            "title"       : str(item.get("title", "")).strip(),
            "title_score" : 0.0,
            "company"     : company_name,
            "linkedin_url": str(item.get("linkedin_url", "")).strip(),
            "email"       : None,
            "phone"       : None,
            "source"      : "browser_use",
        })

    return people


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@tool
async def search_linkedin_people_agent(
    company_name: str,
    target_titles: list[str],
    max_results: int = 5,
) -> list[dict]:
    """
    Layer 3 LinkedIn people search using browser-use + Groq LLM (llama-3.1-8b-instant).
    The LLM drives a real browser with no CSS selectors — it reads the page and decides
    what to click, where to type, and how to navigate.
    Use this when Layer 2 (Camoufox stealth browser) returns empty results.
    Tries each title variant in order and stops as soon as results are found.
    Returns PersonData-shaped dicts with source='browser_use'.
    Returns [] on any failure — signals Layer 4 escalation.
    """
    try:
        from browser_use import Agent
        from browser_use.browser.browser import Browser, BrowserConfig
    except ImportError:
        logger.error("browser-use not installed — Layer 3 unavailable. Run: pip install browser-use")
        return []

    username = os.getenv("LI_USERNAME")
    password = os.getenv("LI_PASSWORD")
    if not username or not password:
        logger.error("Layer 3: LI_USERNAME / LI_PASSWORD not set in .env — cannot authenticate.")
        return []

    for title in target_titles:
        await rate_limiter.check_search()
        await random_delay(3000, 8000)

        task = _TASK_TEMPLATE.format(
            username=username,
            password=password,
            title=title,
            company=company_name,
            max_results=max_results,
        )

        try:
            browser = Browser(config=BrowserConfig(headless=True))
            agent   = Agent(task=task, llm=_get_llm(), browser=browser)
            history = await agent.run()

            # AgentHistoryList exposes final_result(); fall back to str() for older versions
            raw = (
                history.final_result()
                if hasattr(history, "final_result")
                else str(history)
            )

            people = _extract_people(raw, company_name)

            if people:
                logger.info(
                    "Layer 3 found %d result(s) for '%s' @ '%s'",
                    len(people), title, company_name,
                )
                return people[:max_results]

        except Exception as e:
            logger.warning("Layer 3 agent failed (title=%s): %s", title, e)
            continue

    logger.info("Layer 3 returned no results for '%s' — escalate to Layer 4.", company_name)
    return []
