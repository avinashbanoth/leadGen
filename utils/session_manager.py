import json
import logging
import os
from pathlib import Path

from utils.human_behavior import random_delay, human_scroll

logger = logging.getLogger(__name__)

SESSION_FILE = Path(os.getenv("SESSION_FILE", "session.json"))

# Pages visited during warm-up — normal user behaviour before any search
WARM_UP_URLS = [
    "https://www.linkedin.com/feed/",
    "https://www.linkedin.com/notifications/",
    "https://www.linkedin.com/mynetwork/",
]


# ---------------------------------------------------------------------------
# Cookie persistence
# ---------------------------------------------------------------------------

async def save(context) -> None:
    """
    Saves all cookies from a Playwright browser context to SESSION_FILE.
    Call this after a successful login so the next run skips re-authentication.
    """
    cookies = await context.cookies()
    SESSION_FILE.write_text(json.dumps(cookies, indent=2))
    logger.info("Session saved to %s (%d cookies)", SESSION_FILE, len(cookies))


async def load(context) -> bool:
    """
    Loads cookies from SESSION_FILE into a Playwright browser context.
    Returns True if cookies were loaded, False if no session file exists.
    """
    if not SESSION_FILE.exists():
        logger.info("No session file found at %s — fresh login required.", SESSION_FILE)
        return False

    cookies = json.loads(SESSION_FILE.read_text())
    await context.add_cookies(cookies)
    logger.info("Session loaded from %s (%d cookies)", SESSION_FILE, len(cookies))
    return True


# ---------------------------------------------------------------------------
# Login check
# ---------------------------------------------------------------------------

async def is_logged_in(page) -> bool:
    """
    Navigates to LinkedIn feed and checks for the presence of the nav bar
    that only appears for authenticated users.
    Returns True if the session is valid, False if re-login is needed.
    """
    try:
        await page.goto("https://www.linkedin.com/feed/", timeout=15000)
        await random_delay(1000, 2000)
        nav = await page.query_selector("nav.global-nav")
        return nav is not None
    except Exception as e:
        logger.warning("Login check failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Session warm-up
# ---------------------------------------------------------------------------

async def warm(page) -> None:
    """
    Visits normal LinkedIn pages before any search to establish human-like
    session activity. LinkedIn flags sessions that jump straight to search.
    Spends 15–30 seconds browsing: feed → notifications → network.
    """
    logger.info("Warming up LinkedIn session...")

    for url in WARM_UP_URLS:
        try:
            await page.goto(url, timeout=15000)
            await random_delay(3000, 6000)
            await human_scroll(page, total_px=400)
            await random_delay(2000, 4000)
        except Exception as e:
            logger.warning("Warm-up page failed (%s): %s", url, e)
            continue

    logger.info("Session warm-up complete.")


# ---------------------------------------------------------------------------
# Full session bootstrap — load + check + warm in one call
# ---------------------------------------------------------------------------

async def bootstrap(context, page) -> bool:
    """
    Full session setup called at the start of every LinkedIn browser run.
    1. Load saved cookies
    2. Check if session is still valid
    3. Warm up with normal page browsing
    Returns True if ready to search, False if fresh login is required.
    """
    loaded = await load(context)

    if not loaded:
        return False

    valid = await is_logged_in(page)

    if not valid:
        logger.warning("Saved session is expired — fresh login required.")
        SESSION_FILE.unlink(missing_ok=True)
        return False

    await warm(page)
    return True
