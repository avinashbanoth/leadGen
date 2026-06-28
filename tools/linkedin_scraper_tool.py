import logging
import re

from langchain_core.tools import tool

from utils.rate_limiter import rate_limiter
from utils.human_behavior import random_delay, human_scroll, random_mouse_move
from utils import session_manager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LinkedIn search URL — people search scoped to keywords (title + company)
# ---------------------------------------------------------------------------

_SEARCH_URL = (
    "https://www.linkedin.com/search/results/people/"
    "?keywords={keywords}&origin=GLOBAL_SEARCH_HEADER"
)

# ---------------------------------------------------------------------------
# CSS selectors — stored as constants so one-line updates survive DOM changes
# ---------------------------------------------------------------------------

_SEL_CARD  = "li.reusable-search__result-container"
_SEL_NAME  = ".entity-result__title-text a span[aria-hidden='true']"
_SEL_TITLE = ".entity-result__primary-subtitle"
_SEL_LINK  = ".entity-result__title-text a"


# ---------------------------------------------------------------------------
# Browser helpers
# ---------------------------------------------------------------------------

async def _new_stealth_page(context):
    """Creates a Playwright page and applies playwright-stealth if available."""
    page = await context.new_page()
    try:
        from playwright_stealth import stealth_async
        await stealth_async(page)
    except Exception:
        pass  # Camoufox already patches most fingerprints; stealth is best-effort
    return page


async def _extract_text(page, selector: str) -> str | None:
    """Returns stripped inner text of the first matching element, or None."""
    el = await page.query_selector(selector)
    if not el:
        return None
    text = (await el.inner_text()).strip()
    return text or None


# ---------------------------------------------------------------------------
# Result parser
# ---------------------------------------------------------------------------

async def _parse_card(card) -> dict | None:
    """
    Extracts PersonData fields from a single LinkedIn search result card.
    Returns None if the card is missing a name (e.g. LinkedIn Member / private).
    """
    try:
        name_el  = await card.query_selector(_SEL_NAME)
        title_el = await card.query_selector(_SEL_TITLE)
        link_el  = await card.query_selector(_SEL_LINK)

        name  = (await name_el.inner_text()).strip()  if name_el  else ""
        title = (await title_el.inner_text()).strip() if title_el else ""
        href  = await link_el.get_attribute("href")  if link_el  else ""

        if not name or name.lower() == "linkedin member":
            return None

        linkedin_url = ""
        if href:
            match = re.search(r"(https?://[^?]+linkedin\.com/in/[^/?#]+)", href)
            if match:
                linkedin_url = match.group(1)

        return {
            "name"        : name,
            "title"       : title,
            "title_score" : 0.0,   # scored later by People Finder Agent
            "company"     : "",    # stamped by the caller after extraction
            "linkedin_url": linkedin_url,
            "email"       : None,
            "phone"       : None,
            "source"      : "linkedin_scraper",
        }
    except Exception as e:
        logger.debug("_parse_card error: %s", e)
        return None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
async def search_linkedin_people_browser(
    company_name: str,
    target_titles: list[str],
    max_results: int = 5,
) -> list[dict]:
    """
    Layer 2 LinkedIn people search using a stealth Firefox browser (Camoufox + playwright-stealth).
    Use this when Layer 1 (Voyager API) returns empty results or is rate-limited.
    Searches LinkedIn for people at company_name matching any of the given title variants.
    Tries each title in order and stops as soon as results are found.
    Returns a list of PersonData-shaped dicts with source='linkedin_scraper'.
    Returns [] if the session is invalid or the browser fails — signals Layer 3 escalation.
    """
    try:
        from camoufox.async_api import AsyncCamoufox
    except ImportError:
        logger.error("camoufox not installed — Layer 2 unavailable. Run: pip install camoufox")
        return []

    people: list[dict] = []

    try:
        async with AsyncCamoufox(headless=True) as browser:
            context = await browser.new_context()
            page    = await _new_stealth_page(context)

            # Full bootstrap: load cookies → verify session → warm up with feed/notifications/network
            ready = await session_manager.bootstrap(context, page)
            if not ready:
                logger.warning("Layer 2: no valid LinkedIn session — returning [].")
                await context.close()
                return []

            for title in target_titles:
                if len(people) >= max_results:
                    break

                await rate_limiter.check_search()
                await random_delay(2000, 5000)
                await random_mouse_move(page)

                keywords = f"{title} {company_name}"
                url = _SEARCH_URL.format(
                    keywords=keywords.replace(" ", "%20").replace("&", "%26")
                )

                try:
                    await page.goto(url, timeout=20000, wait_until="domcontentloaded")
                    await random_delay(2000, 4000)
                    await human_scroll(page, total_px=400)
                    await random_delay(1000, 2000)
                except Exception as e:
                    logger.warning("Layer 2: page load failed (%s): %s", url, e)
                    continue

                cards = await page.query_selector_all(_SEL_CARD)
                if not cards:
                    logger.info("Layer 2: 0 cards for '%s' @ '%s'", title, company_name)
                    continue

                for card in cards:
                    if len(people) >= max_results:
                        break
                    await rate_limiter.check_profile()
                    person = await _parse_card(card)
                    if person:
                        person["company"] = company_name
                        people.append(person)

                if people:
                    logger.info(
                        "Layer 2 found %d result(s) for '%s' @ '%s'",
                        len(people), title, company_name,
                    )
                    break

            await context.close()

    except Exception as e:
        logger.error("Layer 2 browser session failed: %s", e)
        return []

    if not people:
        logger.info("Layer 2 returned no results for '%s' — escalate to Layer 3.", company_name)

    return people


@tool
async def get_linkedin_profile_browser(linkedin_url: str) -> dict:
    """
    Layer 2 profile contact fetch — visits a LinkedIn profile URL in a stealth browser
    and extracts any visible contact info (email, phone, website).
    Use this to enrich a profile found by search_linkedin_people_browser.
    Returns dict with email, phone, website fields (None where not found).
    Returns {"error": "..."} if session is invalid or navigation fails.
    """
    try:
        from camoufox.async_api import AsyncCamoufox
    except ImportError:
        return {"error": "camoufox not installed — Layer 2 unavailable."}

    try:
        async with AsyncCamoufox(headless=True) as browser:
            context = await browser.new_context()
            page    = await _new_stealth_page(context)

            # Light bootstrap: load cookies + verify only — no warm-up for single profile fetch
            loaded = await session_manager.load(context)
            if not loaded:
                await context.close()
                return {"error": "No saved LinkedIn session for Layer 2."}

            valid = await session_manager.is_logged_in(page)
            if not valid:
                await context.close()
                return {"error": "Saved LinkedIn session is expired."}

            await rate_limiter.check_profile()
            await random_delay(2000, 5000)
            await random_mouse_move(page)

            try:
                await page.goto(linkedin_url, timeout=20000, wait_until="domcontentloaded")
                await random_delay(2000, 4000)
                await human_scroll(page, total_px=500)
            except Exception as e:
                await context.close()
                return {"error": f"Profile navigation failed: {e}"}

            # Open contact info modal if the button is present
            # (only available for 1st-degree connections)
            contact_btn = await page.query_selector("a[href*='contact-info']")
            if contact_btn:
                await contact_btn.click()
                await random_delay(1500, 3000)

            email   = await _extract_text(page, "a[href^='mailto:']")
            phone   = await _extract_text(page, ".pv-contact-info__contact-type.ci-phone .t-14")
            website = await _extract_text(page, ".pv-contact-info__contact-type.ci-websites a")

            await context.close()

            return {
                "email"  : email,
                "phone"  : phone,
                "website": website,
            }

    except Exception as e:
        logger.error("Layer 2 profile fetch failed (%s): %s", linkedin_url, e)
        return {"error": str(e)}
