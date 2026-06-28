import asyncio
import random


# ---------------------------------------------------------------------------
# Timing constants — tuned to realistic human ranges
# ---------------------------------------------------------------------------

DELAY_BETWEEN_ACTIONS_MS  = (800, 3000)    # pause between page interactions
DELAY_BETWEEN_PROFILES_MS = (8000, 20000)  # pause between LinkedIn profile views
DELAY_BETWEEN_SEARCHES_MS = (30000, 90000) # pause between LinkedIn searches
KEYSTROKE_INTERVAL_MS     = (80, 220)      # typing speed per character
SCROLL_STEP_PX            = (80, 180)      # pixels per scroll chunk
SCROLL_PAUSE_MS           = (50, 150)      # pause between scroll chunks


# ---------------------------------------------------------------------------
# Delay
# ---------------------------------------------------------------------------

async def random_delay(
    min_ms: int = DELAY_BETWEEN_ACTIONS_MS[0],
    max_ms: int = DELAY_BETWEEN_ACTIONS_MS[1],
) -> None:
    """Async sleep for a random duration between min_ms and max_ms milliseconds."""
    delay = random.uniform(min_ms, max_ms) / 1000
    await asyncio.sleep(delay)


async def profile_delay() -> None:
    """Delay between viewing LinkedIn profiles — 8–20 seconds."""
    await random_delay(*DELAY_BETWEEN_PROFILES_MS)


async def search_delay() -> None:
    """Delay between LinkedIn searches — 30–90 seconds."""
    await random_delay(*DELAY_BETWEEN_SEARCHES_MS)


# ---------------------------------------------------------------------------
# Typing
# ---------------------------------------------------------------------------

async def human_type(page, selector: str, text: str) -> None:
    """
    Types text into a Playwright element with variable keystroke timing.
    Clicks the element first to focus it, then types character by character.
    """
    await page.click(selector)
    await random_delay(200, 500)

    for char in text:
        await page.keyboard.type(char)
        delay = random.uniform(*KEYSTROKE_INTERVAL_MS) / 1000
        await asyncio.sleep(delay)

        # Occasional longer pause — humans hesitate
        if random.random() < 0.08:
            await asyncio.sleep(random.uniform(0.3, 0.8))


# ---------------------------------------------------------------------------
# Scrolling
# ---------------------------------------------------------------------------

async def human_scroll(page, total_px: int = 600, direction: str = "down") -> None:
    """
    Scrolls a Playwright page gradually in chunks with pauses between each chunk.
    Simulates natural reading scroll rather than instant jump to bottom.
    """
    scrolled = 0
    sign = 1 if direction == "down" else -1

    while scrolled < total_px:
        step = random.randint(*SCROLL_STEP_PX)
        step = min(step, total_px - scrolled)
        await page.mouse.wheel(0, sign * step)
        scrolled += step
        await random_delay(*SCROLL_PAUSE_MS)


# ---------------------------------------------------------------------------
# Mouse movement
# ---------------------------------------------------------------------------

async def random_mouse_move(page) -> None:
    """
    Moves the mouse to a random position on the viewport before a click.
    Breaks the pattern of clicks always originating from the same coordinate.
    """
    viewport = page.viewport_size or {"width": 1280, "height": 800}
    x = random.randint(100, viewport["width"] - 100)
    y = random.randint(100, viewport["height"] - 100)
    await page.mouse.move(x, y)
    await random_delay(100, 300)
