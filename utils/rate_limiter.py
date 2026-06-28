import asyncio
import logging
import time

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LinkedIn-safe limits (from architecture doc)
# ---------------------------------------------------------------------------

MAX_PROFILES_PER_HOUR = 30
MAX_SEARCHES_PER_DAY  = 15

HOUR_SECONDS = 3600
DAY_SECONDS  = 86400


# ---------------------------------------------------------------------------
# RateLimiter — singleton, tracks timestamps of every LinkedIn action
# ---------------------------------------------------------------------------

class RateLimiter:
    def __init__(self):
        self._profile_timestamps: list[float] = []
        self._search_timestamps:  list[float] = []

    def _purge_old(self, timestamps: list[float], window: int) -> list[float]:
        cutoff = time.time() - window
        return [t for t in timestamps if t > cutoff]

    async def check_profile(self) -> None:
        """
        Call before every LinkedIn profile view.
        Sleeps until the hourly profile window has capacity if the limit is hit.
        """
        self._profile_timestamps = self._purge_old(self._profile_timestamps, HOUR_SECONDS)

        if len(self._profile_timestamps) >= MAX_PROFILES_PER_HOUR:
            oldest = self._profile_timestamps[0]
            wait_seconds = HOUR_SECONDS - (time.time() - oldest) + 1
            logger.warning(
                "Profile rate limit hit (%d/hr). Waiting %.0fs.",
                MAX_PROFILES_PER_HOUR,
                wait_seconds,
            )
            await asyncio.sleep(wait_seconds)
            self._profile_timestamps = self._purge_old(self._profile_timestamps, HOUR_SECONDS)

        self._profile_timestamps.append(time.time())

    async def check_search(self) -> None:
        """
        Call before every LinkedIn search.
        Sleeps until the daily search window has capacity if the limit is hit.
        """
        self._search_timestamps = self._purge_old(self._search_timestamps, DAY_SECONDS)

        if len(self._search_timestamps) >= MAX_SEARCHES_PER_DAY:
            oldest = self._search_timestamps[0]
            wait_seconds = DAY_SECONDS - (time.time() - oldest) + 1
            logger.warning(
                "Search rate limit hit (%d/day). Waiting %.0fs.",
                MAX_SEARCHES_PER_DAY,
                wait_seconds,
            )
            await asyncio.sleep(wait_seconds)
            self._search_timestamps = self._purge_old(self._search_timestamps, DAY_SECONDS)

        self._search_timestamps.append(time.time())

    @property
    def profiles_used(self) -> int:
        self._profile_timestamps = self._purge_old(self._profile_timestamps, HOUR_SECONDS)
        return len(self._profile_timestamps)

    @property
    def searches_used(self) -> int:
        self._search_timestamps = self._purge_old(self._search_timestamps, DAY_SECONDS)
        return len(self._search_timestamps)


# ---------------------------------------------------------------------------
# Shared singleton — import this instance, do not instantiate RateLimiter directly
# ---------------------------------------------------------------------------

rate_limiter = RateLimiter()
