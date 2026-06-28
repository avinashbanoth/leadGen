import logging
import re

import aiohttp

from providers.email_provider import EmailProvider

logger = logging.getLogger(__name__)

SEARXNG_URL = "http://localhost:8080/search"

_EMAIL_RE = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}')


class HarvesterProvider(EmailProvider):
    """
    Level 3 — Searches SearXNG for pages that mention the person's name and domain,
    then extracts any email addresses found in the result snippets.
    Mimics theHarvester-style passive collection without requiring a separate tool.
    """

    @property
    def name(self) -> str:
        return "harvester"

    async def find(self, first_name: str, last_name: str, domain: str) -> dict:
        full_name = f"{first_name} {last_name}"
        query     = f'"{full_name}" "@{domain}"'
        params    = {
            "q"         : query,
            "format"    : "json",
            "categories": "general",
            "language"  : "en",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    SEARXNG_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
        except Exception as e:
            logger.warning("Harvester search failed: %s", e)
            return {"email": None, "confidence": 0, "source": self.name}

        for result in data.get("results", [])[:10]:
            text = f"{result.get('title', '')} {result.get('content', '')}"
            matches = _EMAIL_RE.findall(text)
            for email in matches:
                if domain in email:
                    logger.info("Harvester: found email %s in search snippets.", email)
                    return {"email": email, "confidence": 60, "source": self.name}

        return {"email": None, "confidence": 0, "source": self.name}
