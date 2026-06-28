import logging
import re

import aiohttp

from providers.email_provider import EmailProvider

logger = logging.getLogger(__name__)

SEARXNG_URL = "http://localhost:8080/search"

_EMAIL_RE = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}')


class GoogleDorkProvider(EmailProvider):
    """
    Level 4 — Uses SearXNG to run targeted Google dork queries that find pages
    where someone's email is publicly listed (PDFs, contact pages, conference sites).
    Dork: site:{domain} OR filetype:pdf "{first_name} {last_name}" email
    """

    @property
    def name(self) -> str:
        return "google_dork"

    async def find(self, first_name: str, last_name: str, domain: str) -> dict:
        full_name = f"{first_name} {last_name}"
        dork = f'(site:{domain} OR filetype:pdf) "{full_name}" email'
        params = {
            "q"         : dork,
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
            logger.warning("Google dork provider failed: %s", e)
            return {"email": None, "confidence": 0, "source": self.name}

        for result in data.get("results", [])[:10]:
            text = f"{result.get('title', '')} {result.get('content', '')}"
            for email in _EMAIL_RE.findall(text):
                if domain in email:
                    logger.info("GoogleDork: found email %s via dork.", email)
                    return {"email": email, "confidence": 55, "source": self.name}

        return {"email": None, "confidence": 0, "source": self.name}
