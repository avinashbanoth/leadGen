import logging
import os

import aiohttp

from providers.email_provider import EmailProvider

logger = logging.getLogger(__name__)

HUNTER_URL = "https://api.hunter.io/v2/email-finder"


class HunterProvider(EmailProvider):
    """Level 1 — Hunter.io email finder API (25 free lookups/month)."""

    @property
    def name(self) -> str:
        return "hunter"

    async def find(self, first_name: str, last_name: str, domain: str) -> dict:
        api_key = os.getenv("HUNTER_API_KEY", "")
        if not api_key:
            return {"email": None, "confidence": 0, "source": self.name}

        params = {
            "first_name": first_name,
            "last_name" : last_name,
            "domain"    : domain,
            "api_key"   : api_key,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    HUNTER_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status == 429:
                        logger.warning("Hunter.io rate limit hit.")
                        return {"email": None, "confidence": 0, "source": self.name}
                    response.raise_for_status()
                    data = await response.json()

            result = data.get("data", {})
            email  = result.get("email")
            score  = result.get("score", 0)
            return {"email": email, "confidence": score, "source": self.name}

        except Exception as e:
            logger.warning("Hunter provider failed: %s", e)
            return {"email": None, "confidence": 0, "source": self.name}
