import logging

from providers.email_provider import EmailProvider

logger = logging.getLogger(__name__)


class LinkedInContactProvider(EmailProvider):
    """
    Level 5 — Attempts to fetch contact info from a known LinkedIn profile URL
    using Layer 1 (Voyager API). Only works for 1st-degree connections.
    Requires a valid LinkedIn session in .env.
    """

    @property
    def name(self) -> str:
        return "linkedin_contact"

    async def find(self, first_name: str, last_name: str, domain: str) -> dict:
        # linkedin_url must be passed via the caller — this provider needs it externally.
        # The Contact Enricher sets self._linkedin_url before calling find().
        linkedin_url = getattr(self, "_linkedin_url", None)
        if not linkedin_url:
            return {"email": None, "confidence": 0, "source": self.name}

        from tools.linkedin_api_tool import get_linkedin_contact_info
        try:
            result = await get_linkedin_contact_info.ainvoke({"linkedin_url": linkedin_url})
        except Exception as e:
            logger.warning("LinkedIn contact provider failed: %s", e)
            return {"email": None, "confidence": 0, "source": self.name}

        if "error" in result:
            return {"email": None, "confidence": 0, "source": self.name}

        email = result.get("email")
        return {
            "email"     : email,
            "confidence": 85 if email else 0,
            "source"    : self.name,
        }
