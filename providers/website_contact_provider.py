import asyncio
import logging
import re

from providers.email_provider import EmailProvider

logger = logging.getLogger(__name__)

_EMAIL_RE      = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}')
_CONTACT_PATHS = ["/contact", "/about", "/team", "/contact-us", "/about-us"]
_CRAWL_TIMEOUT = 12.0


class WebsiteContactProvider(EmailProvider):
    """
    Level 5 — Last resort. Crawls the company's contact/about/team pages
    using a single shared AsyncWebCrawler context (avoids spawning 5 browsers).
    Returns the first email found that matches the company domain.
    """

    @property
    def name(self) -> str:
        return "website_contact"

    async def find(self, first_name: str, last_name: str, domain: str) -> dict:
        try:
            from crawl4ai import AsyncWebCrawler
        except ImportError:
            return {"email": None, "confidence": 0, "source": self.name}

        base_url = f"https://{domain}"

        # Single crawler context shared across all paths — avoids 5× browser spawns
        try:
            async with AsyncWebCrawler(verbose=False) as crawler:
                for path in _CONTACT_PATHS:
                    url = f"{base_url}{path}"
                    try:
                        result = await asyncio.wait_for(
                            crawler.arun(url=url),
                            timeout=_CRAWL_TIMEOUT,
                        )
                        text = result.markdown or result.extracted_content or ""
                    except (asyncio.TimeoutError, Exception):
                        continue

                    for email in _EMAIL_RE.findall(text):
                        if (domain in email
                                and "noreply" not in email
                                and "no-reply" not in email):
                            logger.info("WebsiteContact: found %s on %s", email, url)
                            return {"email": email, "confidence": 40, "source": self.name}
        except Exception as e:
            logger.warning("WebsiteContactProvider: crawler error — %s", e)

        return {"email": None, "confidence": 0, "source": self.name}
