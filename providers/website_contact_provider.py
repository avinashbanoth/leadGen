import logging
import re

from providers.email_provider import EmailProvider

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}')

_CONTACT_PATHS = ["/contact", "/about", "/team", "/contact-us", "/about-us"]


class WebsiteContactProvider(EmailProvider):
    """
    Level 6 — Last resort. Crawls the company's contact/about/team pages
    with Crawl4AI and extracts any email addresses found in the page content.
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

        for path in _CONTACT_PATHS:
            url = f"{base_url}{path}"
            try:
                async with AsyncWebCrawler(verbose=False) as crawler:
                    result = await crawler.arun(url=url)
                text = result.markdown or result.extracted_content or ""
            except Exception:
                continue

            for email in _EMAIL_RE.findall(text):
                if domain in email and "noreply" not in email and "no-reply" not in email:
                    logger.info("WebsiteContact: found email %s on %s", email, url)
                    return {"email": email, "confidence": 40, "source": self.name}

        return {"email": None, "confidence": 0, "source": self.name}
