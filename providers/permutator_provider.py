import asyncio
import logging
import smtplib

from providers.email_provider import EmailProvider
from tools.crosslinked_tool import get_email_permutations

logger = logging.getLogger(__name__)

_SMTP_TIMEOUT = 5
_SMTP_WALL    = 8.0   # hard cap per candidate


async def _smtp_verify(email: str) -> bool:
    """
    Checks if an email likely exists via SMTP RCPT TO handshake.
    Uses dnspython for a proper MX record lookup (not getfqdn).
    Hard-capped at _SMTP_WALL seconds.
    """
    domain = email.split("@")[-1]

    def _check() -> bool:
        try:
            import dns.resolver
            answers = dns.resolver.resolve(domain, "MX")
            mx_host = str(sorted(answers, key=lambda r: r.preference)[0].exchange).rstrip(".")
        except Exception:
            return False

        try:
            with smtplib.SMTP(timeout=_SMTP_TIMEOUT) as smtp:
                smtp.connect(mx_host, 25)
                smtp.helo("verify.local")
                smtp.mail("verify@verify.local")
                code, _ = smtp.rcpt(email)
                return code == 250
        except Exception:
            return False

    try:
        return await asyncio.wait_for(asyncio.to_thread(_check), timeout=_SMTP_WALL)
    except (asyncio.TimeoutError, Exception):
        return False


class PermutatorProvider(EmailProvider):
    """
    Level 2 — Generates email permutations from the person's name and domain,
    then validates each candidate via SMTP RCPT TO (real MX lookup via dnspython).
    Falls back to returning the most common pattern with low confidence if SMTP blocked.
    """

    @property
    def name(self) -> str:
        return "permutator"

    async def find(self, first_name: str, last_name: str, domain: str) -> dict:
        candidates = await get_email_permutations.ainvoke({
            "first_name"    : first_name,
            "last_name"     : last_name,
            "company_domain": domain,
        })

        for candidate in candidates[:4]:
            valid = await _smtp_verify(candidate)
            if valid:
                logger.info("Permutator: SMTP verified %s", candidate)
                return {"email": candidate, "confidence": 70, "source": self.name}

        if candidates:
            return {"email": candidates[0], "confidence": 30, "source": self.name}

        return {"email": None, "confidence": 0, "source": self.name}
