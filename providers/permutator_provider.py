import asyncio
import logging
import smtplib
import socket

from providers.email_provider import EmailProvider
from tools.crosslinked_tool import get_email_permutations

logger = logging.getLogger(__name__)

_SMTP_TIMEOUT = 3     # seconds per TCP op
_SMTP_WALL    = 4.0   # hard wall per candidate (catches getfqdn hangs)


async def _smtp_verify(email: str) -> bool:
    """
    Checks if an email address likely exists via SMTP RCPT TO handshake.
    Hard-capped at _SMTP_WALL seconds to prevent getfqdn / port-25 hangs.
    """
    domain = email.split("@")[-1]

    def _check() -> bool:
        try:
            # Resolve MX via getfqdn — can hang; killed by wait_for below
            mx = socket.getfqdn(domain)
            with smtplib.SMTP(timeout=_SMTP_TIMEOUT) as smtp:
                smtp.connect(mx, 25)
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
    then validates each candidate via SMTP RCPT TO.
    Falls back to returning the most common pattern with low confidence if SMTP is blocked.
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

        # Try SMTP verification on top 4 candidates only — avoids 15 × 4s wall time
        for candidate in candidates[:4]:
            valid = await _smtp_verify(candidate)
            if valid:
                logger.info("Permutator: SMTP verified %s", candidate)
                return {"email": candidate, "confidence": 70, "source": self.name}

        # SMTP blocked or all failed — return top candidate with low confidence
        if candidates:
            return {"email": candidates[0], "confidence": 30, "source": self.name}

        return {"email": None, "confidence": 0, "source": self.name}
