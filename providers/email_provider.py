from abc import ABC, abstractmethod


class EmailProvider(ABC):
    """
    Abstract base for all email enrichment providers.
    Every provider must implement find() — no direct provider calls from agent code.
    Returns a dict with: email (str|None), confidence (int 0-100), source (str).
    """

    @abstractmethod
    async def find(self, first_name: str, last_name: str, domain: str) -> dict:
        """
        Attempt to find or verify an email address for the given person.

        Args:
            first_name: Person's first name.
            last_name:  Person's last name.
            domain:     Company domain (e.g. "acme.com").

        Returns:
            {
                "email"     : str | None,
                "confidence": int,        # 0–100
                "source"    : str,        # provider name
            }
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in ContactData.tried list."""
