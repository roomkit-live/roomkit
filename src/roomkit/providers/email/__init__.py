"""Email provider abstractions and mock implementation."""

from roomkit.providers.email.base import EmailProvider
from roomkit.providers.email.mock import MockEmailProvider

__all__ = [
    "EmailProvider",
    "MockEmailProvider",
]
