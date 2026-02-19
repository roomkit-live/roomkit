"""HTTP webhook provider configuration."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

from pydantic import BaseModel, Field, SecretStr, field_validator


class HTTPProviderConfig(BaseModel):
    """Configuration for the generic HTTP webhook provider."""

    webhook_url: str
    secret: SecretStr | None = None
    timeout: float = 30.0
    headers: dict[str, str] = Field(default_factory=dict)

    @field_validator("webhook_url")
    @classmethod
    def validate_webhook_url(cls, v: str) -> str:
        """Reject webhook URLs pointing to private/reserved IP ranges."""
        parsed = urlparse(v)
        if not parsed.scheme or not parsed.hostname:
            raise ValueError("webhook_url must be a valid URL with scheme and host")

        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"webhook_url scheme must be http or https, got {parsed.scheme!r}")

        hostname = parsed.hostname

        # Reject localhost variants
        if hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):  # noqa: S104  # nosec B104
            raise ValueError("webhook_url must not point to localhost")

        # Check for private/reserved IP addresses
        try:
            addr = ipaddress.ip_address(hostname)
            if addr.is_private or addr.is_reserved or addr.is_loopback:
                raise ValueError(
                    f"webhook_url must not point to private/reserved address: {hostname}"
                )
            if addr.is_link_local:
                raise ValueError(f"webhook_url must not point to link-local address: {hostname}")
        except ValueError as exc:
            # Re-raise our own ValueErrors
            if "webhook_url" in str(exc):
                raise
            # Not an IP address (it's a hostname) — allow it

        return v
