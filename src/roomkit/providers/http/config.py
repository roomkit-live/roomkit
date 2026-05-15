"""HTTP webhook provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field, SecretStr, field_validator

from roomkit.providers.url_safety import validate_public_url


class HTTPProviderConfig(BaseModel):
    """Configuration for the generic HTTP webhook provider."""

    webhook_url: str
    secret: SecretStr | None = None
    timeout: float = 30.0
    headers: dict[str, str] = Field(default_factory=dict)

    @field_validator("webhook_url")
    @classmethod
    def validate_webhook_url(cls, v: str) -> str:
        """Reject webhook URLs pointing to private/reserved hosts.

        Resolves every A/AAAA record at validation time and rejects any
        URL whose host (literal or DNS-resolved) lands in loopback,
        private, link-local, reserved, multicast, or unspecified space.
        Numeric IPv4 forms (``127.1``, ``2130706433``, ``0x7f000001``)
        and the trailing-dot DNS form (``localhost.``) are normalized
        before the check.

        Note: DNS rebinding between validation and HTTP request is not
        defended against here — see :mod:`roomkit.providers.url_safety`
        for the rationale.
        """
        try:
            return validate_public_url(v)
        except ValueError as exc:
            raise ValueError(f"webhook_url rejected: {exc}") from None
