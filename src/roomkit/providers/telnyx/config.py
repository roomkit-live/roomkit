"""Telnyx provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr, field_validator

from roomkit.providers.url_safety import validate_api_base_url


class TelnyxConfig(BaseModel):
    """Telnyx SMS provider configuration."""

    api_key: SecretStr
    from_number: str
    messaging_profile_id: str | None = None
    timeout: float = 10.0
    connect_timeout: float = 5.0
    """TCP connect timeout in seconds, separate from the request ``timeout``."""
    api_base_url: str = "https://api.telnyx.com"
    """Telnyx's API host; another one (a local fake under test) must be HTTPS,
    or HTTP on this machine only."""

    @field_validator("api_base_url")
    @classmethod
    def _secure_api_base_url(cls, value: str) -> str:
        return validate_api_base_url(value)
