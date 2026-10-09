"""Twilio provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr, field_validator

from roomkit.providers.url_safety import validate_api_base_url


class TwilioConfig(BaseModel):
    """Twilio SMS provider configuration."""

    account_sid: str
    auth_token: SecretStr
    from_number: str
    messaging_service_sid: str | None = None
    timeout: float = 10.0
    connect_timeout: float = 5.0
    """TCP connect timeout in seconds, separate from the request ``timeout``."""
    api_base_url: str = "https://api.twilio.com"
    """Twilio's REST host; another one (a sandbox, a local fake) must be HTTPS,
    or HTTP on this machine only."""

    @field_validator("api_base_url")
    @classmethod
    def _secure_api_base_url(cls, value: str) -> str:
        return validate_api_base_url(value)

    @property
    def api_url(self) -> str:
        return f"{self.api_base_url}/2010-04-01/Accounts/{self.account_sid}/Messages.json"
