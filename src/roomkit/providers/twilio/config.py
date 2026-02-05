"""Twilio provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr


class TwilioConfig(BaseModel):
    """Twilio SMS provider configuration."""

    account_sid: str
    auth_token: SecretStr
    from_number: str
    messaging_service_sid: str | None = None
    timeout: float = 10.0

    @property
    def api_url(self) -> str:
        return f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
