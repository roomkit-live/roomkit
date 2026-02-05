"""Telnyx provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr


class TelnyxConfig(BaseModel):
    """Telnyx SMS provider configuration."""

    api_key: SecretStr
    from_number: str
    messaging_profile_id: str | None = None
    timeout: float = 10.0
