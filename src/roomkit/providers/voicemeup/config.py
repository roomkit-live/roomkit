"""VoiceMeUp provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr, field_validator

from roomkit.providers.url_safety import validate_api_base_url


class VoiceMeUpConfig(BaseModel):
    """VoiceMeUp SMS provider configuration."""

    username: str
    auth_token: SecretStr
    from_number: str
    environment: str = "production"
    timeout: float = 10.0
    connect_timeout: float = 5.0
    """TCP connect timeout in seconds, separate from the request ``timeout``."""
    api_base_url: str | None = None
    """The API host; the environment's when omitted. Another one (a local fake
    under test) must be HTTPS, or HTTP on this machine only."""

    @field_validator("api_base_url")
    @classmethod
    def _secure_api_base_url(cls, value: str | None) -> str | None:
        return None if value is None else validate_api_base_url(value)

    @property
    def base_url(self) -> str:
        host = self.api_base_url or (
            "https://dev-clients.voicemeup.com"
            if self.environment == "sandbox"
            else "https://clients.voicemeup.com"
        )
        return f"{host}/api/v1.1/json/"
