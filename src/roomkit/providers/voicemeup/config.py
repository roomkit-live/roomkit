"""VoiceMeUp provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr


class VoiceMeUpConfig(BaseModel):
    """VoiceMeUp SMS provider configuration."""

    username: str
    auth_token: SecretStr
    from_number: str
    environment: str = "production"
    timeout: float = 10.0
    connect_timeout: float = 5.0
    """TCP connect timeout in seconds, separate from the request ``timeout``."""

    @property
    def base_url(self) -> str:
        if self.environment == "sandbox":
            return "https://dev-clients.voicemeup.com/api/v1.1/json/"
        return "https://clients.voicemeup.com/api/v1.1/json/"
