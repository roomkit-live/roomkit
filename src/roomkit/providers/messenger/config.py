"""Facebook Messenger provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr


class MessengerConfig(BaseModel):
    """Facebook Messenger provider configuration."""

    page_access_token: SecretStr
    app_secret: SecretStr | None = None
    api_version: str = "v21.0"
    timeout: float = 30.0
    connect_timeout: float = 5.0
    """TCP connect timeout in seconds, separate from the request ``timeout``."""

    @property
    def base_url(self) -> str:
        return f"https://graph.facebook.com/{self.api_version}/me/messages"
