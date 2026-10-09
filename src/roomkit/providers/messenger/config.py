"""Facebook Messenger provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr, field_validator

from roomkit.providers.url_safety import validate_api_base_url


class MessengerConfig(BaseModel):
    """Facebook Messenger provider configuration."""

    page_access_token: SecretStr
    app_secret: SecretStr | None = None
    api_version: str = "v21.0"
    timeout: float = 30.0
    connect_timeout: float = 5.0
    """TCP connect timeout in seconds, separate from the request ``timeout``."""
    api_base_url: str = "https://graph.facebook.com"
    """The Graph API host; another one (a local fake under test) must be HTTPS,
    or HTTP on this machine only."""

    @field_validator("api_base_url")
    @classmethod
    def _secure_api_base_url(cls, value: str) -> str:
        return validate_api_base_url(value)

    @property
    def base_url(self) -> str:
        return f"{self.api_base_url}/{self.api_version}/me/messages"
