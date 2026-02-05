"""HTTP webhook provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr


class HTTPProviderConfig(BaseModel):
    """Configuration for the generic HTTP webhook provider."""

    webhook_url: str
    secret: SecretStr | None = None
    timeout: float = 30.0
    headers: dict[str, str] = {}
