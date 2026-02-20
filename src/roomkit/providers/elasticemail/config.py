"""Elastic Email provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr, field_validator


class ElasticEmailConfig(BaseModel):
    """Elastic Email provider configuration."""

    api_key: SecretStr
    from_email: str
    from_name: str | None = None
    is_transactional: bool = True
    base_url: str = "https://api.elasticemail.com/v2/email/send"
    timeout: float = 30.0

    @field_validator("base_url")
    @classmethod
    def _enforce_https(cls, v: str) -> str:
        if not v.startswith("https://"):
            raise ValueError("base_url must use HTTPS (API key is sent in request body)")
        return v
