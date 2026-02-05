"""Elastic Email provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr


class ElasticEmailConfig(BaseModel):
    """Elastic Email provider configuration."""

    api_key: SecretStr
    from_email: str
    from_name: str | None = None
    is_transactional: bool = True
    base_url: str = "https://api.elasticemail.com/v2/email/send"
    timeout: float = 30.0
