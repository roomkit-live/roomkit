"""SendGrid provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr, field_validator

from roomkit.providers.url_safety import validate_api_base_url


class SendGridConfig(BaseModel):
    """SendGrid email provider configuration."""

    api_key: SecretStr
    from_email: str
    from_name: str | None = None
    base_url: str = "https://api.sendgrid.com/v3/mail/send"
    """The mail send endpoint; another one (a local fake under test) must be
    HTTPS, or HTTP on this machine only: the API key rides every request."""
    timeout: float = 30.0
    connect_timeout: float = 5.0
    """TCP connect timeout in seconds, separate from the request ``timeout``."""

    @field_validator("base_url")
    @classmethod
    def _secure_base_url(cls, value: str) -> str:
        return validate_api_base_url(value)
