"""SendGrid provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr


class SendGridConfig(BaseModel):
    """SendGrid email provider configuration."""

    api_key: SecretStr
    from_email: str
    from_name: str | None = None
    base_url: str = "https://api.sendgrid.com/v3/mail/send"
    timeout: float = 30.0
