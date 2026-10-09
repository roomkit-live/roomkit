"""Sinch provider configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, SecretStr, field_validator

from roomkit.providers.url_safety import validate_api_base_url


class SinchConfig(BaseModel):
    """Sinch SMS provider configuration."""

    service_plan_id: str
    api_token: SecretStr
    from_number: str
    region: Literal["us", "eu", "au", "br", "ca"] = "us"
    webhook_secret: SecretStr | None = None
    timeout: float = 10.0
    connect_timeout: float = 5.0
    """TCP connect timeout in seconds, separate from the request ``timeout``."""
    api_base_url: str | None = None
    """The XMS API host; the region's when omitted. Another one (a local fake
    under test) must be HTTPS, or HTTP on this machine only."""

    @field_validator("api_base_url")
    @classmethod
    def _secure_api_base_url(cls, value: str | None) -> str | None:
        return None if value is None else validate_api_base_url(value)

    @property
    def api_url(self) -> str:
        host = self.api_base_url or f"https://{self.region}.sms.api.sinch.com"
        return f"{host}/xms/v1/{self.service_plan_id}/batches"
