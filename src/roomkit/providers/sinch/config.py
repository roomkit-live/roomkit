"""Sinch provider configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, SecretStr


class SinchConfig(BaseModel):
    """Sinch SMS provider configuration."""

    service_plan_id: str
    api_token: SecretStr
    from_number: str
    region: Literal["us", "eu", "au", "br", "ca"] = "us"
    webhook_secret: SecretStr | None = None
    timeout: float = 10.0

    @property
    def api_url(self) -> str:
        return f"https://{self.region}.sms.api.sinch.com/xms/v1/{self.service_plan_id}/batches"
