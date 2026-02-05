"""Microsoft Teams provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr


class TeamsConfig(BaseModel):
    """Microsoft Teams Bot Framework configuration."""

    app_id: str
    app_password: SecretStr
    tenant_id: str = "common"
