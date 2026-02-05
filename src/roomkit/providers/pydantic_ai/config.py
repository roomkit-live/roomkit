"""Pydantic AI provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr


class PydanticAIConfig(BaseModel):
    """Pydantic AI provider configuration."""

    model: str = "openai:gpt-4o"
    api_key: SecretStr | None = None
    temperature: float = 0.7
    max_tokens: int = 1024
