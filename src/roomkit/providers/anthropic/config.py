"""Anthropic provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr


class AnthropicConfig(BaseModel):
    """Anthropic AI provider configuration."""

    api_key: SecretStr
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024
    temperature: float = 0.7
