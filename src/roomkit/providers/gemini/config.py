"""Google Gemini provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr


class GeminiConfig(BaseModel):
    """Google Gemini AI provider configuration."""

    api_key: SecretStr
    model: str = "gemini-3.1-flash-lite-preview"
    max_tokens: int = 1024
    temperature: float = 1.0  # Gemini default
    thinking_level: str | None = None
    """Thinking level for Gemini 3.1 models: minimal, low, medium, high."""
