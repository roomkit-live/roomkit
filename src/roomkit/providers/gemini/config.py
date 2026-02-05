"""Google Gemini provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr


class GeminiConfig(BaseModel):
    """Google Gemini AI provider configuration."""

    api_key: SecretStr
    model: str = "gemini-2.0-flash"
    max_tokens: int = 1024
    temperature: float = 1.0  # Gemini default
