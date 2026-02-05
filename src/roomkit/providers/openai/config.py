"""OpenAI provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr


class OpenAIConfig(BaseModel):
    """OpenAI AI provider configuration.

    Attributes:
        api_key: API key for authentication.
        base_url: Custom base URL for OpenAI-compatible APIs (e.g., Ollama, LM Studio,
            Azure OpenAI, or other providers). If None, uses the default OpenAI API.
        model: Model identifier to use.
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature.
    """

    api_key: SecretStr
    base_url: str | None = None
    model: str = "gpt-4o"
    max_tokens: int = 1024
    temperature: float = 0.7
