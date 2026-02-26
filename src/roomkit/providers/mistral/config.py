"""Mistral AI provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr


class MistralConfig(BaseModel):
    """Mistral AI provider configuration.

    Attributes:
        api_key: Mistral API key for authentication.
        model: Model identifier (e.g. ``'mistral-large-latest'``,
            ``'pixtral-large-latest'``).
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature.
        server_url: Custom base URL for Mistral-compatible APIs.
            If ``None``, uses the default Mistral endpoint.
    """

    api_key: SecretStr
    model: str = "mistral-large-latest"
    max_tokens: int = 1024
    temperature: float = 0.7
    server_url: str | None = None
