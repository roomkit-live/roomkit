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
        reasoning_effort: Reasoning effort for models that expose it
            (``mistral-small-latest``, ``mistral-medium-3-5``): ``"high"``
            streams a reasoning trace before the answer, ``"none"`` omits it.
            ``None`` leaves it to the model's default (Magistral models always
            reason). Overridden per-turn by ``AIContext.thinking_budget``.
    """

    api_key: SecretStr
    model: str = "mistral-large-latest"
    max_tokens: int = 1024
    temperature: float = 0.7
    server_url: str | None = None
    reasoning_effort: str | None = None
