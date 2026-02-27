"""vLLM provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr


class VLLMConfig(BaseModel):
    """Configuration for a local vLLM server.

    vLLM exposes an OpenAI-compatible API, so this config is translated
    into an ``OpenAIConfig`` by :func:`create_vllm_provider`.

    Attributes:
        model: Model name loaded by the vLLM server (required).
        base_url: Base URL of the vLLM OpenAI-compatible endpoint.
        api_key: API key (vLLM usually needs no auth).
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature.
        timeout: HTTP request timeout in seconds. Increase for servers that load
            models on first request (e.g. Ollama cold start).
    """

    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: SecretStr = SecretStr("none")
    max_tokens: int = 1024
    temperature: float = 0.7
    timeout: float = 120.0
    include_stream_usage: bool = False
    """When True, request token usage in streaming responses."""
