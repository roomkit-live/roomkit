"""vLLM provider configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VLLMConfig:
    """Configuration for a local vLLM server.

    vLLM exposes an OpenAI-compatible API, so this config is translated
    into an ``OpenAIConfig`` by :func:`create_vllm_provider`.

    Attributes:
        model: Model name loaded by the vLLM server (required).
        base_url: Base URL of the vLLM OpenAI-compatible endpoint.
        api_key: API key (vLLM usually needs no auth).
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature.
    """

    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "none"
    max_tokens: int = 1024
    temperature: float = 0.7
