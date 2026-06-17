"""Azure AI Studio provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr


class AzureAIConfig(BaseModel):
    """Azure AI Studio provider configuration.

    Uses the OpenAI-compatible Chat Completions API exposed by Azure AI Foundry
    deployments (DeepSeek, GPT-4o, Mistral, etc.).

    Attributes:
        api_key: Azure API key for authentication.
        azure_endpoint: Azure AI Foundry project endpoint URL.
        api_version: Azure API version string.
        model: Deployment name (no default — user must specify).
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature.
        timeout: HTTP request timeout in seconds.
    """

    api_key: SecretStr
    azure_endpoint: str
    api_version: str = "2024-12-01-preview"
    model: str
    max_tokens: int = 1024
    temperature: float = 0.7
    timeout: float = 30.0
    max_retries: int = 0
    """SDK-level retry count. Default 0 because RoomKit's RetryPolicy
    handles retries at the right layer with proper backoff and fallback."""
    include_stream_usage: bool = False
    """When True, request token usage in streaming responses."""
    use_max_completion_tokens: bool = False
    """Send the output cap as ``max_completion_tokens`` rather than the
    deprecated ``max_tokens``. Required by newer Azure-hosted OpenAI models;
    leave False for deployments that only understand ``max_tokens``."""
    supports_custom_temperature: bool = True
    """When False, ``temperature`` is omitted — reasoning deployments accept
    only the default and reject any other value with HTTP 400."""
    reasoning_effort: str | None = None
    """Reasoning depth for reasoning deployments (``"low"``/``"medium"``/
    ``"high"``); ``None`` uses the model default. Only sent for models that
    accept it."""
