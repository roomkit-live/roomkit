"""Azure AI Studio provider configuration."""

from __future__ import annotations

from typing import Any

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
    extra_body: dict[str, Any] | None = None
    """Extra JSON fields merged into every request body via the SDK's
    ``extra_body`` — for deployment-specific params the OpenAI schema omits.
    ``None`` sends a vanilla body."""


class AzureImageConfig(BaseModel):
    """Azure OpenAI image-generation provider configuration (RFC §25).

    Configures the images endpoint of an Azure OpenAI resource — the same
    ``gpt-image-*`` lineup :class:`~roomkit.providers.openai.config.OpenAIImageConfig`
    reaches on openai.com, deployed under a name the resource owner chose.
    Separate from :class:`AzureAIConfig` for the reason the OpenAI configs are
    separate: a different endpoint, a disjoint model lineup, and none of the
    chat request fields mean anything to it.

    Attributes:
        api_key: Azure API key for authentication.
        azure_endpoint: Azure OpenAI resource endpoint URL.
        api_version: Azure API version string. The default is the version
            Azure's image-generation documentation currently requires; older
            versions predate ``gpt-image-1`` and reject it.
        model: Deployment name (no default — deployment names are chosen per
            Azure resource, so there is nothing sensible to guess).
        quality: ``"low"`` | ``"medium"`` | ``"high"`` | ``"auto"``, or ``None``
            for the deployment's default.
        background: ``"transparent"`` | ``"opaque"`` | ``"auto"``. Transparent
            requires a ``png`` or ``webp`` output format.
        output_format: ``"png"`` | ``"jpeg"``. ``None`` leaves the vendor
            default. Azure does not offer ``webp`` on this endpoint.
        timeout: HTTP request timeout in seconds. Higher than the chat default
            because a high-quality image routinely takes more than 30s.
        max_retries: SDK-level retry count. 0 because RoomKit's RetryPolicy
            handles retries at the right layer.
    """

    api_key: SecretStr
    azure_endpoint: str
    api_version: str = "2025-04-01-preview"
    model: str
    quality: str | None = None
    background: str | None = None
    output_format: str | None = None
    timeout: float = 120.0
    max_retries: int = 0
