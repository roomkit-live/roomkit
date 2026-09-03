"""Azure AI Studio provider configuration."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, SecretStr, field_validator

# Path suffixes that can never belong to an Azure *base* endpoint, because the
# SDK is about to append its own copy of them. Azure hands out a different full
# URL depending on the blade you copied from — the portal's Keys and Endpoint
# gives the resource root, the v1 API documentation says to append
# ``/openai/v1``, and a deployment's page gives the whole target URI — so all
# three are pasted into this field in practice, and only the first works.
#
# A bare trailing ``/openai`` is deliberately NOT in this list: it is
# indistinguishable from the route prefix of an API Management gateway sitting
# in front of the resource, which is a legitimate base.
_REWRITABLE_SUFFIX = re.compile(
    r"(?:/openai/v1|/openai/deployments/[^/]+|/chat/completions|/responses)/?$",
    re.IGNORECASE,
)


def normalize_azure_endpoint(value: str) -> str:
    """Reduce a pasted Azure URL to the resource base the SDK expects.

    Drops the query string (an ``api-version`` copied along with a target URI),
    the fragment, trailing slashes, and any suffix the client re-appends. Host
    and scheme are never touched, so a Foundry ``services.ai.azure.com``
    resource, a ``cognitiveservices.azure.com`` one and an APIM gateway all
    survive unchanged.
    """
    raw = value.strip()
    if not raw:
        return raw
    parts = urlsplit(raw)
    if not parts.scheme or not parts.netloc:
        # Not a URL we can reason about (a bare host, a half-typed value).
        # Leave it alone: refusing here would be a second, worse failure.
        return raw.rstrip("/")
    path = parts.path
    # Loop: a full target URI carries two rewritable suffixes
    # (``/openai/deployments/<name>`` then ``/chat/completions``), and the v1
    # form carries ``/openai/v1`` under ``/chat/completions``.
    for _ in range(4):
        stripped = _REWRITABLE_SUFFIX.sub("", path)
        if stripped == path:
            break
        path = stripped
    return urlunsplit((parts.scheme, parts.netloc, path.rstrip("/"), "", ""))


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
        connect_timeout: TCP connect timeout in seconds, kept apart from
            ``timeout`` so a host that no longer accepts connections is given
            up on in seconds rather than after the read budget.
    """

    api_key: SecretStr
    azure_endpoint: str
    api_version: str = "2024-12-01-preview"
    model: str
    max_tokens: int = 1024
    temperature: float = 0.7
    timeout: float = 30.0
    connect_timeout: float = 5.0
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

    @field_validator("azure_endpoint")
    @classmethod
    def _normalize_endpoint(cls, value: str) -> str:
        """Accept any of the URLs Azure hands out; keep the one the SDK needs.

        ``AsyncAzureOpenAI`` builds ``{azure_endpoint}/openai/deployments/...``
        itself, so a pasted ``/openai/v1`` suffix produces a doubled path and
        an opaque 404 one agent turn later. Normalising here rather than
        refusing keeps every documented copy-paste working.
        """
        return normalize_azure_endpoint(value)

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
        connect_timeout: TCP connect timeout in seconds, kept apart from
            ``timeout`` so a host that no longer accepts connections is given
            up on in seconds rather than after the read budget.
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
    connect_timeout: float = 5.0
    max_retries: int = 0

    @field_validator("azure_endpoint")
    @classmethod
    def _normalize_endpoint(cls, value: str) -> str:
        """Accept any of the URLs Azure hands out; keep the one the SDK needs.

        ``AsyncAzureOpenAI`` builds ``{azure_endpoint}/openai/deployments/...``
        itself, so a pasted ``/openai/v1`` suffix produces a doubled path and
        an opaque 404 one agent turn later. Normalising here rather than
        refusing keeps every documented copy-paste working.
        """
        return normalize_azure_endpoint(value)
