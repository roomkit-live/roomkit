"""OpenRouter provider configuration."""

from __future__ import annotations

from roomkit.providers.openai.config import OpenAIConfig, OpenAIImageConfig


class OpenRouterConfig(OpenAIConfig):
    """OpenRouter AI provider configuration.

    OpenRouter exposes the OpenAI-compatible Chat Completions API for 300+
    models behind a single key, so this **subclasses** :class:`OpenAIConfig`
    and inherits every request field (``temperature``, ``reasoning_effort``,
    ``include_stream_usage``, ``use_max_completion_tokens``,
    ``supports_custom_temperature`` …). Inheriting — rather than re-declaring
    those fields — keeps the two configs from drifting apart: any field the
    inherited :class:`~roomkit.providers.openai.ai.OpenAIAIProvider` reads is
    guaranteed to exist here.

    Only OpenRouter's routing endpoint and optional app-attribution headers
    are added on top.
    """

    base_url: str = "https://openrouter.ai/api/v1"
    """OpenRouter's OpenAI-compatible endpoint. Override only to point at a
    self-hosted proxy."""

    model: str
    """OpenRouter model slug — e.g. ``"anthropic/claude-sonnet-4.5"`` or
    ``"openai/gpt-5.5"``. Required (the value of OpenRouter is choosing the
    model). Browse the full live set with
    :meth:`~roomkit.providers.openrouter.ai.OpenRouterAIProvider.list_models`."""

    site_url: str | None = None
    """Sent as the ``HTTP-Referer`` header. Set to your app/site URL to appear
    on OpenRouter's app-attribution leaderboards and analytics."""

    app_name: str | None = None
    """Sent as the ``X-Title`` header — your app's display name in OpenRouter's
    rankings. Only creates an app page when paired with ``site_url``."""


class OpenRouterImageConfig(OpenAIImageConfig):
    """OpenRouter image-generation provider configuration (RFC §25).

    OpenRouter's Image API routes one request shape to every image model it
    aggregates, so this **subclasses** :class:`OpenAIImageConfig` and inherits
    its request fields (``quality``, ``background``, ``output_format``,
    ``timeout``, ``max_retries``, ``default_headers``) — the Image API accepts
    each of them, forwarding what the routed model understands. Only the
    endpoint and OpenRouter's app-attribution headers are added on top.

    The endpoint is OpenRouter's own (``POST {base_url}/images``), not the
    OpenAI images path — see
    :class:`~roomkit.providers.openrouter.image.OpenRouterImageProvider`.
    """

    base_url: str = "https://openrouter.ai/api/v1"
    """OpenRouter's API root. Override only to point at a self-hosted proxy."""

    model: str
    """OpenRouter image model slug — e.g. ``"google/gemini-3.1-flash-image"``
    or ``"x-ai/grok-imagine-image-2.0"``. Required (the value of OpenRouter is
    choosing the model). Browse the live set at ``GET /api/v1/images/models``."""

    site_url: str | None = None
    """Sent as the ``HTTP-Referer`` header — same attribution role as on
    :class:`OpenRouterConfig`."""

    app_name: str | None = None
    """Sent as the ``X-Title`` header — same attribution role as on
    :class:`OpenRouterConfig`."""
