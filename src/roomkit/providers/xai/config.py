"""xAI provider configuration — chat (Grok text) and realtime (Grok voice)."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr

from roomkit.providers.openai.config import OpenAIConfig


class XAIConfig(OpenAIConfig):
    """xAI (Grok) chat provider configuration.

    xAI serves an OpenAI-compatible Chat Completions API at
    ``https://api.x.ai/v1``, so this **subclasses** :class:`OpenAIConfig` and
    inherits every request field (``temperature``, ``reasoning_effort``,
    ``include_stream_usage``, ``use_max_completion_tokens``,
    ``supports_custom_temperature``, ``extra_body`` …). Inheriting — rather than
    re-declaring them — keeps the two configs from drifting apart: any field the
    inherited :class:`~roomkit.providers.openai.ai.OpenAIAIProvider` reads is
    guaranteed to exist here.

    Only the endpoint and the model default are changed on top. Distinct from
    :class:`XAIRealtimeConfig`, which configures the speech-to-speech WebSocket
    API — same vendor, different protocol.
    """

    base_url: str = "https://api.x.ai/v1"
    """xAI's OpenAI-compatible endpoint. Override only to point at a proxy."""

    model: str = "grok-4.6"
    """Grok model id — see :mod:`roomkit.providers.xai.models` for the curated
    catalog. Aliases the API resolves (``grok-latest``, ``grok-code-fast``, …)
    are accepted too."""

    use_max_completion_tokens: bool = True
    """xAI deprecated ``max_tokens`` in favour of ``max_completion_tokens``, so
    the newer field is the default here (the OpenAI parent defaults to ``False``
    for the benefit of OpenAI-compatible servers that only know ``max_tokens``)."""

    include_stream_usage: bool = True
    """xAI supports ``stream_options.include_usage``, and token accounting is
    worth more than the one extra chunk it costs."""


class XAIImageConfig(BaseModel):
    """xAI (Grok Imagine) image-generation provider configuration (RFC §25).

    Separate from :class:`XAIConfig` because it configures a different
    endpoint with a disjoint model lineup — sampling temperature and reasoning
    effort mean nothing to ``/v1/images``, and an image model means nothing to
    Chat Completions. Distinct from :class:`XAIRealtimeConfig` for the same
    reason — same vendor, three protocols.

    Attributes:
        api_key: xAI API key for authentication.
        base_url: xAI's API endpoint. Override only to point at a proxy.
        model: Grok Imagine image model id — see
            :mod:`roomkit.providers.xai.image_models` for the curated catalog.
            Defaults to the model xAI's own docs recommend for images.
        quality: ``"low"`` | ``"medium"``, or ``None`` for the model's default.
            Only ``grok-imagine-image-2.0`` accepts it; sent only when set.
        resolution: Default resolution tier — ``"1k"`` | ``"2k"`` — applied
            when a call names no size. ``None`` leaves the vendor default. A
            ``size`` passed to ``generate`` wins over this.
        timeout: HTTP request timeout in seconds. Higher than the chat default
            because image synthesis routinely takes more than 30s.
        connect_timeout: TCP connect timeout in seconds, kept apart from
            ``timeout`` so a host that no longer accepts connections is given
            up on in seconds rather than after the read budget.
        max_retries: SDK-level retry count. 0 because RoomKit's RetryPolicy
            handles retries at the right layer.
    """

    api_key: SecretStr
    base_url: str = "https://api.x.ai/v1"
    model: str = "grok-imagine-image-2.0"
    quality: str | None = None
    resolution: str | None = None
    timeout: float = 120.0
    connect_timeout: float = 5.0
    max_retries: int = 0


class XAIRealtimeConfig(BaseModel):
    """Configuration for the xAI Grok Realtime provider.

    Attributes:
        api_key: xAI API key (or set ``XAI_API_KEY`` env var).
        model: Model identifier for the realtime session.
        base_url: WebSocket base URL for the xAI Realtime API.
        voice: Default voice — ``eve``, ``ara``, ``rex``, ``sal``, ``leo``.
        transcription_model: Model used for input audio transcription.
    """

    api_key: SecretStr
    model: str = "grok-2-audio"
    base_url: str = "wss://api.x.ai/v1/realtime"
    voice: str = "eve"
    transcription_model: str = "grok-2-audio"
