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
