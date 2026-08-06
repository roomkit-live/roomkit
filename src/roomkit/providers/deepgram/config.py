"""Deepgram Voice Agent provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr


class DeepgramAgentConfig(BaseModel):
    """Configuration for the Deepgram Voice Agent speech-to-speech provider.

    Deepgram composes an agent from three independently chosen pieces — ``listen``
    (speech-to-text), ``think`` (the LLM) and ``speak`` (text-to-speech) — so the
    defaults here name one model per stage rather than a single end-to-end model.
    Every field is overridable per session through ``connect(provider_config=...)``.

    Distinct from :class:`~roomkit.voice.stt.deepgram.DeepgramConfig`, which
    configures the batch/streaming transcription API — same vendor, different
    protocol and different endpoint.

    Attributes:
        api_key: Deepgram API key (sent as ``Authorization: Token <key>``).
        base_url: Voice Agent WebSocket URL. Point at
            ``wss://api.eu.deepgram.com/v1/agent/converse`` for EU processing.
        listen_model: Speech-to-text model (e.g. ``nova-3``, ``flux-general-en``).
        listen_version: API version for the listen provider — required by Flux
            (``"v2"``); leave ``None`` for Nova.
        listen_language: Language code for transcription (e.g. ``"fr"``).
        think_provider: LLM provider type (``open_ai``, ``anthropic``, ``google``…).
        think_model: LLM model id served by ``think_provider``.
        speak_model: Aura voice id — see :mod:`roomkit.providers.deepgram.voices`.
        speak_language: Language code for synthesis, when the voice supports it.
        greeting: Optional line the agent speaks as soon as the session opens.
        keepalive_interval: Seconds between ``KeepAlive`` messages. Deepgram closes
            connections that go silent; its docs prescribe one every 8 seconds.
    """

    api_key: SecretStr
    base_url: str = "wss://agent.deepgram.com/v1/agent/converse"
    listen_model: str = "nova-3"
    listen_version: str | None = None
    listen_language: str | None = None
    think_provider: str = "open_ai"
    think_model: str = "gpt-4o-mini"
    speak_model: str = "aura-2-thalia-en"
    speak_language: str | None = None
    greeting: str | None = None
    keepalive_interval: float = 8.0
