"""Deepgram Voice Agent provider configuration."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, SecretStr, field_validator


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
        speak_provider: Full ``agent.speak.provider`` dict, sent verbatim — names
            any TTS vendor Deepgram supports (``eleven_labs``, ``cartesia``,
            ``open_ai``, ``aws_polly``…) with that vendor's own field shape.
            Takes precedence over ``speak_model``/``speak_language``.
        speak_endpoint: ``agent.speak.endpoint`` dict (URL + auth headers).
            Required for BYO-key TTS vendors (e.g. ElevenLabs, whose voice id
            rides in the endpoint URL); Deepgram-managed vendors need none.
        greeting: Optional line the agent speaks as soon as the session opens.
        keepalive_interval: Seconds between ``KeepAlive`` messages. Deepgram closes
            connections that go silent; its docs prescribe one every 8 seconds.
        max_prompt_chars: Warn when the system prompt exceeds this many characters.
            Defaults to Deepgram's documented 25,000-character cap for managed
            LLMs, past which Deepgram truncates the prompt (``PROMPT_TOO_LONG``).
            ``None`` disables the warning. Sessions pointing at a bring-your-own
            ``think_endpoint`` are never warned — Deepgram applies no cap there.
    """

    api_key: SecretStr = Field(min_length=1)
    base_url: str = Field(default="wss://agent.deepgram.com/v1/agent/converse", min_length=1)
    listen_model: str = Field(default="nova-3", min_length=1)
    listen_version: str | None = None
    listen_language: str | None = None
    think_provider: str = Field(default="open_ai", min_length=1)
    think_model: str = Field(default="gpt-4o-mini", min_length=1)
    speak_model: str = Field(default="aura-2-thalia-en", min_length=1)
    speak_language: str | None = None
    speak_provider: dict[str, Any] | None = None
    speak_endpoint: dict[str, Any] | None = None
    greeting: str | None = None
    keepalive_interval: float = Field(default=8.0, ge=0)
    max_prompt_chars: int | None = Field(default=25_000, ge=1)

    @field_validator("base_url")
    @classmethod
    def _validate_websocket_url(cls, value: str) -> str:
        if not value.startswith(("ws://", "wss://")):
            raise ValueError("base_url must use ws:// or wss://")
        return value

    @field_validator("speak_provider")
    @classmethod
    def _validate_speak_provider(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is not None and not value.get("type"):
            raise ValueError("speak_provider must include a 'type' field")
        return value
