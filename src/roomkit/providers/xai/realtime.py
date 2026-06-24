"""xAI Grok Realtime API provider for speech-to-speech conversations.

xAI exposes a WebSocket-based realtime API at ``wss://api.x.ai/v1/realtime``
that is wire-compatible with the OpenAI Realtime protocol but uses a flatter
session configuration format and offers native ``web_search`` / ``x_search``
tool types. The shared wire plumbing lives in
:class:`~roomkit.providers.openai.realtime_base.OpenAIRealtimeBase`; only the
session-config shape and a few log lines differ here.

Requires the ``websockets`` package::

    pip install websockets
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import SecretStr

from roomkit.providers.openai.realtime_base import OpenAIRealtimeBase
from roomkit.providers.xai.config import XAIRealtimeConfig
from roomkit.providers.xai.voices import VOICES as _VOICES
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.provider import VoiceInfo

logger = logging.getLogger("roomkit.providers.xai.realtime")

# Voices available on the xAI Realtime API.
XAI_VOICES = ("eve", "ara", "rex", "sal", "leo")


class XAIRealtimeProvider(OpenAIRealtimeBase):
    """Realtime voice provider using the xAI Grok Realtime API.

    Connects via WebSocket to xAI's Realtime API, handling bidirectional
    audio streaming with built-in VAD, transcription, and AI responses.
    The wire protocol is compatible with OpenAI Realtime but uses a
    flatter session config and supports xAI-native tools (``web_search``,
    ``x_search``).

    Requires the ``websockets`` package.

    Example::

        from roomkit.providers.xai.config import XAIRealtimeConfig
        from roomkit.providers.xai.realtime import XAIRealtimeProvider

        config = XAIRealtimeConfig(api_key="xai-...")
        provider = XAIRealtimeProvider(config)
        provider.on_audio(handle_output_audio)
        provider.on_transcription(handle_transcription)

        await provider.connect(session, system_prompt="You are a helpful assistant.")
        await provider.send_audio(session, audio_bytes)
    """

    def __init__(
        self,
        config: XAIRealtimeConfig | None = None,
        *,
        api_key: str | SecretStr | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__()
        if config is not None:
            self._config = config
        else:
            if api_key is None:
                raise ValueError("Either config or api_key must be provided")
            key = SecretStr(api_key) if isinstance(api_key, str) else api_key
            self._config = XAIRealtimeConfig(
                api_key=key,
                model=model or "grok-2-audio",
                base_url=base_url or "wss://api.x.ai/v1/realtime",
            )

        self._model = self._config.model

    @property
    def name(self) -> str:
        return "XAIRealtimeProvider"

    @classmethod
    def available_voices(cls) -> list[VoiceInfo]:
        """Curated, offline catalog of xAI Grok built-in voices.

        Not a closed set — the ``voice`` field also accepts custom voice ids.
        """
        return list(_VOICES)

    # -- Provider-specific extension points ---------------------------------

    @property
    def _log_tag(self) -> str:
        return "xAI"

    @property
    def _recv_task_prefix(self) -> str:
        return "xai_rt_recv"

    @property
    def _websockets_install_hint(self) -> str:
        return "pip install websockets"

    def _connect_url(self) -> str:
        return self._config.base_url

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._config.api_key.get_secret_value()}"}

    def _build_session_config(
        self,
        *,
        system_prompt: str | None,
        voice: str | None,
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        input_sample_rate: int,
        output_sample_rate: int,
        server_vad: bool,
        pc: dict[str, Any],
    ) -> dict[str, Any]:
        # xAI uses a flat session config (not nested like OpenAI GA).
        session_config: dict[str, Any] = {}

        voice_id = voice or self._config.voice
        session_config["voice"] = voice_id

        if system_prompt:
            session_config["instructions"] = system_prompt
        if temperature is not None:
            session_config["temperature"] = temperature

        # Turn detection / VAD
        if server_vad:
            td_type = pc.get("turn_detection_type", "server_vad")
            td: dict[str, Any] = {"type": td_type}
            if pc.get("threshold") is not None:
                td["threshold"] = float(pc["threshold"])
            if pc.get("silence_duration_ms") is not None:
                td["silence_duration_ms"] = int(pc["silence_duration_ms"])
            if pc.get("prefix_padding_ms") is not None:
                td["prefix_padding_ms"] = int(pc["prefix_padding_ms"])
            session_config["turn_detection"] = td

        # Input audio transcription
        transcription_model = pc.get("transcription_model", self._config.transcription_model)
        session_config["input_audio_transcription"] = {"model": transcription_model}

        # Tools — xAI supports both function tools and native tools
        # (web_search, x_search). Function tools are projected to the
        # API-accepted fields; native tools pass through unchanged.
        if tools:
            session_config["tools"] = self._format_session_tools(tools)

        # Audio format — xAI uses nested structure
        input_rate = pc.get("input_audio_rate", input_sample_rate)
        output_rate = pc.get("output_audio_rate", output_sample_rate)
        audio_type = pc.get("audio_format_type", "audio/pcm")
        session_config["audio"] = {
            "input": {"format": {"type": audio_type, "rate": input_rate}},
            "output": {"format": {"type": audio_type, "rate": output_rate}},
        }

        # Modalities
        session_config["modalities"] = pc.get("modalities", ["text", "audio"])

        logger.info(
            "Sending session.update: voice=%s, turn_detection=%s, model=%s",
            voice_id,
            session_config.get("turn_detection"),
            self._config.model,
        )
        return session_config

    # -- Provider-specific logging ------------------------------------------

    async def _on_session_created(self, session: VoiceSession, event: dict[str, Any]) -> None:
        sid = event.get("session", {}).get("id", "")
        logger.info("[xAI] session.created: id=%s (session %s)", sid, session.id)
