"""OpenAI Realtime API provider for speech-to-speech conversations."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import SecretStr

from roomkit.providers.openai.realtime_base import OpenAIRealtimeBase
from roomkit.providers.openai.voices import VOICES as _VOICES
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.provider import VoiceInfo

logger = logging.getLogger("roomkit.providers.openai.realtime")

# Default OpenAI Realtime API endpoint
_DEFAULT_BASE_URL = "wss://api.openai.com/v1/realtime"


class OpenAIRealtimeProvider(OpenAIRealtimeBase):
    """Realtime voice provider using the OpenAI Realtime API.

    Connects via WebSocket to OpenAI's Realtime API (GA), handling
    bidirectional audio streaming with built-in VAD, transcription,
    and AI responses.

    **Audio format constraints (GA API):**

    - ``audio/pcm`` is only accepted at ``24000`` Hz (``rate`` is fixed).
    - ``audio/pcmu`` (G.711 μ-law) and ``audio/pcma`` (G.711 A-law) are
      accepted for 8 kHz telephony; they have no ``rate`` field.
    - Other sample rates are rejected by the API.

    ``input_sample_rate`` / ``output_sample_rate`` must therefore be
    ``24000`` or ``8000``. For 8 kHz, pass ``provider_config["codec"]``
    as ``"pcmu"`` (default) or ``"pcma"``.

    **Note:** the GA API does not accept ``temperature``; passing it
    logs a warning and is ignored.

    Requires the ``websockets`` package.

    Example:
        provider = OpenAIRealtimeProvider(api_key="sk-...", model="gpt-realtime-1.5")
        provider.on_audio(handle_output_audio)
        provider.on_transcription(handle_transcription)

        await provider.connect(session, system_prompt="You are a helpful assistant.")
        await provider.send_audio(session, audio_bytes)
    """

    def __init__(
        self,
        *,
        api_key: str | SecretStr,
        model: str = "gpt-realtime-1.5",
        base_url: str | None = None,
    ) -> None:
        super().__init__()
        self._api_key = SecretStr(api_key) if isinstance(api_key, str) else api_key
        self._model = model
        self._base_url = base_url or _DEFAULT_BASE_URL

    @property
    def name(self) -> str:
        return "OpenAIRealtimeProvider"

    @classmethod
    def available_voices(cls) -> list[VoiceInfo]:
        """Curated, offline catalog of OpenAI Realtime voices (fixed set)."""
        return list(_VOICES)

    # -- Provider-specific extension points ---------------------------------

    @property
    def _log_tag(self) -> str:
        return "OpenAI"

    @property
    def _recv_task_prefix(self) -> str:
        return "openai_rt_recv"

    @property
    def _websockets_install_hint(self) -> str:
        return "pip install 'roomkit[realtime-openai]'"

    def _connect_url(self) -> str:
        return f"{self._base_url}?model={self._model}"

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}

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
        if temperature is not None:
            logger.warning(
                "OpenAI Realtime GA API no longer supports the temperature parameter; ignoring"
            )

        # Validate audio rates up-front — building the format objects raises
        # ValueError for unsupported rates before any WebSocket is opened.
        codec = pc.get("codec", "pcmu")
        input_format = self._build_audio_format(input_sample_rate, codec)
        output_format = self._build_audio_format(output_sample_rate, codec)

        # Build GA session config — audio settings nest under audio.input / audio.output.
        transcription: dict[str, Any] = {"model": pc.get("stt_model", "gpt-4o-transcribe")}
        if pc.get("language"):
            transcription["language"] = pc["language"]
        if pc.get("transcription_prompt"):
            transcription["prompt"] = pc["transcription_prompt"]

        # noise_reduction: "near_field" for headphones/close mic,
        # "far_field" for laptop/conference room speakers.
        nr_type = pc.get("noise_reduction", "far_field")
        audio_input: dict[str, Any] = {
            "format": input_format,
            "transcription": transcription,
            "noise_reduction": {"type": nr_type},
        }
        audio_output: dict[str, Any] = {"format": output_format}
        if voice:
            audio_output["voice"] = voice
        if pc.get("speed") is not None:
            audio_output["speed"] = float(pc["speed"])

        # --- Turn detection / VAD (nested under audio.input in GA) ---
        # Default to semantic_vad — it uses a turn detection model that
        # distinguishes real speech from echo/noise residuals, which is
        # critical for laptop mic+speaker setups where AEC can't suppress
        # 100% of the echo.  server_vad (energy-based) is too sensitive.
        td_type = pc.get("turn_detection_type", "semantic_vad" if server_vad else None)
        turn_detection = self._build_turn_detection(td_type, pc)
        if turn_detection is not None:
            audio_input["turn_detection"] = turn_detection

        session_config: dict[str, Any] = {
            "type": "realtime",
            "output_modalities": ["audio"],
            "audio": {"input": audio_input, "output": audio_output},
        }
        if system_prompt:
            session_config["instructions"] = system_prompt
        if tools:
            session_config["tools"] = self._format_session_tools(tools)

        logger.info("Sending session.update: turn_detection=%s, voice=%s", turn_detection, voice)
        return session_config

    # -- Builders ------------------------------------------------------------

    @staticmethod
    def _build_turn_detection(td_type: str | None, pc: dict[str, Any]) -> dict[str, Any] | None:
        """Build the turn_detection dict for the GA session config."""
        td: dict[str, Any]
        if td_type == "semantic_vad":
            td = {"type": "semantic_vad"}
            if pc.get("eagerness"):
                td["eagerness"] = pc["eagerness"]
            if pc.get("interrupt_response") is not None:
                td["interrupt_response"] = bool(pc["interrupt_response"])
            if pc.get("create_response") is not None:
                td["create_response"] = bool(pc["create_response"])
            return td
        if td_type == "server_vad":
            td = {"type": "server_vad"}
            if pc.get("threshold") is not None:
                td["threshold"] = float(pc["threshold"])
            if pc.get("silence_duration_ms") is not None:
                td["silence_duration_ms"] = int(pc["silence_duration_ms"])
            if pc.get("prefix_padding_ms") is not None:
                td["prefix_padding_ms"] = int(pc["prefix_padding_ms"])
            if pc.get("idle_timeout_ms") is not None:
                td["idle_timeout_ms"] = int(pc["idle_timeout_ms"])
            if pc.get("interrupt_response") is not None:
                td["interrupt_response"] = bool(pc["interrupt_response"])
            if pc.get("create_response") is not None:
                td["create_response"] = bool(pc["create_response"])
            return td
        return None

    @staticmethod
    def _build_audio_format(rate: int, codec: str) -> dict[str, Any]:
        """Map a PCM sample rate to the GA API's audio format object.

        The GA API only accepts:
          * ``audio/pcm`` at 24000 Hz
          * ``audio/pcmu`` (G.711 μ-law) — 8 kHz implied, no ``rate`` field
          * ``audio/pcma`` (G.711 A-law) — 8 kHz implied, no ``rate`` field
        """
        if rate == 24000:
            return {"type": "audio/pcm", "rate": 24000}
        if rate == 8000:
            if codec not in ("pcmu", "pcma"):
                raise ValueError(
                    f"OpenAI Realtime 8 kHz requires codec='pcmu' or 'pcma', got {codec!r}"
                )
            return {"type": f"audio/{codec}"}
        raise ValueError(
            f"OpenAI Realtime API only accepts 24000 Hz (PCM) or 8000 Hz (G.711), got {rate}"
        )

    # -- Provider-specific logging ------------------------------------------

    def _log_usage(
        self,
        session: VoiceSession,
        input_tokens: int,
        output_tokens: int,
        input_details: dict[str, Any],
        output_details: dict[str, Any],
    ) -> None:
        logger.info(
            "[OpenAI] usage: input=%d output=%d "
            "(cached_input=%d, text_input=%d, audio_input=%d, "
            "text_output=%d, audio_output=%d) (session %s)",
            input_tokens,
            output_tokens,
            input_details.get("cached_tokens", 0),
            input_details.get("text_tokens", 0),
            input_details.get("audio_tokens", 0),
            output_details.get("text_tokens", 0),
            output_details.get("audio_tokens", 0),
            session.id,
        )

    async def _on_session_created(self, session: VoiceSession, event: dict[str, Any]) -> None:
        td_type = (
            event.get("session", {})
            .get("audio", {})
            .get("input", {})
            .get("turn_detection", {})
            .get("type")
        )
        logger.info(
            "[OpenAI] session.created: turn_detection=%s (session %s)", td_type, session.id
        )

    async def _on_session_updated(self, session: VoiceSession, event: dict[str, Any]) -> None:
        td_type = (
            event.get("session", {})
            .get("audio", {})
            .get("input", {})
            .get("turn_detection", {})
            .get("type")
        )
        logger.info(
            "[OpenAI] session.updated: turn_detection=%s (session %s)", td_type, session.id
        )
