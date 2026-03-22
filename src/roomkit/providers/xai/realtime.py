"""xAI Grok Realtime API provider for speech-to-speech conversations.

xAI exposes a WebSocket-based realtime API at ``wss://api.x.ai/v1/realtime``
that is wire-compatible with the OpenAI Realtime protocol but uses a flatter
session configuration format and offers native ``web_search`` / ``x_search``
tool types.

Requires the ``websockets`` package::

    pip install websockets
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any

from pydantic import SecretStr

from roomkit.providers.xai.config import XAIRealtimeConfig
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.provider import (
    RealtimeAudioCallback,
    RealtimeErrorCallback,
    RealtimeResponseEndCallback,
    RealtimeResponseStartCallback,
    RealtimeSpeechEndCallback,
    RealtimeSpeechStartCallback,
    RealtimeToolCallCallback,
    RealtimeTranscriptionCallback,
    RealtimeVoiceProvider,
)

logger = logging.getLogger("roomkit.providers.xai.realtime")

# Voices available on the xAI Realtime API.
XAI_VOICES = ("eve", "ara", "rex", "sal", "leo")


class XAIRealtimeProvider(RealtimeVoiceProvider):
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

        # Active WebSocket connections: session_id -> ws
        self._connections: dict[str, Any] = {}
        self._receive_tasks: dict[str, asyncio.Task[None]] = {}
        self._sessions: dict[str, VoiceSession] = {}

        # Callbacks
        self._audio_callbacks: list[RealtimeAudioCallback] = []
        self._transcription_callbacks: list[RealtimeTranscriptionCallback] = []
        self._speech_start_callbacks: list[RealtimeSpeechStartCallback] = []
        self._speech_end_callbacks: list[RealtimeSpeechEndCallback] = []
        self._tool_call_callbacks: list[RealtimeToolCallCallback] = []
        self._response_start_callbacks: list[RealtimeResponseStartCallback] = []
        self._response_end_callbacks: list[RealtimeResponseEndCallback] = []
        self._error_callbacks: list[RealtimeErrorCallback] = []

        # Track active responses per session to avoid inject_text conflicts
        self._responding: set[str] = set()

    def is_responding(self, session_id: str) -> bool:
        return session_id in self._responding

    @property
    def name(self) -> str:
        return "XAIRealtimeProvider"

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(
        self,
        session: VoiceSession,
        *,
        system_prompt: str | None = None,
        voice: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        input_sample_rate: int = 16000,
        output_sample_rate: int = 24000,
        server_vad: bool = True,
        provider_config: dict[str, Any] | None = None,
    ) -> None:
        try:
            import websockets
        except ImportError as exc:
            raise ImportError(
                "websockets is required for XAIRealtimeProvider. "
                "Install with: pip install websockets"
            ) from exc

        url = self._config.base_url
        headers = {
            "Authorization": f"Bearer {self._config.api_key.get_secret_value()}",
        }

        ws = await asyncio.wait_for(
            websockets.connect(url, additional_headers=headers),
            timeout=30.0,
        )

        self._connections[session.id] = ws
        self._sessions[session.id] = session

        pc = provider_config or {}

        # Build xAI session config — flat structure (not nested like OpenAI GA).
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
        # (web_search, x_search).
        if tools:
            session_config["tools"] = [{**t, "type": t.get("type", "function")} for t in tools]

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

        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": session_config,
                }
            )
        )

        session.state = VoiceSessionState.ACTIVE
        session.provider_session_id = session.id

        # Start receive loop
        self._receive_tasks[session.id] = asyncio.create_task(
            self._receive_loop(session),
            name=f"xai_rt_recv:{session.id}",
        )

        logger.info("xAI Realtime session connected: %s", session.id)

    async def send_audio(self, session: VoiceSession, audio: bytes) -> None:
        ws = self._connections.get(session.id)
        if ws is None:
            return
        await ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(audio).decode("ascii"),
                }
            )
        )

    async def inject_text(
        self,
        session: VoiceSession,
        text: str,
        *,
        role: str = "user",
        silent: bool = False,
    ) -> None:
        ws = self._connections.get(session.id)
        if ws is None:
            return

        logger.debug(
            "[xAI →] conversation.item.create (input_text, role=%s, silent=%s)",
            role,
            silent,
        )
        await ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": role if role in ("user", "system") else "user",
                        "content": [{"type": "input_text", "text": text}],
                    },
                }
            )
        )

        if silent:
            logger.debug("[xAI] Silent inject — no response.create")
            return

        if session.id in self._responding:
            logger.debug(
                "[xAI] Skipping response.create — response already active (session %s)",
                session.id,
            )
            return

        logger.debug("[xAI →] response.create")
        await ws.send(json.dumps({"type": "response.create"}))

    async def submit_tool_result(self, session: VoiceSession, call_id: str, result: str) -> None:
        ws = self._connections.get(session.id)
        if ws is None:
            return

        await ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": result,
                    },
                }
            )
        )

        logger.debug("[xAI →] response.create (after tool result)")
        await ws.send(json.dumps({"type": "response.create"}))

    async def interrupt(self, session: VoiceSession) -> None:
        ws = self._connections.get(session.id)
        if ws is None:
            return
        logger.debug("[xAI →] response.cancel")
        await ws.send(json.dumps({"type": "response.cancel"}))

    async def send_event(self, session: VoiceSession, event: dict[str, Any]) -> None:
        ws = self._connections.get(session.id)
        if ws is None:
            return
        await ws.send(json.dumps(event))

    async def disconnect(self, session: VoiceSession) -> None:
        import contextlib

        task = self._receive_tasks.pop(session.id, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

        ws = self._connections.pop(session.id, None)
        self._sessions.pop(session.id, None)
        self._responding.discard(session.id)
        if ws is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(ws.close(), timeout=2.0)

        session.state = VoiceSessionState.ENDED

    async def close(self) -> None:
        for session_id in list(self._sessions.keys()):
            session = self._sessions.get(session_id)
            if session:
                await self.disconnect(session)

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def on_audio(self, callback: RealtimeAudioCallback) -> None:
        self._audio_callbacks.append(callback)

    def on_transcription(self, callback: RealtimeTranscriptionCallback) -> None:
        self._transcription_callbacks.append(callback)

    def on_speech_start(self, callback: RealtimeSpeechStartCallback) -> None:
        self._speech_start_callbacks.append(callback)

    def on_speech_end(self, callback: RealtimeSpeechEndCallback) -> None:
        self._speech_end_callbacks.append(callback)

    def on_tool_call(self, callback: RealtimeToolCallCallback) -> None:
        self._tool_call_callbacks.append(callback)

    def on_response_start(self, callback: RealtimeResponseStartCallback) -> None:
        self._response_start_callbacks.append(callback)

    def on_response_end(self, callback: RealtimeResponseEndCallback) -> None:
        self._response_end_callbacks.append(callback)

    def on_error(self, callback: RealtimeErrorCallback) -> None:
        self._error_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Receive loop
    # ------------------------------------------------------------------

    async def _receive_loop(self, session: VoiceSession) -> None:
        """Process server events from xAI Realtime API."""
        ws = self._connections.get(session.id)
        if ws is None:
            return

        try:
            async for raw_message in ws:
                try:
                    event = json.loads(raw_message)
                    await self._handle_server_event(session, event)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from xAI for session %s", session.id)
                except Exception:
                    logger.exception("Error handling xAI event for session %s", session.id)
        except asyncio.CancelledError:
            raise
        except Exception:
            if session.state == VoiceSessionState.ACTIVE:
                logger.warning(
                    "xAI WebSocket closed unexpectedly for session %s",
                    session.id,
                )
                session.state = VoiceSessionState.ENDED
                await self._fire_error_callbacks(
                    session,
                    "connection_closed",
                    f"WebSocket closed unexpectedly for session {session.id}",
                )
            else:
                logger.debug("xAI WebSocket closed for session %s", session.id)

    # Event types that carry bulk audio data — skip in protocol log.
    _NOISY_EVENTS = frozenset(
        {
            "response.output_audio.delta",
            "response.output_audio_transcript.delta",
        }
    )

    async def _handle_server_event(self, session: VoiceSession, event: dict[str, Any]) -> None:
        """Map xAI server events to callbacks."""
        event_type = event.get("type", "")

        if event_type not in self._NOISY_EVENTS:
            logger.debug(
                "[xAI ←] %s %s",
                event_type,
                {k: v for k, v in event.items() if k not in ("type", "delta", "audio")},
            )

        if event_type == "input_audio_buffer.speech_started":
            logger.info("[VAD] speech_start (session %s)", session.id)
            await self._fire_callbacks(self._speech_start_callbacks, session)

        elif event_type == "input_audio_buffer.speech_stopped":
            logger.info("[VAD] speech_end (session %s)", session.id)
            await self._fire_callbacks(self._speech_end_callbacks, session)

        elif event_type == "response.output_audio.delta":
            audio_b64 = event.get("delta", "")
            if audio_b64:
                audio_bytes = base64.b64decode(audio_b64)
                await self._fire_audio_callbacks(session, audio_bytes)

        elif event_type == "response.output_audio_transcript.delta":
            text = event.get("delta", "")
            if text:
                await self._fire_transcription_callbacks(session, text, "assistant", False)

        elif event_type == ("conversation.item.input_audio_transcription.completed"):
            text = event.get("transcript", "")
            if text:
                await self._fire_transcription_callbacks(session, text, "user", True)

        elif event_type == "response.output_audio_transcript.done":
            text = event.get("transcript", "")
            if text:
                await self._fire_transcription_callbacks(session, text, "assistant", True)

        elif event_type == "response.function_call_arguments.done":
            call_id = event.get("call_id", "")
            name = event.get("name", "")
            args_str = event.get("arguments", "{}")
            try:
                arguments = json.loads(args_str)
            except json.JSONDecodeError:
                arguments = {"raw": args_str}
            await self._fire_tool_call_callbacks(session, call_id, name, arguments)

        elif event_type == "response.created":
            self._responding.add(session.id)
            logger.info("[xAI] response_start (session %s)", session.id)
            await self._fire_callbacks(self._response_start_callbacks, session)

        elif event_type == "response.done":
            response = event.get("response", {})
            status = response.get("status", "")
            if status == "failed":
                details = response.get("status_details", {})
                err = details.get("error", {})
                err_type = err.get("type", "unknown")
                err_code = err.get("code", "")
                err_message = err.get("message", "Unknown error")
                logger.error(
                    "[xAI] response FAILED: type=%s code=%s message=%s (session %s)",
                    err_type,
                    err_code,
                    err_message,
                    session.id,
                )
                await self._fire_error_callbacks(session, err_code or err_type, err_message)
            else:
                logger.info(
                    "[xAI] response_done status=%s (session %s)",
                    status,
                    session.id,
                )

            # Extract token usage
            usage = response.get("usage", {})
            if usage:
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                input_details = usage.get("input_token_details", {})
                output_details = usage.get("output_token_details", {})
                logger.info(
                    "[xAI] usage: input=%d output=%d (session %s)",
                    input_tokens,
                    output_tokens,
                    session.id,
                )
                self._record_usage(
                    session,
                    input_tokens,
                    output_tokens,
                    details={
                        "input_token_details": input_details,
                        "output_token_details": output_details,
                    },
                )

            self._responding.discard(session.id)
            await self._fire_callbacks(self._response_end_callbacks, session)

        elif event_type == "session.created":
            sid = event.get("session", {}).get("id", "")
            logger.info("[xAI] session.created: id=%s (session %s)", sid, session.id)

        elif event_type == "session.updated":
            logger.info("[xAI] session.updated (session %s)", session.id)

        elif event_type == "input_audio_buffer.committed":
            logger.debug("[xAI] audio_buffer committed (session %s)", session.id)

        elif event_type == "error":
            error = event.get("error", {})
            code = error.get("code", "unknown")
            message = error.get("message", "Unknown error")
            logger.error("[xAI] error [%s] %s (session %s)", code, message, session.id)
            await self._fire_error_callbacks(session, code, message)

    # ------------------------------------------------------------------
    # Callback helpers
    # ------------------------------------------------------------------

    async def _fire_callbacks(
        self,
        callbacks: list[Any],
        session: VoiceSession,
    ) -> None:
        for cb in callbacks:
            try:
                result = cb(session)
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                logger.exception("Error in callback for session %s", session.id)

    async def _fire_audio_callbacks(self, session: VoiceSession, audio: bytes) -> None:
        for cb in self._audio_callbacks:
            try:
                result = cb(session, audio)
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                logger.exception("Error in audio callback for session %s", session.id)

    async def _fire_transcription_callbacks(
        self,
        session: VoiceSession,
        text: str,
        role: str,
        is_final: bool,
    ) -> None:
        for cb in self._transcription_callbacks:
            try:
                result = cb(session, text, role, is_final)
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                logger.exception(
                    "Error in transcription callback for session %s",
                    session.id,
                )

    async def _fire_tool_call_callbacks(
        self,
        session: VoiceSession,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> None:
        for cb in self._tool_call_callbacks:
            try:
                result = cb(session, call_id, name, arguments)
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                logger.exception("Error in tool call callback for session %s", session.id)

    async def _fire_error_callbacks(self, session: VoiceSession, code: str, message: str) -> None:
        for cb in self._error_callbacks:
            try:
                result = cb(session, code, message)
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                logger.exception("Error in error callback for session %s", session.id)
