"""OpenAI Realtime API provider for speech-to-speech conversations."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any

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

logger = logging.getLogger("roomkit.providers.openai.realtime")

# Default OpenAI Realtime API endpoint
_DEFAULT_BASE_URL = "wss://api.openai.com/v1/realtime"


class OpenAIRealtimeProvider(RealtimeVoiceProvider):
    """Realtime voice provider using the OpenAI Realtime API.

    Connects via WebSocket to OpenAI's speech-to-speech API,
    handling bidirectional audio streaming with built-in VAD,
    transcription, and AI responses.

    Requires the ``websockets`` package.

    Example:
        provider = OpenAIRealtimeProvider(api_key="sk-...", model="gpt-4o-realtime-preview")
        provider.on_audio(handle_output_audio)
        provider.on_transcription(handle_transcription)

        await provider.connect(session, system_prompt="You are a helpful assistant.")
        await provider.send_audio(session, audio_bytes)
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-4o-realtime-preview",
        base_url: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url or _DEFAULT_BASE_URL

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

    @property
    def name(self) -> str:
        return "OpenAIRealtimeProvider"

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
                "websockets is required for OpenAIRealtimeProvider. "
                "Install with: pip install 'roomkit[realtime-openai]'"
            ) from exc

        url = f"{self._base_url}?model={self._model}"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        ws = await websockets.connect(url, additional_headers=headers)

        self._connections[session.id] = ws
        self._sessions[session.id] = session

        pc = provider_config or {}

        # Configure session
        session_config: dict[str, Any] = {
            "modalities": ["text", "audio"],
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {"model": pc.get("stt_model", "gpt-4o-transcribe")},
        }

        if voice:
            session_config["voice"] = voice
        if temperature is not None:
            session_config["temperature"] = temperature
        if system_prompt:
            session_config["instructions"] = system_prompt
        if tools:
            session_config["tools"] = tools

        # --- Turn detection / VAD ---
        td_type = pc.get("turn_detection_type", "server_vad" if server_vad else None)
        if td_type == "semantic_vad":
            td: dict[str, Any] = {"type": "semantic_vad"}
            eagerness = pc.get("eagerness")
            if eagerness:
                td["eagerness"] = eagerness
            if pc.get("interrupt_response") is not None:
                td["interrupt_response"] = bool(pc["interrupt_response"])
            if pc.get("create_response") is not None:
                td["create_response"] = bool(pc["create_response"])
            session_config["turn_detection"] = td
        elif td_type == "server_vad":
            td = {"type": "server_vad"}
            if pc.get("threshold") is not None:
                td["threshold"] = float(pc["threshold"])
            if pc.get("silence_duration_ms") is not None:
                td["silence_duration_ms"] = int(pc["silence_duration_ms"])
            if pc.get("prefix_padding_ms") is not None:
                td["prefix_padding_ms"] = int(pc["prefix_padding_ms"])
            if pc.get("interrupt_response") is not None:
                td["interrupt_response"] = bool(pc["interrupt_response"])
            if pc.get("create_response") is not None:
                td["create_response"] = bool(pc["create_response"])
            session_config["turn_detection"] = td
        else:
            session_config["turn_detection"] = None

        logger.info(
            "Sending session.update: turn_detection=%s, voice=%s",
            session_config.get("turn_detection"),
            session_config.get("voice"),
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
            name=f"openai_rt_recv:{session.id}",
        )

        logger.info("OpenAI Realtime session connected: %s", session.id)

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

    async def inject_text(self, session: VoiceSession, text: str, *, role: str = "user") -> None:
        ws = self._connections.get(session.id)
        if ws is None:
            return

        # Create a conversation item with the injected text
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

        # Trigger a response
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

        # Trigger a response after tool result
        await ws.send(json.dumps({"type": "response.create"}))

    async def interrupt(self, session: VoiceSession) -> None:
        ws = self._connections.get(session.id)
        if ws is None:
            return
        await ws.send(json.dumps({"type": "response.cancel"}))

    async def send_event(self, session: VoiceSession, event: dict[str, Any]) -> None:
        ws = self._connections.get(session.id)
        if ws is None:
            return
        await ws.send(json.dumps(event))

    async def disconnect(self, session: VoiceSession) -> None:
        import contextlib

        # Cancel receive task
        task = self._receive_tasks.pop(session.id, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

        # Close WebSocket
        ws = self._connections.pop(session.id, None)
        self._sessions.pop(session.id, None)
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()

        session.state = VoiceSessionState.ENDED

    async def close(self) -> None:
        for session_id in list(self._sessions.keys()):
            session = self._sessions.get(session_id)
            if session:
                await self.disconnect(session)

    # -- Callback registration --

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

    # -- Receive loop --

    async def _receive_loop(self, session: VoiceSession) -> None:
        """Process server events from OpenAI Realtime API."""
        ws = self._connections.get(session.id)
        if ws is None:
            return

        try:
            async for raw_message in ws:
                try:
                    event = json.loads(raw_message)
                    await self._handle_server_event(session, event)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from OpenAI for session %s", session.id)
                except Exception:
                    logger.exception("Error handling OpenAI event for session %s", session.id)
        except asyncio.CancelledError:
            raise
        except Exception:
            if session.state == VoiceSessionState.ACTIVE:
                logger.warning("OpenAI WebSocket closed unexpectedly for session %s", session.id)
                session.state = VoiceSessionState.ENDED
                await self._fire_error_callbacks(
                    session,
                    "connection_closed",
                    f"WebSocket closed unexpectedly for session {session.id}",
                )
            else:
                logger.debug("OpenAI WebSocket closed for session %s", session.id)

    async def _handle_server_event(self, session: VoiceSession, event: dict[str, Any]) -> None:
        """Map OpenAI server events to callbacks."""
        event_type = event.get("type", "")

        if event_type == "input_audio_buffer.speech_started":
            logger.info("[VAD] speech_start (session %s)", session.id)
            await self._fire_callbacks(self._speech_start_callbacks, session)

        elif event_type == "input_audio_buffer.speech_stopped":
            logger.info("[VAD] speech_end (session %s)", session.id)
            await self._fire_callbacks(self._speech_end_callbacks, session)

        elif event_type == "response.audio.delta":
            audio_b64 = event.get("delta", "")
            if audio_b64:
                audio_bytes = base64.b64decode(audio_b64)
                await self._fire_audio_callbacks(session, audio_bytes)

        elif event_type == "response.audio_transcript.delta":
            text = event.get("delta", "")
            if text:
                await self._fire_transcription_callbacks(session, text, "assistant", False)

        elif event_type == "conversation.item.input_audio_transcription.completed":
            text = event.get("transcript", "")
            if text:
                await self._fire_transcription_callbacks(session, text, "user", True)

        elif event_type == "response.audio_transcript.done":
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
            logger.info("[OpenAI] response_start (session %s)", session.id)
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
                    "[OpenAI] response FAILED: type=%s code=%s message=%s (session %s)",
                    err_type,
                    err_code,
                    err_message,
                    session.id,
                )
                await self._fire_error_callbacks(session, err_code or err_type, err_message)
            else:
                logger.info("[OpenAI] response_done status=%s (session %s)", status, session.id)
            await self._fire_callbacks(self._response_end_callbacks, session)

        elif event_type == "session.created":
            td = event.get("session", {}).get("turn_detection", {})
            logger.info(
                "[OpenAI] session.created: turn_detection=%s (session %s)",
                td,
                session.id,
            )

        elif event_type == "session.updated":
            td = event.get("session", {}).get("turn_detection", {})
            logger.info(
                "[OpenAI] session.updated: turn_detection=%s (session %s)",
                td,
                session.id,
            )

        elif event_type == "input_audio_buffer.committed":
            logger.debug("[OpenAI] audio_buffer committed (session %s)", session.id)

        elif event_type == "error":
            error = event.get("error", {})
            code = error.get("code", "unknown")
            message = error.get("message", "Unknown error")
            logger.error("[OpenAI] error [%s] %s (session %s)", code, message, session.id)
            await self._fire_error_callbacks(session, code, message)

    # -- Callback helpers --

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
        self, session: VoiceSession, text: str, role: str, is_final: bool
    ) -> None:
        for cb in self._transcription_callbacks:
            try:
                result = cb(session, text, role, is_final)
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                logger.exception("Error in transcription callback for session %s", session.id)

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
