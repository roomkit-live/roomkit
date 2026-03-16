"""OpenAI Realtime API provider for speech-to-speech conversations."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any

from pydantic import SecretStr

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

    Connects via WebSocket to OpenAI's Realtime API (GA),
    handling bidirectional audio streaming with built-in VAD,
    transcription, and AI responses.

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
        self._api_key = SecretStr(api_key) if isinstance(api_key, str) else api_key
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

        # Track active responses per session to avoid inject_text conflicts
        self._responding: set[str] = set()

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

        if temperature is not None:
            logger.warning(
                "OpenAI Realtime GA API no longer supports the temperature parameter; ignoring"
            )

        url = f"{self._base_url}?model={self._model}"
        headers = {
            "Authorization": f"Bearer {self._api_key.get_secret_value()}",
        }

        ws = await asyncio.wait_for(
            websockets.connect(url, additional_headers=headers),
            timeout=30.0,
        )

        self._connections[session.id] = ws
        self._sessions[session.id] = session

        pc = provider_config or {}

        # Build GA session config — audio settings nest under audio.input / audio.output
        # noise_reduction: "near_field" for headphones/close mic,
        # "far_field" for laptop/conference room speakers.
        nr_type = pc.get("noise_reduction", "far_field")
        audio_input: dict[str, Any] = {
            "format": {"type": "audio/pcm", "rate": 24000},
            "transcription": {"model": pc.get("stt_model", "gpt-4o-transcribe")},
            "noise_reduction": {"type": nr_type},
        }
        audio_output: dict[str, Any] = {"format": {"type": "audio/pcm", "rate": 24000}}
        if voice:
            audio_output["voice"] = voice

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
            # OpenAI Realtime requires "type": "function" on each tool.
            # Normalize tools that omit it (e.g. from DescribeScreenTool).
            session_config["tools"] = [{**t, "type": t.get("type", "function")} for t in tools]

        logger.info(
            "Sending session.update: turn_detection=%s, voice=%s",
            turn_detection,
            voice,
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
            "[OpenAI →] conversation.item.create (input_text, role=%s, silent=%s)",
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

        # Silent: add to context without requesting a response.
        # The agent sees it on the next user turn.
        if silent:
            logger.debug("[OpenAI] Silent inject — no response.create")
            return

        # Only request a new response if none is in progress.
        if session.id in self._responding:
            logger.debug(
                "[OpenAI] Skipping response.create — response already active (session %s)",
                session.id,
            )
            return

        logger.debug("[OpenAI →] response.create")
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

        logger.debug("[OpenAI →] response.create (after tool result)")
        await ws.send(json.dumps({"type": "response.create"}))

    async def interrupt(self, session: VoiceSession) -> None:
        ws = self._connections.get(session.id)
        if ws is None:
            return
        logger.debug("[OpenAI →] response.cancel")
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

        # Close WebSocket (short timeout to avoid blocking on close handshake)
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

    # -- Turn detection builder --

    @staticmethod
    def _build_turn_detection(td_type: str | None, pc: dict[str, Any]) -> dict[str, Any] | None:
        """Build the turn_detection dict for the GA session config."""
        if td_type == "semantic_vad":
            td: dict[str, Any] = {"type": "semantic_vad"}
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
            if pc.get("interrupt_response") is not None:
                td["interrupt_response"] = bool(pc["interrupt_response"])
            if pc.get("create_response") is not None:
                td["create_response"] = bool(pc["create_response"])
            return td
        return None

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

    # Server event types that carry bulk audio data — skip in protocol log.
    _NOISY_EVENTS = frozenset(
        {
            "response.output_audio.delta",
            "response.output_audio_transcript.delta",
        }
    )

    async def _handle_server_event(self, session: VoiceSession, event: dict[str, Any]) -> None:
        """Map OpenAI server events to callbacks."""
        event_type = event.get("type", "")

        # Protocol-level log: show every event except high-frequency audio deltas.
        if event_type not in self._NOISY_EVENTS:
            logger.debug(
                "[OpenAI ←] %s %s",
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

        elif event_type == "conversation.item.input_audio_transcription.completed":
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

            # Extract token usage from response.done and record via telemetry
            usage = response.get("usage", {})
            if usage:
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                input_token_details = usage.get("input_token_details", {})
                output_token_details = usage.get("output_token_details", {})
                logger.info(
                    "[OpenAI] usage: input=%d output=%d "
                    "(cached_input=%d, text_input=%d, audio_input=%d, "
                    "text_output=%d, audio_output=%d) (session %s)",
                    input_tokens,
                    output_tokens,
                    input_token_details.get("cached_tokens", 0),
                    input_token_details.get("text_tokens", 0),
                    input_token_details.get("audio_tokens", 0),
                    output_token_details.get("text_tokens", 0),
                    output_token_details.get("audio_tokens", 0),
                    session.id,
                )
                # Store on session for telemetry span attribution
                if not hasattr(session, "_last_usage"):
                    object.__setattr__(session, "_last_usage", {})
                object.__setattr__(session, "_last_usage", {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "input_token_details": input_token_details,
                    "output_token_details": output_token_details,
                })
                # Record via telemetry if available
                telemetry = getattr(self, "_telemetry", None)
                if telemetry is not None:
                    telemetry.record_metric(
                        "roomkit.realtime.input_tokens",
                        float(input_tokens),
                        unit="tokens",
                        attributes={
                            "session_id": session.id,
                            "model": self._model,
                        },
                    )
                    telemetry.record_metric(
                        "roomkit.realtime.output_tokens",
                        float(output_tokens),
                        unit="tokens",
                        attributes={
                            "session_id": session.id,
                            "model": self._model,
                        },
                    )

            self._responding.discard(session.id)
            await self._fire_callbacks(self._response_end_callbacks, session)

        elif event_type == "session.created":
            td_type = (
                event.get("session", {})
                .get("audio", {})
                .get("input", {})
                .get("turn_detection", {})
                .get("type")
            )
            logger.info(
                "[OpenAI] session.created: turn_detection=%s (session %s)",
                td_type,
                session.id,
            )

        elif event_type == "session.updated":
            td_type = (
                event.get("session", {})
                .get("audio", {})
                .get("input", {})
                .get("turn_detection", {})
                .get("type")
            )
            logger.info(
                "[OpenAI] session.updated: turn_detection=%s (session %s)",
                td_type,
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
