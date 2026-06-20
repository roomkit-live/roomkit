"""Shared base for OpenAI-Realtime-wire-compatible providers.

The OpenAI Realtime WebSocket protocol is also spoken by xAI Grok. This base
class owns everything that is identical between them — the connection
lifecycle, audio/text/tool plumbing, the receive loop, and the server-event
dispatch table — leaving subclasses to supply only what genuinely differs:
the session-config shape, auth/URL, and a few provider-specific log lines.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from abc import abstractmethod
from typing import Any

from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.provider import RealtimeVoiceProvider

logger = logging.getLogger("roomkit.providers.openai.realtime_base")

_CONNECT_TIMEOUT = 30.0
_CLOSE_TIMEOUT = 2.0

# Server event types on the OpenAI Realtime wire (shared by xAI Grok).
_EVT_SPEECH_STARTED = "input_audio_buffer.speech_started"
_EVT_SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
_EVT_AUDIO_DELTA = "response.output_audio.delta"
_EVT_TRANSCRIPT_DELTA = "response.output_audio_transcript.delta"
_EVT_INPUT_TRANSCRIPT_DONE = "conversation.item.input_audio_transcription.completed"
_EVT_TRANSCRIPT_DONE = "response.output_audio_transcript.done"
_EVT_FUNCTION_CALL_DONE = "response.function_call_arguments.done"
_EVT_RESPONSE_CREATED = "response.created"
_EVT_RESPONSE_DONE = "response.done"
_EVT_SESSION_CREATED = "session.created"
_EVT_SESSION_UPDATED = "session.updated"
_EVT_BUFFER_COMMITTED = "input_audio_buffer.committed"
_EVT_ERROR = "error"

# Event types that carry bulk audio data — skipped in the protocol-level log.
_NOISY_EVENTS = frozenset({_EVT_AUDIO_DELTA, _EVT_TRANSCRIPT_DELTA})


class OpenAIRealtimeBase(RealtimeVoiceProvider):
    """Connection + event plumbing shared by OpenAI and xAI realtime providers.

    Subclasses must implement: :attr:`name`, :meth:`available_voices`,
    :attr:`_log_tag`, :attr:`_recv_task_prefix`, :attr:`_websockets_install_hint`,
    :meth:`_connect_url`, :meth:`_auth_headers`, and :meth:`_build_session_config`.
    They may override :meth:`_log_usage`, :meth:`_on_session_created`, and
    :meth:`_on_session_updated` for provider-specific logging.
    """

    def __init__(self) -> None:
        super().__init__()
        # Active WebSocket connections: session_id -> ws
        self._connections: dict[str, Any] = {}
        self._receive_tasks: dict[str, asyncio.Task[None]] = {}
        self._sessions: dict[str, VoiceSession] = {}
        # Track active responses per session to avoid inject_text conflicts
        self._responding: set[str] = set()

    def is_responding(self, session_id: str) -> bool:
        return session_id in self._responding

    # -- Provider-specific extension points ---------------------------------

    @property
    @abstractmethod
    def _log_tag(self) -> str:
        """Short provider tag used in log lines (e.g. ``"OpenAI"``)."""
        ...

    @property
    @abstractmethod
    def _recv_task_prefix(self) -> str:
        """Prefix for the receive-loop task name (e.g. ``"openai_rt_recv"``)."""
        ...

    @property
    @abstractmethod
    def _websockets_install_hint(self) -> str:
        """Install command shown when the ``websockets`` dependency is missing."""
        ...

    @abstractmethod
    def _connect_url(self) -> str:
        """Full WebSocket URL to connect to."""
        ...

    @abstractmethod
    def _auth_headers(self) -> dict[str, str]:
        """Authorization headers for the WebSocket handshake."""
        ...

    @abstractmethod
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
        """Build the provider-specific ``session.update`` config payload.

        Implementations also perform any pre-connect validation (so it fails
        before a socket is opened) and emit the provider's "Sending
        session.update" info log.
        """
        ...

    # -- Connection lifecycle -----------------------------------------------

    def _import_websockets(self) -> Any:
        try:
            import websockets
        except ImportError as exc:
            raise ImportError(
                f"websockets is required for {self.name}. "
                f"Install with: {self._websockets_install_hint}"
            ) from exc
        return websockets

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
        websockets = self._import_websockets()
        pc = provider_config or {}

        # Built before opening the socket so validation errors fail fast.
        session_config = self._build_session_config(
            system_prompt=system_prompt,
            voice=voice,
            tools=tools,
            temperature=temperature,
            input_sample_rate=input_sample_rate,
            output_sample_rate=output_sample_rate,
            server_vad=server_vad,
            pc=pc,
        )

        ws = await asyncio.wait_for(
            websockets.connect(self._connect_url(), additional_headers=self._auth_headers()),
            timeout=_CONNECT_TIMEOUT,
        )

        self._connections[session.id] = ws
        self._sessions[session.id] = session

        await ws.send(json.dumps({"type": "session.update", "session": session_config}))

        session.state = VoiceSessionState.ACTIVE
        session.provider_session_id = session.id

        self._receive_tasks[session.id] = asyncio.create_task(
            self._receive_loop(session),
            name=f"{self._recv_task_prefix}:{session.id}",
        )

        logger.info("%s Realtime session connected: %s", self._log_tag, session.id)

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
            "[%s →] conversation.item.create (input_text, role=%s, silent=%s)",
            self._log_tag,
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
        if silent:
            logger.debug("[%s] Silent inject — no response.create", self._log_tag)
            return

        # Only request a new response if none is in progress.
        if session.id in self._responding:
            logger.debug(
                "[%s] Skipping response.create — response already active (session %s)",
                self._log_tag,
                session.id,
            )
            return

        logger.debug("[%s →] response.create", self._log_tag)
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

        logger.debug("[%s →] response.create (after tool result)", self._log_tag)
        await ws.send(json.dumps({"type": "response.create"}))

    async def interrupt(self, session: VoiceSession) -> None:
        ws = self._connections.get(session.id)
        if ws is None:
            return
        logger.debug("[%s →] response.cancel", self._log_tag)
        await ws.send(json.dumps({"type": "response.cancel"}))

    async def send_event(self, session: VoiceSession, event: dict[str, Any]) -> None:
        ws = self._connections.get(session.id)
        if ws is None:
            return
        await ws.send(json.dumps(event))

    async def send_activity_start(self, session: VoiceSession) -> None:
        """No-op — audio flows continuously via input_audio_buffer.append."""
        logger.debug("[%s] activity_start (no-op, session %s)", self._log_tag, session.id)

    async def send_activity_end(self, session: VoiceSession) -> None:
        """Commit audio buffer and request a response (manual VAD mode)."""
        ws = self._connections.get(session.id)
        if ws is None:
            return
        logger.debug("[%s →] input_audio_buffer.commit (session %s)", self._log_tag, session.id)
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        if session.id in self._responding:
            logger.debug(
                "[%s] skip response.create — responding (session %s)", self._log_tag, session.id
            )
            return
        logger.debug("[%s →] response.create (session %s)", self._log_tag, session.id)
        await ws.send(json.dumps({"type": "response.create"}))

    async def disconnect(self, session: VoiceSession) -> None:
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
                await asyncio.wait_for(ws.close(), timeout=_CLOSE_TIMEOUT)

        session.state = VoiceSessionState.ENDED

    async def close(self) -> None:
        for session_id in list(self._sessions.keys()):
            session = self._sessions.get(session_id)
            if session:
                await self.disconnect(session)

    # -- Receive loop + event dispatch --------------------------------------

    async def _receive_loop(self, session: VoiceSession) -> None:
        """Process server events from the realtime API."""
        ws = self._connections.get(session.id)
        if ws is None:
            return

        try:
            async for raw_message in ws:
                try:
                    event = json.loads(raw_message)
                    await self._handle_server_event(session, event)
                except json.JSONDecodeError:
                    logger.warning(
                        "Invalid JSON from %s for session %s", self._log_tag, session.id
                    )
                except Exception:
                    logger.exception(
                        "Error handling %s event for session %s", self._log_tag, session.id
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            if session.state == VoiceSessionState.ACTIVE:
                logger.warning(
                    "%s WebSocket closed unexpectedly for session %s", self._log_tag, session.id
                )
                session.state = VoiceSessionState.ENDED
                await self._fire(
                    self._error_callbacks,
                    session,
                    "connection_closed",
                    f"WebSocket closed unexpectedly for session {session.id}",
                    label="error",
                )
            else:
                logger.debug("%s WebSocket closed for session %s", self._log_tag, session.id)

    # Maps each server event type to the handler method that processes it.
    _EVENT_HANDLERS: dict[str, str] = {
        _EVT_SPEECH_STARTED: "_on_speech_started",
        _EVT_SPEECH_STOPPED: "_on_speech_stopped",
        _EVT_AUDIO_DELTA: "_on_audio_delta",
        _EVT_TRANSCRIPT_DELTA: "_on_transcript_delta",
        _EVT_INPUT_TRANSCRIPT_DONE: "_on_input_transcript_done",
        _EVT_TRANSCRIPT_DONE: "_on_transcript_done",
        _EVT_FUNCTION_CALL_DONE: "_on_function_call_done",
        _EVT_RESPONSE_CREATED: "_on_response_created",
        _EVT_RESPONSE_DONE: "_on_response_done",
        _EVT_SESSION_CREATED: "_on_session_created",
        _EVT_SESSION_UPDATED: "_on_session_updated",
        _EVT_BUFFER_COMMITTED: "_on_buffer_committed",
        _EVT_ERROR: "_on_error",
    }

    async def _handle_server_event(self, session: VoiceSession, event: dict[str, Any]) -> None:
        """Route a server event to its handler via the dispatch table."""
        event_type = event.get("type", "")

        # Protocol-level log: show every event except high-frequency audio deltas.
        if event_type not in _NOISY_EVENTS:
            logger.debug(
                "[%s ←] %s %s",
                self._log_tag,
                event_type,
                {k: v for k, v in event.items() if k not in ("type", "delta", "audio")},
            )

        handler_name = self._EVENT_HANDLERS.get(event_type)
        if handler_name is not None:
            await getattr(self, handler_name)(session, event)

    async def _on_speech_started(self, session: VoiceSession, event: dict[str, Any]) -> None:
        logger.info("[VAD] speech_start (session %s)", session.id)
        await self._fire(self._speech_start_callbacks, session, label="speech_start")

    async def _on_speech_stopped(self, session: VoiceSession, event: dict[str, Any]) -> None:
        logger.info("[VAD] speech_end (session %s)", session.id)
        await self._fire(self._speech_end_callbacks, session, label="speech_end")

    async def _on_audio_delta(self, session: VoiceSession, event: dict[str, Any]) -> None:
        audio_b64 = event.get("delta", "")
        if audio_b64:
            audio_bytes = base64.b64decode(audio_b64)
            await self._fire(self._audio_callbacks, session, audio_bytes, label="audio")

    async def _on_transcript_delta(self, session: VoiceSession, event: dict[str, Any]) -> None:
        text = event.get("delta", "")
        if text:
            await self._fire(
                self._transcription_callbacks,
                session,
                text,
                "assistant",
                False,
                label="transcription",
            )

    async def _on_input_transcript_done(
        self, session: VoiceSession, event: dict[str, Any]
    ) -> None:
        text = event.get("transcript", "")
        if text:
            await self._fire(
                self._transcription_callbacks,
                session,
                text,
                "user",
                True,
                label="transcription",
            )

    async def _on_transcript_done(self, session: VoiceSession, event: dict[str, Any]) -> None:
        text = event.get("transcript", "")
        if text:
            await self._fire(
                self._transcription_callbacks,
                session,
                text,
                "assistant",
                True,
                label="transcription",
            )

    async def _on_function_call_done(self, session: VoiceSession, event: dict[str, Any]) -> None:
        call_id = event.get("call_id", "")
        name = event.get("name", "")
        args_str = event.get("arguments", "{}")
        try:
            arguments = json.loads(args_str)
        except json.JSONDecodeError:
            arguments = {"raw": args_str}
        await self._fire(
            self._tool_call_callbacks,
            session,
            call_id,
            name,
            arguments,
            label="tool_call",
        )

    async def _on_response_created(self, session: VoiceSession, event: dict[str, Any]) -> None:
        self._responding.add(session.id)
        logger.info("[%s] response_start (session %s)", self._log_tag, session.id)
        await self._fire(self._response_start_callbacks, session, label="response_start")

    async def _on_response_done(self, session: VoiceSession, event: dict[str, Any]) -> None:
        response = event.get("response", {})
        status = response.get("status", "")
        if status == "failed":
            details = response.get("status_details", {})
            err = details.get("error", {})
            err_type = err.get("type", "unknown")
            err_code = err.get("code", "")
            err_message = err.get("message", "Unknown error")
            logger.error(
                "[%s] response FAILED: type=%s code=%s message=%s (session %s)",
                self._log_tag,
                err_type,
                err_code,
                err_message,
                session.id,
            )
            await self._fire(
                self._error_callbacks,
                session,
                err_code or err_type,
                err_message,
                label="error",
            )
        else:
            logger.info(
                "[%s] response_done status=%s (session %s)", self._log_tag, status, session.id
            )

        usage = response.get("usage", {})
        if usage:
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            input_details = usage.get("input_token_details", {})
            output_details = usage.get("output_token_details", {})
            self._log_usage(session, input_tokens, output_tokens, input_details, output_details)
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
        await self._fire(self._response_end_callbacks, session, label="response_end")

    async def _on_buffer_committed(self, session: VoiceSession, event: dict[str, Any]) -> None:
        logger.debug("[%s] audio_buffer committed (session %s)", self._log_tag, session.id)

    async def _on_error(self, session: VoiceSession, event: dict[str, Any]) -> None:
        error = event.get("error", {})
        code = error.get("code", "unknown")
        message = error.get("message", "Unknown error")
        logger.error("[%s] error [%s] %s (session %s)", self._log_tag, code, message, session.id)
        await self._fire(self._error_callbacks, session, code, message, label="error")

    # -- Overridable logging hooks ------------------------------------------

    def _log_usage(
        self,
        session: VoiceSession,
        input_tokens: int,
        output_tokens: int,
        input_details: dict[str, Any],
        output_details: dict[str, Any],
    ) -> None:
        """Log token usage. Subclasses may override for a richer breakdown."""
        logger.info(
            "[%s] usage: input=%d output=%d (session %s)",
            self._log_tag,
            input_tokens,
            output_tokens,
            session.id,
        )

    async def _on_session_created(self, session: VoiceSession, event: dict[str, Any]) -> None:
        logger.info("[%s] session.created (session %s)", self._log_tag, session.id)

    async def _on_session_updated(self, session: VoiceSession, event: dict[str, Any]) -> None:
        logger.info("[%s] session.updated (session %s)", self._log_tag, session.id)
