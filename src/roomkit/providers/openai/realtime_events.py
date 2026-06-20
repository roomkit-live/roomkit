"""Inbound server-event handling for OpenAI-Realtime-wire providers.

Translates the OpenAI Realtime server events (also spoken by xAI Grok) into
RoomKit provider callbacks: the receive loop, a dispatch table keyed on the
wire event type, and one handler per event. Kept separate from the outbound
client API (``OpenAIRealtimeBase``) so each side stays one responsibility.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from abc import abstractmethod
from typing import Any

from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.provider import RealtimeVoiceProvider

logger = logging.getLogger("roomkit.providers.openai.realtime_events")

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


class OpenAIRealtimeEventHandlersMixin(RealtimeVoiceProvider):
    """Receive loop + server-event → callback dispatch for the OpenAI wire.

    Mixed into ``OpenAIRealtimeBase``, which supplies the connection state and
    the ``_log_tag``. Subclasses may override :meth:`_log_usage`,
    :meth:`_on_session_created`, and :meth:`_on_session_updated`.
    """

    # Connection state owned by OpenAIRealtimeBase.__init__; declared for typing.
    _connections: dict[str, Any]
    _responding: set[str]

    @property
    @abstractmethod
    def _log_tag(self) -> str:
        """Short provider tag used in log lines (e.g. ``"OpenAI"``)."""
        ...

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
