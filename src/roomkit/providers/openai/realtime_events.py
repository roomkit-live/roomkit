"""Inbound server-event handling for OpenAI-Realtime-wire providers.

Translates the OpenAI Realtime server events (also spoken by xAI Grok) into
RoomKit provider callbacks: the receive loop, a dispatch table keyed on the
wire event type, and one handler per event. Kept separate from the outbound
client API (``OpenAIRealtimeBase``) so each side stays one responsibility.
The one outbound concern here is the response-request gate: every
``response.create`` goes through it, because the server events (a response's
start and end, the caller's speech, an error) are what move its state.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from abc import abstractmethod
from typing import Any

from roomkit.providers.ai.tool_calls import realtime_call_arguments
from roomkit.providers.openai.response_calls import PendingResponse
from roomkit.voice._g711 import _G711Codec
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.provider import RealtimeVoiceProvider

logger = logging.getLogger("roomkit.providers.openai.realtime_events")

# Server event types on the OpenAI Realtime wire (shared by xAI Grok).
_EVT_SPEECH_STARTED = "input_audio_buffer.speech_started"
_EVT_SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
_EVT_AUDIO_DELTA = "response.output_audio.delta"
_EVT_TRANSCRIPT_DELTA = "response.output_audio_transcript.delta"
_EVT_INPUT_TRANSCRIPT_DONE = "conversation.item.input_audio_transcription.completed"
_EVT_TRANSCRIPT_DONE = "response.output_audio_transcript.done"
_EVT_OUTPUT_ITEM_DONE = "response.output_item.done"
_EVT_RESPONSE_CREATED = "response.created"
_EVT_RESPONSE_DONE = "response.done"
_EVT_SESSION_CREATED = "session.created"
_EVT_SESSION_UPDATED = "session.updated"
_EVT_BUFFER_COMMITTED = "input_audio_buffer.committed"
_EVT_ERROR = "error"

# Event types that carry bulk audio data — skipped in the protocol-level log.
_NOISY_EVENTS = frozenset({_EVT_AUDIO_DELTA, _EVT_TRANSCRIPT_DELTA})


class _OutputAudioState:
    """Wire identity and generated duration of the latest audio item."""

    __slots__ = ("content_index", "item_id", "received_bytes", "truncated_at_ms")

    def __init__(self, item_id: str, content_index: int) -> None:
        self.item_id = item_id
        self.content_index = content_index
        self.received_bytes = 0
        self.truncated_at_ms: int | None = None


class OpenAIRealtimeEventHandlersMixin(RealtimeVoiceProvider):
    """Receive loop + server-event → callback dispatch for the OpenAI wire.

    It also owns the gate every ``response.create`` passes (RFC §12.4): the
    one a response's tool calls owe, the one a caller's turn is owed, and the
    requests of ``inject_text`` and the end of the caller's turn.
    Mixed into ``OpenAIRealtimeBase``, which supplies the connection state and
    the ``_log_tag``. Subclasses may override :meth:`_log_usage`,
    :meth:`_on_session_created`, and :meth:`_on_session_updated`.
    """

    # Connection state owned by OpenAIRealtimeBase.__init__; declared for typing.
    _connections: dict[str, Any]
    _responding: set[str]
    _pending_responses: dict[str, PendingResponse]
    _floor_held: set[str]
    _turns_owed: set[str]
    _provider_configs: dict[str, dict[str, Any]]
    _output_audio: dict[str, _OutputAudioState]
    _audio_codecs: dict[str, tuple[_G711Codec | None, _G711Codec | None]]

    @property
    @abstractmethod
    def _log_tag(self) -> str:
        """Short provider tag used in log lines (e.g. ``"OpenAI"``)."""
        ...

    @abstractmethod
    async def _retire_lost_connection(self, session: VoiceSession, ws: Any, message: str) -> None:
        """Release a socket whose receive loop stopped unexpectedly."""
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
        except Exception as exc:
            await self._retire_lost_connection(
                session,
                ws,
                f"WebSocket closed unexpectedly for session {session.id}: {exc}",
            )
        else:
            await self._retire_lost_connection(
                session,
                ws,
                f"WebSocket closed unexpectedly for session {session.id}",
            )

    # Maps each server event type to the handler method that processes it.
    _EVENT_HANDLERS: dict[str, str] = {
        _EVT_SPEECH_STARTED: "_on_speech_started",
        _EVT_SPEECH_STOPPED: "_on_speech_stopped",
        _EVT_AUDIO_DELTA: "_on_audio_delta",
        _EVT_TRANSCRIPT_DELTA: "_on_transcript_delta",
        _EVT_INPUT_TRANSCRIPT_DONE: "_on_input_transcript_done",
        _EVT_TRANSCRIPT_DONE: "_on_transcript_done",
        _EVT_OUTPUT_ITEM_DONE: "_on_output_item_done",
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
        # Server VAD: the caller holds the floor until its speech stops
        self._floor_held.add(session.id)
        logger.info(
            "[VAD] speech_start audio_start=%sms item=%s (session %s)",
            event.get("audio_start_ms", "?"),
            event.get("item_id", "?"),
            session.id,
        )
        await self._fire(self._speech_start_callbacks, session, label="speech_start")

    async def _on_speech_stopped(self, session: VoiceSession, event: dict[str, Any]) -> None:
        self._floor_held.discard(session.id)
        logger.info(
            "[VAD] speech_end audio_end=%sms item=%s (session %s)",
            event.get("audio_end_ms", "?"),
            event.get("item_id", "?"),
            session.id,
        )
        await self._fire(self._speech_end_callbacks, session, label="speech_end")
        if not self._server_answers_turns(session.id):
            # Nobody else asks for the response that covers the held continuation
            await self._continue_after_tool_results(session)

    def _server_answers_turns(self, session_id: str) -> bool:
        """Whether the server VAD requests the response to a turn itself.

        It does unless the session was configured with ``create_response``
        false; its response then covers a continuation held for the turn.
        """
        create_response = self._provider_configs.get(session_id, {}).get("create_response")
        return create_response is None or bool(create_response)

    async def _on_audio_delta(self, session: VoiceSession, event: dict[str, Any]) -> None:
        audio_b64 = event.get("delta", "")
        if audio_b64:
            audio_bytes = base64.b64decode(audio_b64)
            item_id = event.get("item_id", "")
            if item_id:
                state = self._output_audio.get(session.id)
                if state is None or state.item_id != item_id:
                    state = _OutputAudioState(
                        item_id=item_id,
                        content_index=int(event.get("content_index", 0)),
                    )
                    self._output_audio[session.id] = state
                state.received_bytes += len(audio_bytes)
            # Keep received_bytes in wire units for truncation accounting;
            # every RoomKit audio callback receives PCM16 regardless of codec.
            codec = self._audio_codecs.get(session.id, (None, None))[1]
            if codec is not None:
                audio_bytes = codec.decode(audio_bytes)
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

    async def _on_output_item_done(self, session: VoiceSession, event: dict[str, Any]) -> None:
        """A function call is handed on once its item is done, the first event
        that says whether the response cut it: ``function_call_arguments.done``
        comes before the item's status (RFC §6.4)."""
        item = event.get("item") or {}
        if item.get("type") != "function_call":
            return
        call_id = item.get("call_id") or ""
        name = item.get("name") or ""
        # A mapping, or the model's text when it does not read as one or the
        # output cap cut it: the channel refuses that call (RFC §12.4).
        cut = item.get("status") == "incomplete"
        arguments = realtime_call_arguments(item.get("arguments"), cut=cut)
        if self._book_tool_call(session, call_id):
            # Only a call the channel may answer holds the response open: one
            # without an id, or under an id in flight, is refused and reported
            # with nothing sent (RFC §12.4).
            pending = self._pending_responses.setdefault(session.id, PendingResponse())
            pending.call_ids.add(call_id)
            pending.had_calls = True
        await self._fire(
            self._tool_call_callbacks,
            session,
            call_id,
            name,
            arguments,
            label="tool_call",
        )

    async def _on_response_created(self, session: VoiceSession, event: dict[str, Any]) -> None:
        # A new response supersedes the last item's truncation target. Keep the
        # completed item after response.done while its buffered audio is still
        # playing, but never carry it into a subsequent response.
        self._output_audio.pop(session.id, None)
        # A call still open in an earlier response does not hold this one. A
        # result submitted after our request is not part of the response it
        # starts, so that response owes a continuation of its own.
        requested = self._pending_responses.get(session.id)
        owed = requested is not None and requested.requested and requested.had_calls
        self._pending_responses[session.id] = PendingResponse(had_calls=owed)
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
        await self._continue_after_tool_results(session, response_ended=True)
        await self._answer_owed_turn(session)

    async def _continue_after_tool_results(
        self, session: VoiceSession, *, response_ended: bool = False
    ) -> None:
        """Ask the model to go on once, when its response is done and fully answered.

        One ``response.create`` per result would start a second response while
        the first is active, and the API rejects it (RFC §12.4).
        """
        pending = self._pending_responses.get(session.id)
        if pending is None:
            return
        pending.finished = pending.finished or response_ended
        if not pending.settled:
            return
        ws = self._connections.get(session.id)
        if not pending.ready_to_continue or ws is None:
            del self._pending_responses[session.id]
            return
        if session.id in self._floor_held:
            # Held, not dropped: the request that answers the caller's turn
            # covers the results, which sit ahead of it in the conversation.
            logger.debug(
                "[%s] continuation held: the caller has the floor (session %s)",
                self._log_tag,
                session.id,
            )
            return
        await self._request_response(session, ws, "after tool results")

    def _response_in_progress(self, session_id: str) -> bool:
        """A response is active, or requested and not yet begun (RFC §12.4)."""
        pending = self._pending_responses.get(session_id)
        return session_id in self._responding or (pending is not None and pending.requested)

    async def _request_response(
        self, session: VoiceSession, ws: Any, why: str, *, owed: bool = False
    ) -> None:
        """Send ``response.create`` unless a response is already in progress.

        Every request goes through here, so none doubles another: the service
        rejects a second request while one is active, and the one requested
        counts as active until the server begins it (RFC §12.4). An *owed*
        request (the caller's turn) that meets one in progress is kept for
        when it ends; any request sent covers it.
        """
        if self._response_in_progress(session.id):
            if owed:
                self._turns_owed.add(session.id)
            logger.debug(
                "[%s] no response.create (%s): a response is in progress (session %s)",
                self._log_tag,
                why,
                session.id,
            )
            return
        self._turns_owed.discard(session.id)
        self._pending_responses[session.id] = PendingResponse(requested=True)
        logger.debug("[%s →] response.create (%s, session %s)", self._log_tag, why, session.id)
        await ws.send(json.dumps({"type": "response.create"}))

    async def _answer_owed_turn(self, session: VoiceSession) -> None:
        """Ask for the response a caller's turn is still owed, now that the
        one in its way has ended (RFC §12.4)."""
        ws = self._connections.get(session.id)
        if session.id not in self._turns_owed or ws is None:
            return
        self._turns_owed.discard(session.id)
        if session.id in self._floor_held:
            return  # the caller speaks again: that turn's own request covers both
        await self._request_response(session, ws, "caller's turn owed", owed=True)

    async def _on_buffer_committed(self, session: VoiceSession, event: dict[str, Any]) -> None:
        logger.debug("[%s] audio_buffer committed (session %s)", self._log_tag, session.id)

    async def _on_error(self, session: VoiceSession, event: dict[str, Any]) -> None:
        error = event.get("error", {})
        code = error.get("code", "unknown")
        message = error.get("message", "Unknown error")
        logger.error("[%s] error [%s] %s (session %s)", self._log_tag, code, message, session.id)
        self._release_rejected_request(session.id)
        await self._fire(self._error_callbacks, session, code, message, label="error")

    def _release_rejected_request(self, session_id: str) -> None:
        """An error while a request waits and no response is active answers
        that request: it is no longer in progress (RFC §12.4).

        The wire does not say which client event an error answers unless the
        client names it; an unrelated error in the same window costs at most a
        second request the service rejects, where keeping the state would
        leave the session asking for nothing ever again.
        """
        pending = self._pending_responses.get(session_id)
        if pending is not None and pending.requested and session_id not in self._responding:
            del self._pending_responses[session_id]

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
