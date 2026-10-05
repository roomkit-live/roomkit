"""Tool-call handling for the Gemini Live provider.

The model's function calls, the results the application submits, and the
state of the calls still in flight: which ones block input, what got queued
behind them, and how a cancelled or orphaned call is released. Kept
apart from the server-message dispatch because it is the one piece of the
provider with a state machine of its own.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from typing import Any

from roomkit.providers.gemini.realtime_config import enum_value, genai_types, warn_unsupported
from roomkit.providers.gemini.realtime_models import live_model_profile
from roomkit.providers.gemini.realtime_state import _GeminiSessionState
from roomkit.providers.gemini.request import function_response_body
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.provider import RealtimeVoiceProvider

logger = logging.getLogger("roomkit.providers.gemini.realtime")


class GeminiLiveToolsMixin(RealtimeVoiceProvider):
    """Tool calls and the bookkeeping of the ones still in flight.

    Mixed into ``GeminiLiveProvider``, which owns the sessions and the model
    id. The provider's book (RFC §12.4) names every call the current
    connection issued and has not released, each with the function it named.
    From 3.8 a tool runs in the background by default and a call the model
    does wait on (a BLOCKING declaration, or any call on the pre-3.8 family)
    closes the input channel: ``blocking_call_ids`` says which, and the
    queued injections wait behind it. Three things release a call: the
    result, a server-side cancellation, or the loss of the connection that
    issued the id.
    """

    # Owned by GeminiLiveProvider / its other mixins; declared for typing. An
    # ``async def`` is declared as returning a ``Coroutine``: mypy rejects an
    # ``Awaitable`` annotation that precedes the implementation in the MRO.
    _model: str
    _sessions: dict[str, _GeminiSessionState]
    _get_active_state: Callable[[VoiceSession], _GeminiSessionState | None]
    _log_event: Callable[..., None]
    _send_text: Callable[[_GeminiSessionState, str, str, bool], Coroutine[Any, Any, None]]
    _send_image: Callable[[_GeminiSessionState, bytes, str, str, bool], Coroutine[Any, Any, None]]
    _flush_transcription_buffer: Callable[[VoiceSession, str], Coroutine[Any, Any, None]]

    async def submit_tool_result(self, session: VoiceSession, call_id: str, result: str) -> None:
        await self._submit_function_response(session, call_id, result, is_error=False)

    async def submit_tool_error(self, session: VoiceSession, call_id: str, result: str) -> None:
        """A failed call's result under the ``error`` key, Gemini's failure
        flag, as Gemini text sends it (RFC §12.4)."""
        await self._submit_function_response(session, call_id, result, is_error=True)

    async def _submit_function_response(
        self, session: VoiceSession, call_id: str, result: str, *, is_error: bool
    ) -> None:
        """Answer *call_id* with *result*, releasing the call as it goes."""
        types = genai_types()

        # A call the server discarded (tool_call_cancellation), one whose
        # connection or session is gone, or one never issued is off the book:
        # the server will not read its result, and a FunctionResponse for an
        # id it does not know is an error the application never asked for.
        # Checked before the connection is: the call was released the moment
        # its socket was lost, so a result arriving during the back-off or
        # after the session ended is stale, not an error (RFC §12.4).
        if not self._holds_tool_call(session, call_id):
            self._log_dropped_result(session, call_id)
            return
        state = self._sessions.get(session.id)
        if state is None or state.live_session is None:
            raise RuntimeError("Cannot deliver tool result without an active Gemini connection")
        # A scheduling the config cannot take is refused before the call
        # leaves the book: it stays owed, and is abandoned as any other.
        scheduling = self._response_scheduling(state)
        # Released as its result goes, before the send yields: the channel
        # frees the id at the same step, so a call the model issues under it
        # meanwhile is a new call to both (RFC §12.4).
        _, name = self._answerable_tool_call(session, call_id)
        was_blocking = call_id in state.blocking_call_ids
        state.blocking_call_ids.discard(call_id)

        self._log_tool_result(state, session, call_id, result)

        # The response names the function the call named. The id alone was
        # enough through 3.1, and the name went out empty on that account;
        # gemini-3.8-live-extended-thinking reads an unnamed response as a
        # call that failed and tells the user a system error occurred, while
        # the same payload under the call's name is read as the result
        # (verified against the live API, 2026-09-18).
        #
        # A background call returns while the model is mid-sentence, so the
        # response has to say when to use it. WHEN_IDLE waits for the end of
        # what is being said, which is what a voice agent wants by default;
        # INTERRUPT cuts in, SILENT files it into context without a word. A
        # blocking call needs none of this: the model is already waiting, and
        # leaving the field unset keeps the pre-3.8 wire byte for byte.
        response_kwargs: dict[str, Any] = {
            "id": call_id,
            "name": name,
            "response": function_response_body(result, is_error=is_error),
        }
        if scheduling is not None and not was_blocking:
            response_kwargs["scheduling"] = scheduling

        await state.live_session.send_tool_response(
            function_responses=[types.FunctionResponse(**response_kwargs)],
        )

        # Flush what its blocking waited on.
        if not state.blocking_call_ids:
            await self._flush_queued_injections(state)

    def _response_scheduling(self, state: _GeminiSessionState) -> Any:
        """The scheduling a background call's result is sent with, as the
        session's config asks for it; ``None`` when it asks for none or the
        model refuses it (warned once).

        Only when the caller asks. A default here looked harmless and was
        not: gemini-3.8-live-extended-thinking closes the session with
        `1007 Function response scheduling is not supported for this model`,
        and the models that do take it already deliver a background result
        sensibly on their own. Nothing to gain, a session to lose.
        """
        scheduling = state.provider_config.get("tool_response_scheduling")
        if not scheduling:
            return None
        if not live_model_profile(self._model).response_scheduling:
            warn_unsupported(self._model, "tool_response_scheduling", state.warned_unsupported)
            return None
        return enum_value(
            genai_types().FunctionResponseScheduling, scheduling, "tool_response_scheduling"
        )

    def _log_tool_result(
        self, state: _GeminiSessionState, session: VoiceSession, call_id: str, result: str
    ) -> None:
        """Count, log and size-check a tool result before it goes out."""
        # Track tool result bytes for debugging
        state.tool_result_bytes += len(result)

        # Diagnostic: log every tool result we send back to Gemini so
        # the request → response → result cycle is visible end-to-end.
        # Body is truncated to 800 chars in the log; the full thing is
        # still sent to Gemini.
        self._log_event(
            session.id,
            "submit_tool_result",
            call_id=call_id,
            len=len(result),
            preview=(result[:800] + ("…" if len(result) > 800 else "")),
        )

        if len(result) > 16384:
            logger.warning(
                "Large tool result (%d chars) for call %s may cause Gemini to "
                "disconnect or silently fail (session %s)",
                len(result),
                call_id,
                session.id,
            )

    async def _flush_queued_injections(self, state: _GeminiSessionState) -> None:
        """Send what the blocking calls held back, text first, then images.

        Called once nothing blocks any more: a result came back, the server
        cancelled the call, or the connection that owned it is gone. Each
        queue drains from its head, so an injection made meanwhile queues
        behind it rather than overtaking it, and a call that blocks again
        stops the drain.
        """
        while state.queued_text_injections and not state.blocking_call_ids:
            text, role, silent = state.queued_text_injections.pop(0)
            logger.debug(
                "Flushing queued text injection for session %s (len=%d)",
                state.session.id,
                len(text),
            )
            await self._send_text(state, text, role, silent)
        while state.queued_injections and not state.blocking_call_ids:
            image_data, mime_type, prompt, silent = state.queued_injections.pop(0)
            logger.debug(
                "Flushing queued image injection for session %s (mime=%s, size=%d)",
                state.session.id,
                mime_type,
                len(image_data),
            )
            await self._send_image(state, image_data, mime_type, prompt, silent)

    async def _release_calls_lost_with_the_connection(self, state: _GeminiSessionState) -> None:
        """Forget every tool call the old socket issued, then deliver the
        injections a blocking one held.

        Call ids are connection-scoped: the new socket never issued them and
        will not read their results, blocking or not. Left in the books, a
        blocking one held every injection queued behind it until the
        application's handler finished work the model had already lost, and
        the result of any of them then went out for an id the server did not
        know.
        """
        await self._abandon_open_calls(state)
        try:
            await self._flush_queued_injections(state)
        except Exception:
            # The reconnect stands; what could not be delivered is logged.
            logger.warning(
                "Failed to flush queued injections after reconnecting session %s",
                state.session.id,
                exc_info=True,
            )

    async def _abandon_open_calls(self, state: _GeminiSessionState) -> None:
        """Forget every tool call the connection issued and report them: no
        other connection will read their results (RFC §12.4)."""
        state.blocking_call_ids.clear()
        orphaned = sorted(self._take_tool_calls(state.session))
        if not orphaned:
            return
        logger.info(
            "[Gemini] %d tool call(s) did not survive the connection (session %s)",
            len(orphaned),
            state.session.id,
        )
        # The application is still working for the old socket. Same fact as
        # a server cancellation: the model will not read the result.
        await self._abandon_tool_calls(state.session, orphaned)

    async def _on_tool_call(
        self, session: VoiceSession, state: _GeminiSessionState, tool_call: Any
    ) -> None:
        # A tool call is the model acting on the user's utterance — the
        # utterance is over. Flush its final before emitting the call, for the
        # same reason as the output-transcription flush: consumers must see
        # the user final ahead of everything the model does in answer to it
        # (a late final reads as new user speech downstream).
        await self._flush_transcription_buffer(session, "user")
        state.has_conversation = True
        for fc in tool_call.function_calls:
            # A call without an id can be neither answered nor cancelled, nor
            # can a second call under an id in flight: there is nothing to keep
            # for either, the channel refusing them (RFC §12.4). One with an id
            # belongs to this connection now, even if an earlier one cancelled
            # the same id.
            if (
                self._book_tool_call(session, fc.id or "", fc.name or "")
                and fc.name in state.blocking_tool_names
            ):
                state.blocking_call_ids.add(fc.id)
            args_dict = dict(fc.args) if fc.args else {}
            self._log_event(
                session.id,
                "function_call",
                name=fc.name,
                id=fc.id,
                args=args_dict,
            )
            await self._fire(
                self._tool_call_callbacks,
                session,
                fc.id or "",
                fc.name or "",
                args_dict,
                label="tool_call",
            )

    async def _on_tool_call_cancellation(
        self, session: VoiceSession, state: _GeminiSessionState, cancellation: Any
    ) -> None:
        """Release the calls the server discarded.

        Sent when the user interrupts while calls are outstanding: the server
        will not read their results, and a blocking one no longer holds the
        input channel. Left in the books, the id kept every injection queued
        until the application's handler finished work the model had already
        abandoned, and if the stale FunctionResponse then failed to send,
        nothing else ever cleared it. The application hears of it through
        ``on_tool_call_cancelled`` and stops the handler; a result that still
        arrives is dropped.
        """
        ids = [call_id for call_id in (getattr(cancellation, "ids", None) or []) if call_id]
        if not ids:
            return
        logger.info("[Gemini] server cancelled tool call(s) %s (session %s)", ids, session.id)
        self._log_event(session.id, "tool_call_cancellation", ids=ids)
        state.blocking_call_ids.difference_update(ids)
        await self._abandon_open_tool_calls(session, ids)
        if not state.blocking_call_ids:
            await self._flush_queued_injections(state)
