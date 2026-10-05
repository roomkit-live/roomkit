"""Connection lifecycle of the Gemini Live provider after the first connect.

The loop that reads the socket and the machine that reopens it: exponential
back-off up to a bound, proactive reconnect on GoAway, fail-fast on the close
codes that a retry would only reproduce, resumption handles, and the replay
of the caller audio buffered while the socket was down. The messages the
loop reads are dispatched by
:class:`~roomkit.providers.gemini.realtime_handlers.GeminiLiveEventHandlersMixin`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any

from roomkit.core.task_utils import _finish_cleanup
from roomkit.providers.gemini.realtime_state import _GeminiSessionState, _GoAwayError
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.provider import RealtimeVoiceProvider

logger = logging.getLogger("roomkit.providers.gemini.realtime")


class GeminiLiveConnectionMixin(RealtimeVoiceProvider):
    """The receive loop and the reconnect machine of a Live session.

    Mixed into ``GeminiLiveProvider``, which opens the first connection in
    ``connect`` and owns the client. This is where a dropped socket is
    retried with back-off, a GoAway becomes a proactive reconnect, a
    permanent close code ends the session with its reason, and a session
    that must keep its whole context stops rather than resume.
    """

    # Owned by GeminiLiveProvider / its other mixins; declared for typing. An
    # ``async def`` is declared as returning a ``Coroutine``: mypy rejects an
    # ``Awaitable`` annotation that precedes the implementation in the MRO.
    _sessions: dict[str, _GeminiSessionState]
    _client: Any
    _model: str
    _clear_transcription_buffers: Callable[[str], None]
    _make_audio_blob: Callable[[bytes, int], Any]
    _release_calls_lost_with_the_connection: Callable[
        [_GeminiSessionState], Coroutine[Any, Any, None]
    ]
    _abandon_open_calls: Callable[[_GeminiSessionState], Coroutine[Any, Any, None]]
    _handle_server_response: Callable[[VoiceSession, Any], Coroutine[Any, Any, None]]

    async def _open_live_session(self, live_config: Any) -> tuple[Any, Any]:
        """Open one Live connection for *live_config*: the context manager and its session.

        The one place the socket is opened, for ``connect`` and for both
        attempts of ``_reconnect``; the caller owns the closing.
        """
        ctxmgr = self._client.aio.live.connect(model=self._model, config=live_config)
        live_session = await ctxmgr.__aenter__()
        return ctxmgr, live_session

    def _start_receive_loop(self, state: _GeminiSessionState) -> None:
        """Start the receive loop of *state*'s session on its own task.

        In a context of its own: ``reconfigure`` runs inside whatever called
        it, a tool handler for a handoff, and a loop created there would carry
        that call's context (its voice session, its AI loop, the call it
        serves) into every event of the new connection.
        """
        session = state.session
        state.receive_task = self._session_task(
            self._receive_loop(session), name=f"gemini_live_recv:{session.id}"
        )

    _MAX_RECONNECTS = 5

    # WebSocket close codes that signal a *permanent* setup/policy problem,
    # not a transient drop. Reconnecting with the same config just reproduces
    # the failure (and stalls the user ~10s through 5 back-off retries), so we
    # fail fast and surface the exact code to the embedder instead.
    #   1007 — invalid argument (e.g. a tool schema Gemini Live won't accept)
    #   1008 — policy violation
    #   1011 — internal error, used by Gemini for quota/billing exhaustion
    _NON_RETRYABLE_CLOSE_CODES = frozenset({1007, 1008, 1011})

    @property
    def supports_context_preservation(self) -> bool:
        """Strict sessions disable compression and stop before reconnection."""
        return True

    async def _end_preserved_context(
        self, session: VoiceSession, state: _GeminiSessionState
    ) -> None:
        state.audio_buffer.clear()
        state.queued_text_injections.clear()
        state.queued_injections.clear()
        await self._abandon_open_calls(state)
        session.state = VoiceSessionState.ENDED
        await self._fire(
            self._error_callbacks,
            session,
            "context_preservation_ended",
            "Voice session ended because its full instruction context could not be "
            "preserved across a connection change. Start a new session; pending "
            "operations have not been replayed.",
            label="error",
        )

    async def _receive_loop(self, session: VoiceSession) -> None:
        """Process server events from Gemini Live API.

        If the connection drops mid-session, the loop will attempt to
        reconnect up to ``_MAX_RECONNECTS`` times with exponential back-off.
        """
        reconnect_count = 0

        while True:
            state = self._sessions.get(session.id)
            if state is None:
                return

            # If the session was closed by the user, stop the loop
            if session.state == VoiceSessionState.ENDED:
                return

            # Handle reconnection if needed
            if state.live_session is None:
                if state.provider_config.get("preserve_context"):
                    await self._end_preserved_context(session, state)
                    return
                session.state = VoiceSessionState.CONNECTING
                reconnect_count += 1
                if reconnect_count > self._MAX_RECONNECTS:
                    logger.error(
                        "Gemini Live session %s: connection lost, max reconnects (%d) reached",
                        session.id,
                        self._MAX_RECONNECTS,
                    )
                    state.audio_buffer.clear()
                    await self._abandon_open_calls(state)
                    session.state = VoiceSessionState.ENDED
                    await self._fire(
                        self._error_callbacks,
                        session,
                        "max_reconnects",
                        f"Connection lost after {self._MAX_RECONNECTS} reconnect attempts",
                        label="error",
                    )
                    return

                # The calls the old socket issued die with it. Say so now, not
                # after the handshake: a handler finishing during the back-off
                # would otherwise submit into a connection that is not there,
                # and an error is not what happened to its call.
                await self._release_calls_lost_with_the_connection(state)

                delay = min(0.5 * (2 ** (reconnect_count - 1)), 4.0)
                logger.warning(
                    "Gemini Live connection lost for session %s (attempt %d/%d), "
                    "reconnecting in %.1fs…",
                    session.id,
                    reconnect_count,
                    self._MAX_RECONNECTS,
                    delay,
                )
                await asyncio.sleep(delay)

                try:
                    await self._reconnect(session)
                except Exception:
                    logger.exception("Reconnect failed for session %s", session.id)
                    continue

            # Process messages from the current session
            try:
                # local ref to live_session
                live_session = state.live_session
                if live_session is None:
                    continue

                async for response in live_session.receive():
                    reconnect_count = 0
                    await self._handle_server_response(session, response)

            except asyncio.CancelledError:
                raise
            except _GoAwayError:
                # Server warned it's about to disconnect — proactive reconnect.
                # This does NOT count against the reconnect limit.
                logger.info("Proactive reconnect (GoAway) for session %s", session.id)
                state.live_session = None
                reconnect_count = 0
            except Exception as exc:
                if session.state == VoiceSessionState.ENDED:
                    return  # state may be mutated by close() during await

                uptime = time.monotonic() - state.started_at if state.started_at else 0.0
                close_code = getattr(exc, "code", None)
                # Extract detailed error info from Gemini APIError
                response_json = getattr(exc, "response_json", None)
                status_code = getattr(exc, "status_code", None)
                logger.warning(
                    "Gemini session %s disconnected — "
                    "uptime=%.1fs, turns=%d, tool_result_bytes=%d, "
                    "audio_chunks=%d, close_code=%s, error=%s: %s, "
                    "status_code=%s, response_json=%s, pending_tools=%d",
                    session.id,
                    uptime,
                    state.turn_count,
                    state.tool_result_bytes,
                    state.audio_chunk_count,
                    close_code,
                    type(exc).__name__,
                    exc,
                    status_code,
                    response_json,
                    self._open_tool_call_count(state.session),
                )
                state.live_session = None

                # Permanent failures (bad config, quota) won't recover by
                # reconnecting — end the session now and surface the exact
                # close code so the embedder can show the user a precise reason
                # instead of a silent disconnect after 5 useless retries.
                if close_code in self._NON_RETRYABLE_CLOSE_CODES:
                    state.audio_buffer.clear()
                    await self._abandon_open_calls(state)
                    session.state = VoiceSessionState.ENDED
                    await self._fire(
                        self._error_callbacks,
                        session,
                        f"ws_{close_code}",
                        str(exc),
                        label="error",
                    )
                    return

                # Suppress duplicate send_audio_failed errors during reconnect
                state.error_suppressed = True

    async def _reconnect(self, session: VoiceSession) -> None:
        """Reconnect to Gemini Live using the stored config."""
        state = self._sessions.get(session.id)
        if state is None or session.state == VoiceSessionState.ENDED:
            raise RuntimeError("No session state for reconnection")
        if state.provider_config.get("preserve_context"):
            await self._end_preserved_context(session, state)
            raise RuntimeError("Cannot resume Gemini with uncertain context")

        # Suppress audio sends during reconnection
        session.state = VoiceSessionState.CONNECTING

        # Tear down old connection
        old_ctxmgr = state.ctxmgr
        state.ctxmgr = None
        state.live_session = None
        if old_ctxmgr:
            with contextlib.suppress(Exception):
                await old_ctxmgr.__aexit__(None, None, None)
        # The calls the old socket issued die with it: said now, before the
        # handshake, as the receive loop says it (a no-op when it already did).
        await self._release_calls_lost_with_the_connection(state)

        # Clear stale transcription buffers
        self._clear_transcription_buffers(session.id)

        live_config = state.live_config
        if not live_config:
            raise RuntimeError("No stored config for reconnection")

        # Use stored resumption handle to preserve conversation context
        resumption_handle = state.resumption_handle
        if resumption_handle and live_config.session_resumption is not None:
            live_config.session_resumption.handle = resumption_handle
            logger.info("Reconnecting session %s with resumption handle", session.id)
        elif live_config.session_resumption is not None:
            # No handle available — reset to None for a fresh session
            live_config.session_resumption.handle = None

        try:
            ctxmgr, live_session = await self._open_live_session(live_config)
        except Exception as exc:
            # Fallback: if reconnection with handle failed, try one fresh connect
            if resumption_handle and live_config.session_resumption is not None:
                logger.warning(
                    "Gemini reconnection with handle failed for %s, trying fresh: %s",
                    session.id,
                    exc,
                )
                state.resumption_handle = None
                live_config.session_resumption.handle = None
                ctxmgr, live_session = await self._open_live_session(live_config)
            else:
                raise

        # Keep ownership local until replay succeeds. A concurrent disconnect
        # cannot close this context twice or publish a late ACTIVE session.
        def check_owner() -> None:
            if (
                self._sessions.get(session.id) is not state
                or session.state == VoiceSessionState.ENDED
            ):
                raise asyncio.CancelledError("Session ended during reconnection")

        try:
            check_owner()
            while (chunk := state.pop_audio()) is not None:
                await live_session.send_realtime_input(
                    audio=self._make_audio_blob(chunk, state.input_sample_rate),
                )
                check_owner()
                state.realtime_input_sent = True
        except BaseException:
            await _finish_cleanup(ctxmgr.__aexit__(None, None, None))
            raise

        # No await between observing an empty buffer and activating: new
        # microphone frames cannot overtake buffered speech.
        state.ctxmgr = ctxmgr
        state.live_session = live_session
        state.response_started = False
        state.response_ended_by_interrupt = False
        session.state = VoiceSessionState.ACTIVE

        # Re-enable error callbacks for the next reconnection cycle
        state.error_suppressed = False

        logger.info("Gemini Live session %s reconnected", session.id)
