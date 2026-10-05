"""Inbound server-message handling for the Gemini Live provider.

Translates Live API server messages into RoomKit provider callbacks: the
dispatch table keyed on the message field, one handler per field, and the
reading of ``interaction_status`` that decides when a response is over. The
tool-call messages are handled by
:class:`~roomkit.providers.gemini.realtime_tools.GeminiLiveToolsMixin`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from typing import Any

from roomkit.providers.ai.tool_calls import MALFORMED_CALL_NUDGE, is_malformed_call
from roomkit.providers.gemini.realtime_state import _GeminiSessionState, _GoAwayError
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.provider import RealtimeVoiceProvider

logger = logging.getLogger("roomkit.providers.gemini.realtime")


def _modality_counts(details: Any) -> dict[str, int]:
    """A list of ``ModalityTokenCount`` as ``{"AUDIO": n, "TEXT": n}``."""
    counts: dict[str, int] = {}
    for entry in details or ():
        modality = getattr(entry, "modality", None)
        name = getattr(modality, "name", None) or str(modality or "UNKNOWN")
        counts[name] = counts.get(name, 0) + (getattr(entry, "token_count", 0) or 0)
    return counts


def _usage_details(meta: Any) -> dict[str, Any]:
    """Everything the Live API reports beside its two totals.

    A spoken turn is mostly audio, the modalities are priced apart, and the
    cached share is priced apart again — so ``prompt_token_count`` alone
    cannot say what a session spent its context on. The OpenAI realtime
    handler already passes its own breakdown through ``_record_usage``; this
    is the same reading for Gemini, and an empty dict where the server sends
    none of it.
    """
    details: dict[str, Any] = {}
    for key in (
        "cached_content_token_count",
        "thoughts_token_count",
        "tool_use_prompt_token_count",
        "total_token_count",
    ):
        value = getattr(meta, key, 0) or 0
        if value:
            details[key] = value
    for key in (
        "prompt_tokens_details",
        "cache_tokens_details",
        "response_tokens_details",
        "tool_use_prompt_tokens_details",
    ):
        counts = _modality_counts(getattr(meta, key, None))
        if counts:
            details[key] = counts
    return details


_IDLE_STATUSES = frozenset({"IDLE", "INTERACTION_STATUS_IDLE"})
"""What ``interaction_status`` reads when the request is over. A set rather
than a suffix match: a future ``NOT_IDLE`` would end every response early."""


_KNOWN_INTERACTION_STATUSES = _IDLE_STATUSES | {"IN_PROGRESS", "INTERACTION_STATUS_IN_PROGRESS"}
"""Statuses that prove the server reports its interaction state. ``UNSPECIFIED``
does not: latching on it would retire ``turn_complete`` as an end-of-response
signal while nothing ever reads IDLE, and the session would never hand back."""


def _interaction_status_is_known(status: Any) -> bool:
    """Whether *status* is a state this build recognises."""
    return str(getattr(status, "value", status)).upper() in _KNOWN_INTERACTION_STATUSES


def _interaction_is_idle(status: Any) -> bool:
    """True when the server says the whole interaction is over.

    Accepts the SDK enum or the bare wire string: older SDKs hand back the
    latter, and the provider must not care which it got.
    """
    if status is None:
        return False
    return str(getattr(status, "value", status)).upper() in _IDLE_STATUSES


def _ended_on_malformed_call(content: Any) -> bool:
    """Whether the server ended the turn on a function call it could not parse."""
    reason = getattr(content, "turn_complete_reason", None)
    if reason is None:
        return False
    return is_malformed_call(getattr(reason, "name", None) or str(reason))


class GeminiLiveEventHandlersMixin(RealtimeVoiceProvider):
    """Server message to provider callback, one handler per field.

    Mixed into ``GeminiLiveProvider``, which owns the sessions and the
    receive loop that feeds :meth:`_handle_server_response`. This is where
    the end of a turn is told from the end of an interaction (3.8 speaks
    several times per request), where a barge-in ends the response once, and
    where a GoAway becomes a reconnect.
    """

    # Owned by GeminiLiveProvider / its other mixins; declared for typing. An
    # ``async def`` is declared as returning a ``Coroutine``: mypy rejects an
    # ``Awaitable`` annotation that precedes the implementation in the MRO.
    _sessions: dict[str, _GeminiSessionState]
    _log_event: Callable[..., None]
    _flush_transcription_buffer: Callable[[VoiceSession, str], Coroutine[Any, Any, None]]
    _handle_transcription_chunk: Callable[
        [VoiceSession, str, str, bool], Coroutine[Any, Any, None]
    ]

    # Ordered dispatch table for server response handling.  Each entry is
    # (response_attribute, handler_method).  Order matters: go_away is
    # processed LAST so all data in the message is handled first.
    _RESPONSE_HANDLERS: list[tuple[str, str]] = [
        ("session_resumption_update", "_on_session_resumption"),
        ("voice_activity", "_on_voice_activity"),
        ("server_content", "_on_server_content"),
        ("data", "_on_audio_data"),
        ("tool_call", "_on_tool_call"),
        ("tool_call_cancellation", "_on_tool_call_cancellation"),
        ("usage_metadata", "_on_usage_metadata"),
        ("go_away", "_on_go_away"),
    ]

    async def _handle_server_response(self, session: VoiceSession, response: Any) -> None:
        """Map Gemini Live responses to callbacks."""
        state = self._sessions.get(session.id)
        if state is None:
            return

        self._log_server_message(session, response)

        for attr, method in self._RESPONSE_HANDLERS:
            value = getattr(response, attr, None)
            if value:
                await getattr(self, method)(session, state, value)

    def _log_server_message(self, session: VoiceSession, response: Any) -> None:
        """Build a compact debug log line summarising the server message."""
        parts: list[str] = []
        if getattr(response, "data", None):
            parts.append(f"audio={len(response.data)}B")
        sc = getattr(response, "server_content", None)
        if sc:
            if getattr(sc, "model_turn", None):
                parts.append("model_turn")
            if getattr(sc, "turn_complete", None):
                parts.append("turn_complete")
            if getattr(sc, "interaction_status", None):
                parts.append(f"interaction={sc.interaction_status}")
            if getattr(sc, "interrupted", None):
                parts.append("interrupted")
            if getattr(sc, "input_transcription", None):
                parts.append(f"input_tx={sc.input_transcription.text!r}")
            if getattr(sc, "output_transcription", None):
                parts.append(f"output_tx={sc.output_transcription.text!r}")
        if getattr(response, "tool_call", None):
            parts.append("tool_call")
        va = getattr(response, "voice_activity", None)
        if va:
            parts.append(f"vad={getattr(va, 'voice_activity_type', '?')}")
        if getattr(response, "go_away", None):
            parts.append("go_away")
        if getattr(response, "session_resumption_update", None):
            parts.append("resumption_update")
        um = getattr(response, "usage_metadata", None)
        if um:
            parts.append(
                f"usage(prompt={getattr(um, 'prompt_token_count', '?')}"
                f",response={getattr(um, 'response_token_count', '?')}"
                f",total={getattr(um, 'total_token_count', '?')})"
            )
        if not parts:
            parts.append(f"unknown_keys={[k for k in dir(response) if not k.startswith('_')]}")
        logger.debug("[Gemini] recv: %s (session %s)", ", ".join(parts), session.id)

    async def _on_session_resumption(
        self, session: VoiceSession, state: _GeminiSessionState, update: Any
    ) -> None:
        if not update.resumable or state.provider_config.get("preserve_context"):
            state.resumption_handle = None
        elif update.new_handle:
            state.resumption_handle = update.new_handle
            logger.debug(
                "Session resumption handle updated for %s (resumable=%s)",
                session.id,
                update.resumable,
            )

    async def _on_voice_activity(
        self, session: VoiceSession, state: _GeminiSessionState, va: Any
    ) -> None:
        vtype = getattr(va, "voice_activity_type", None)
        if not vtype:
            return
        if vtype == "ACTIVITY_START":
            logger.info("[VAD] speech_start (session %s)", session.id)
            await self._user_speech_started(session, state)
        elif vtype == "ACTIVITY_END":
            logger.info("[VAD] speech_end (session %s)", session.id)
            state.user_speech_active = False
            await self._flush_transcription_buffer(session, "user")
            await self._fire(self._speech_end_callbacks, session, label="speech_end")

    async def _on_server_content(
        self, session: VoiceSession, state: _GeminiSessionState, content: Any
    ) -> None:
        # Input transcription (user speech-to-text)
        tr = getattr(content, "input_transcription", None)
        if tr and tr.text:
            await self._handle_transcription_chunk(session, tr.text, "user", bool(tr.finished))

        out_tr = getattr(content, "output_transcription", None)
        model_turn = getattr(content, "model_turn", None)
        if (tr and tr.text) or (out_tr and out_tr.text) or model_turn:
            state.has_conversation = True

        # The user's utterance is over the moment the model starts replying —
        # flush it BEFORE any assistant transcription goes out. One server
        # message can carry both the reply's first transcript chunk and
        # model_turn; emitting that chunk ahead of the user final inverts the
        # conversation downstream, where the late final reads as *new* user
        # speech (phantom barge-in, duplicated user entry).
        if not state.response_started and ((out_tr and out_tr.text) or model_turn):
            await self._flush_transcription_buffer(session, "user")

        # A response may first appear as output_transcription, as model_turn,
        # or as both in one message. Lift the assistant duplicate guard before
        # consuming that first chunk; doing it in the model_turn block below is
        # too late for the coalesced form and would discard a valid repeated
        # reply.
        if (
            not state.response_started
            and not state.assistant_response_observed
            and ((out_tr and out_tr.text) or model_turn)
        ):
            state.last_final_text.pop("assistant", None)
            state.assistant_response_observed = True

        # Output transcription (model speech-to-text)
        if out_tr and out_tr.text:
            await self._handle_transcription_chunk(
                session, out_tr.text, "assistant", bool(out_tr.finished)
            )

        # Model started generating
        if model_turn and not state.response_started:
            state.response_started = True
            state.response_ended_by_interrupt = False
            state.audio_chunk_count = 0
            logger.info("[Gemini] response_start (session %s)", session.id)
            self._log_event(session.id, "response_start", turn=state.turn_count)
            await self._fire(self._response_start_callbacks, session, label="response_start")

        if getattr(content, "interrupted", None):
            await self._on_interrupted(session, state)

        # Turn complete, and separately, interaction complete.
        #
        # Through 3.1 the two were the same event: one request, one spoken
        # turn, ``turn_complete`` at the end of it. From 3.8 the model
        # reasons and runs tools in the background while it keeps talking, so
        # it produces several turns per request and ``turn_complete`` no
        # longer means it has finished. ``interaction_status`` does: it reads
        # IN_PROGRESS while work remains and IDLE when the request is done.
        #
        # Firing ``response_end`` on every ``turn_complete`` would tell the
        # channel the reply is over while the model is still speaking, which
        # desynchronises the interruption handler and the bridge.
        status = getattr(content, "interaction_status", None)
        if status is not None and _interaction_status_is_known(status):
            state.reports_interaction_status = True
        interaction_done = _interaction_is_idle(status)

        turn_complete = bool(getattr(content, "turn_complete", None))
        if turn_complete:
            state.turn_count += 1
            logger.info(
                "[Gemini] turn_complete (session %s, %d audio chunks, status=%s)",
                session.id,
                state.audio_chunk_count,
                status,
            )
            self._log_event(
                session.id,
                "turn_complete",
                turn=state.turn_count,
                audio_chunks=state.audio_chunk_count,
                pending_tool_calls=self._open_tool_call_count(session),
                interaction_status=str(status) if status is not None else None,
            )

        if turn_complete or interaction_done:
            await self._flush_transcription_buffer(session, "user")
            await self._flush_transcription_buffer(session, "assistant")

        # Where the server reports its state, only IDLE closes the response.
        # Where it does not, ``turn_complete`` is the only signal there is and
        # keeps its old meaning, so 2.0 Flash Live and 2.5 native audio still
        # hand back control.
        if interaction_done or (turn_complete and not state.reports_interaction_status):
            state.response_started = False
            state.user_speech_active = False
            state.awaiting_new_user_utterance = True
            state.assistant_response_observed = False
            if state.response_ended_by_interrupt:
                # The barge-in above already ended this response. The server
                # still closes the interrupted request with turn_complete (and
                # IDLE from 3.8), and ending it again ran the channel's flush
                # and end-of-response signalling twice per interruption.
                state.response_ended_by_interrupt = False
            else:
                await self._fire(self._response_end_callbacks, session, label="response_end")

        # After the response's end: the reminder starts the model's next one.
        if turn_complete and _ended_on_malformed_call(content):
            await self._nudge_malformed_call(session, state)

    async def _on_interrupted(self, session: VoiceSession, state: _GeminiSessionState) -> None:
        """The user barged in while the model was speaking."""
        logger.info("[Gemini] INTERRUPTED — AI cut off by barge-in (session %s)", session.id)
        await self._flush_transcription_buffer(session, "assistant")
        # Fire speech_start ONLY if ACTIVITY_START wasn't already
        # received — Gemini doesn't always send voice_activity before
        # interrupted, so this may be the only trigger.
        if not state.user_speech_active:
            await self._user_speech_started(session, state)
        if state.response_started:
            state.response_started = False
            state.response_ended_by_interrupt = True
            await self._fire(self._response_end_callbacks, session, label="response_end")
        state.assistant_response_observed = False

    async def _user_speech_started(
        self, session: VoiceSession, state: _GeminiSessionState
    ) -> None:
        """Open a new user utterance, as the server detected it."""
        state.user_speech_active = True
        # A repeat of the previous words is legitimate again, and so is
        # another reminder of a call the model could not write.
        state.last_final_text.pop("user", None)
        state.malformed_call_nudged = False
        state.awaiting_new_user_utterance = False
        await self._fire(self._speech_start_callbacks, session, label="speech_start")

    async def _nudge_malformed_call(
        self, session: VoiceSession, state: _GeminiSessionState
    ) -> None:
        """Tell the model its function call could not be parsed and did not
        run, once until the user speaks again (RFC §12.4): the turn ended on
        it in silence, and nothing else would tell the model."""
        if state.malformed_call_nudged:
            logger.warning(
                "[Gemini] another unparsable function call ended a turn (session %s); "
                "the model was already told since the user last spoke",
                session.id,
            )
            return
        state.malformed_call_nudged = True
        logger.warning(
            "[Gemini] a turn ended on a function call the server could not parse "
            "(session %s); telling the model it did not run",
            session.id,
        )
        await self.inject_text(session, MALFORMED_CALL_NUDGE, role="system")

    async def _on_audio_data(
        self, session: VoiceSession, state: _GeminiSessionState, data: bytes
    ) -> None:
        state.audio_chunk_count += 1
        if state.audio_chunk_count % 50 == 1:
            logger.debug(
                "[Gemini] audio chunk #%d (%d bytes) for session %s",
                state.audio_chunk_count,
                len(data),
                session.id,
            )
        await self._fire(self._audio_callbacks, session, data, label="audio")

    async def _on_usage_metadata(
        self, session: VoiceSession, state: _GeminiSessionState, meta: Any
    ) -> None:
        prompt_tokens = getattr(meta, "prompt_token_count", 0) or 0
        response_tokens = getattr(meta, "response_token_count", 0) or 0
        total_tokens = getattr(meta, "total_token_count", 0) or 0
        logger.debug(
            "[Gemini] usage: prompt=%d response=%d total=%d (session %s)",
            prompt_tokens,
            response_tokens,
            total_tokens,
            session.id,
        )
        # Only log the "real" usage tick — when the model evaluates a
        # new prompt round (prompt_tokens > 0). Gemini emits one usage
        # event per audio chunk during a response, ALL with
        # prompt_tokens=0 — those would flood the diagnostic log
        # without telling us anything useful.
        if prompt_tokens:
            self._log_event(
                session.id,
                "usage",
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                total_tokens=total_tokens,
            )
        # Thinking is billed as output but counted outside the response count:
        # the output total counts it, as on the text path (RFC §6, §12.4.2).
        thoughts = getattr(meta, "thoughts_token_count", 0) or 0
        self._record_usage(
            session, prompt_tokens, response_tokens + thoughts, details=_usage_details(meta)
        )

    async def _on_go_away(
        self, session: VoiceSession, state: _GeminiSessionState, go_away: Any
    ) -> None:
        time_left = getattr(go_away, "time_left", "unknown")
        logger.warning(
            "Gemini GoAway received for session %s (time_left=%s)",
            session.id,
            time_left,
        )
        raise _GoAwayError()
