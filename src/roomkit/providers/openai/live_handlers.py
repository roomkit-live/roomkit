"""Inbound server-event handling for the OpenAI GPT-Live provider.

Translates Live API server events into RoomKit provider callbacks: the
dispatch table keyed on the wire event type, one handler per event, and the
turn synthesis a full-duplex wire calls for (RFC §12.4.1) — response and
speech boundaries closed by a quiet gap on each speaker's transcript. The
hosted-delegation envelope (``response.event``) is handled by
:class:`~roomkit.providers.openai.live_hosted.OpenAILiveHostedDelegationMixin`.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

from roomkit.providers.openai.live_config import _LOG_TAG, _LiveSession
from roomkit.providers.openai.live_events import (
    DELEGATION_TARGETS,
    EVT_DELEGATION_CREATED,
    EVT_ERROR,
    EVT_INPUT_TRANSCRIPT_DELTA,
    EVT_OUTPUT_AUDIO_DELTA,
    EVT_OUTPUT_TRANSCRIPT_DELTA,
    EVT_RESPONSE_EVENT,
    EVT_SESSION_CLOSED,
    EVT_SESSION_STARTED,
    EVT_SESSION_UPDATED,
    EVT_SESSION_USAGE_UPDATED,
    NOISY_EVENTS,
)
from roomkit.telemetry.base import Attr
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.provider import RealtimeVoiceProvider

logger = logging.getLogger("roomkit.providers.openai.live")


class OpenAILiveEventHandlersMixin(RealtimeVoiceProvider):
    """Server event → callback dispatch for the GPT-Live wire.

    Mixed into ``OpenAILiveProvider``, which owns the connection state
    (``_states``) and the model id. Handlers receive the session's connection
    state and the decoded event.
    """

    # Connection state owned by OpenAILiveProvider.__init__; declared for typing.
    _states: dict[str, _LiveSession]
    _model: str

    _EVENT_HANDLERS: dict[str, str] = {
        EVT_SESSION_STARTED: "_on_session_started",
        EVT_SESSION_UPDATED: "_on_session_updated",
        EVT_SESSION_CLOSED: "_on_session_closed",
        EVT_SESSION_USAGE_UPDATED: "_on_usage_updated",
        EVT_OUTPUT_AUDIO_DELTA: "_on_output_audio_delta",
        EVT_OUTPUT_TRANSCRIPT_DELTA: "_on_output_transcript_delta",
        EVT_INPUT_TRANSCRIPT_DELTA: "_on_input_transcript_delta",
        EVT_DELEGATION_CREATED: "_on_delegation_created",
        EVT_RESPONSE_EVENT: "_on_response_event",
        EVT_ERROR: "_on_error",
    }

    async def _handle_server_event(self, state: _LiveSession, event: dict[str, Any]) -> None:
        event_type = str(event.get("type", ""))
        if event_type not in NOISY_EVENTS:
            logger.debug(
                "[%s ←] %s %s",
                _LOG_TAG,
                event_type,
                {k: v for k, v in event.items() if k not in ("type", "delta", "audio")},
            )
        handler_name = self._EVENT_HANDLERS.get(event_type)
        if handler_name is not None:
            await getattr(self, handler_name)(state, event)

    async def _on_session_started(self, state: _LiveSession, event: dict[str, Any]) -> None:
        live = event.get("session") or {}
        state.session.provider_session_id = str(live.get("id") or state.session.id)
        logger.info(
            "[%s] session.started id=%s expires_at=%s (session %s)",
            _LOG_TAG,
            live.get("id"),
            live.get("expires_at"),
            state.session.id,
        )
        state.started.set()

    async def _on_session_updated(self, state: _LiveSession, event: dict[str, Any]) -> None:
        logger.debug("[%s] session.updated (session %s)", _LOG_TAG, state.session.id)

    async def _on_session_closed(self, state: _LiveSession, event: dict[str, Any]) -> None:
        usage = event.get("usage") or {}
        seconds = usage.get("seconds")
        if isinstance(seconds, int | float):
            self._record_live_seconds(state, float(seconds))
        logger.info(
            "[%s] session.closed reason=%s (session %s)",
            _LOG_TAG,
            event.get("reason"),
            state.session.id,
        )
        state.closed.set()

    async def _on_usage_updated(self, state: _LiveSession, event: dict[str, Any]) -> None:
        seconds = (event.get("usage") or {}).get("seconds")
        if isinstance(seconds, int | float):
            self._record_live_seconds(state, float(seconds))

    async def _on_output_audio_delta(self, state: _LiveSession, event: dict[str, Any]) -> None:
        audio_b64 = event.get("delta", "")
        if not audio_b64:
            return
        audio = base64.b64decode(audio_b64)
        if state.codec is not None:
            audio = state.codec.decode(audio)
        await self._fire(self._audio_callbacks, state.session, audio, label="audio")

    async def _on_output_transcript_delta(
        self, state: _LiveSession, event: dict[str, Any]
    ) -> None:
        delta = event.get("delta", "")
        if not delta:
            return
        # Opening the turn fires response_start before the first partial.
        await state.assistant_turn.feed(delta)
        await self._fire(
            self._transcription_callbacks,
            state.session,
            delta,
            "assistant",
            False,
            label="transcription",
        )

    async def _on_input_transcript_delta(self, state: _LiveSession, event: dict[str, Any]) -> None:
        delta = event.get("delta", "")
        if not delta:
            return
        await state.user_turn.feed(delta)
        await self._fire(
            self._transcription_callbacks,
            state.session,
            delta,
            "user",
            False,
            label="transcription",
        )

    async def _assistant_turn_opened(self, session: VoiceSession) -> None:
        state = self._states.get(session.id)
        if state is not None:
            state.responding = True
        logger.info("[%s] response_start (synthesized, session %s)", _LOG_TAG, session.id)
        await self._fire(self._response_start_callbacks, session, label="response_start")

    async def _assistant_turn_closed(self, session: VoiceSession, text: str) -> None:
        if text.strip():
            await self._fire(
                self._transcription_callbacks,
                session,
                text.strip(),
                "assistant",
                True,
                label="transcription",
            )
        state = self._states.get(session.id)
        if state is not None:
            state.responding = False
        logger.info("[%s] response_end (synthesized, session %s)", _LOG_TAG, session.id)
        await self._fire(self._response_end_callbacks, session, label="response_end")

    async def _user_turn_opened(self, session: VoiceSession) -> None:
        logger.info("[%s] speech_start (synthesized, session %s)", _LOG_TAG, session.id)
        await self._fire(self._speech_start_callbacks, session, label="speech_start")

    async def _user_turn_closed(self, session: VoiceSession, text: str) -> None:
        if text.strip():
            await self._fire(
                self._transcription_callbacks,
                session,
                text.strip(),
                "user",
                True,
                label="transcription",
            )
        logger.info("[%s] speech_end (synthesized, session %s)", _LOG_TAG, session.id)
        await self._fire(self._speech_end_callbacks, session, label="speech_end")

    async def _on_delegation_created(self, state: _LiveSession, event: dict[str, Any]) -> None:
        delegation = event.get("delegation") or {}
        delegation_id = delegation.get("id")
        target = DELEGATION_TARGETS.get(str(delegation.get("target", "")))
        if not delegation_id or target is None:
            logger.warning(
                "[%s] delegation without id or known target ignored: %s (session %s)",
                _LOG_TAG,
                delegation,
                state.session.id,
            )
            return
        logger.info(
            "[%s] delegation %s → %s backend (session %s)",
            _LOG_TAG,
            delegation_id,
            target,
            state.session.id,
        )
        await self._fire(
            self._delegation_callbacks,
            state.session,
            str(delegation_id),
            target,
            label="delegation",
        )

    async def _on_error(self, state: _LiveSession, event: dict[str, Any]) -> None:
        error = event.get("error") or {}
        code = str(error.get("code") or error.get("type") or "unknown")
        message = str(error.get("message") or "Unknown error")
        if error.get("param"):
            message = f"{message} (param: {error['param']})"
        logger.error("[%s] error [%s] %s (session %s)", _LOG_TAG, code, message, state.session.id)
        if not state.started.is_set():
            # No session has started on this connection, so this one will not.
            state.start_error = f"{code}: {message}"
            state.started.set()
            return
        await self._fire(self._error_callbacks, state.session, code, message, label="error")

    def _record_live_seconds(self, state: _LiveSession, seconds: float) -> None:
        """Record the live model's cumulative audio duration, never as tokens."""
        delta = max(0.0, seconds - state.live_seconds)
        state.live_seconds = seconds
        state.session._last_usage["live_seconds"] = seconds
        telemetry = getattr(self, "_telemetry", None)
        if telemetry is not None and delta > 0:
            telemetry.record_metric(
                "roomkit.realtime.live_seconds",
                delta,
                unit="s",
                attributes={"session_id": state.session.id, Attr.MODEL: self._model},
            )
