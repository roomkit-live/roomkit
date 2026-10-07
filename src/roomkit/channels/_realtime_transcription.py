"""Transcription processing for RealtimeVoiceChannel."""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.channels._realtime_tool_calls import transcript_starts_answer
from roomkit.core.exceptions import ChannelNotFoundError
from roomkit.models.enums import HookTrigger
from roomkit.models.event import TextContent
from roomkit.voice.realtime._answer_depth import AnswerDepth
from roomkit.voice.realtime.provider import RealtimeVoiceProvider

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.voice.base import VoiceSession

logger = logging.getLogger("roomkit.channels.realtime_voice")


@runtime_checkable
class RealtimeTranscriptionHost(Protocol):
    """Contract: capabilities a host class must provide for RealtimeTranscriptionMixin.

    Attributes provided by the host's ``__init__``:
        _state_lock: Guards mutable per-session state from concurrent access.
        _session_rooms: Maps session IDs to room IDs.
        _barge_in_active: Session IDs with an active barge-in.
        _last_assistant_text: Last assistant utterance per session.
        _answer_depth: What each session's model heard last (RFC §12.4).
        _emit_transcription_events: Whether to emit transcriptions as RoomEvents.
        _framework: The RoomKit framework instance (or None).
        channel_id: Channel identifier.
        provider_name: Name of the realtime voice provider.

    Cross-mixin methods (implemented elsewhere in the MRO):
        _track_task: Schedule an async task with exception handling.
        _send_client_message: Send a JSON message to the client UI.
        _rt_span_ctx: Get the telemetry span context for a session.
    """

    _provider: RealtimeVoiceProvider
    _state_lock: threading.Lock
    _session_rooms: dict[str, str]
    _barge_in_active: set[str]
    _last_assistant_text: dict[str, str]
    _answer_depth: dict[str, AnswerDepth]
    _user_turn_start_at: dict[str, Any]
    _emit_transcription_events: bool
    _framework: RoomKit | None
    _transcription_order_locks: dict[str, asyncio.Lock]
    channel_id: str
    provider_name: str | None

    def _track_task(self, loop: Any, coro: Any, *, name: str) -> Any: ...

    async def _send_client_message(self, session: Any, message: dict[str, Any]) -> None: ...

    def _rt_span_ctx(self, session_id: str) -> tuple[Any, Any]: ...


class RealtimeTranscriptionMixin:
    """Transcription hooks and RoomEvent emission for RealtimeVoiceChannel.

    Host contract: :class:`RealtimeTranscriptionHost`.
    """

    _provider: RealtimeVoiceProvider
    _state_lock: threading.Lock
    _session_rooms: dict[str, str]
    _barge_in_active: set[str]
    _last_assistant_text: dict[str, str]
    _answer_depth: dict[str, AnswerDepth]
    _user_turn_start_at: dict[str, Any]
    _emit_transcription_events: bool
    _framework: RoomKit | None
    _transcription_order_locks: dict[str, asyncio.Lock]
    channel_id: str
    provider_name: str | None

    _note_provider_output: Any
    _track_task: Any  # see RealtimeTranscriptionHost — cross-mixin
    _send_client_message: Any  # see RealtimeTranscriptionHost — cross-mixin
    _rt_span_ctx: Any  # see RealtimeTranscriptionHost — cross-mixin

    def _init_transcription_state(self) -> None:
        """The per-session state this mixin keeps, empty."""
        # Last assistant text per session (for barge-in event context)
        self._last_assistant_text = {}
        # What each session's model heard last, which its answer's chain
        # depth follows (RFC §12.4, §8.3).
        self._answer_depth = {}
        # FIFO lock per session keeping transcription processing in arrival
        # order — each event runs in its own task and the partial/final code
        # paths await a different number of hops (see _process_transcription).
        self._transcription_order_locks = {}

    def _forget_transcription_state(self, session_id: str) -> None:
        """Drop an ended session's transcription state. Call under ``_state_lock``."""
        self._last_assistant_text.pop(session_id, None)
        self._answer_depth.pop(session_id, None)
        self._transcription_order_locks.pop(session_id, None)

    def _on_provider_transcription(
        self, session: VoiceSession, text: str, role: str, is_final: bool
    ) -> Any:
        """Handle transcription from provider."""
        if transcript_starts_answer(self._provider.full_duplex, role, text, is_final):
            self._note_provider_output(session.id)
        answer_depth = self._session_answer_depth(session.id)
        if role == "user":
            # Noted on arrival, partial or final: the answer being said may
            # be transcribed before the user's final transcription is.
            answer_depth.user_spoke()
        # Read on arrival too: processing runs in a task, after an injection
        # that arrives meanwhile and must not deepen what came before it.
        chain_depth = 0 if role == "user" else answer_depth.answer
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._track_task(
            loop,
            self._process_transcription(session, text, role, is_final, chain_depth),
            name=f"rt_transcription:{session.id}",
        )

    async def _process_transcription(
        self, session: VoiceSession, text: str, role: str, is_final: bool, chain_depth: int
    ) -> None:
        """Process a transcription: fire hooks, emit event, send to client.

        Partial transcriptions skip hooks and telemetry spans — they are
        forwarded to the client UI only.  Final transcriptions go through
        the full pipeline: hooks, client UI, and RoomEvent emission.

        Each event arrives in its own task, and the partial path awaits one
        hop more than the final path — unserialised, a final overtakes the
        partial that preceded it on the wire and the late partial resurrects
        an already-finalised utterance downstream (duplicate chat bubbles).
        The per-session lock is FIFO, so processing keeps arrival order.
        """
        if not self._framework:
            return

        with self._state_lock:
            lock = self._transcription_order_locks.setdefault(session.id, asyncio.Lock())
        async with lock:
            await self._process_transcription_locked(session, text, role, is_final, chain_depth)

    async def _process_transcription_locked(
        self, session: VoiceSession, text: str, role: str, is_final: bool, chain_depth: int
    ) -> None:
        if not self._framework:
            return

        with self._state_lock:
            room_id = self._session_rooms.get(session.id)
        if not room_id:
            return

        # Partial transcriptions: send to client UI and fire async hooks
        if not is_final:
            try:
                await self._send_client_message(
                    session,
                    {
                        "type": "transcription",
                        "text": text,
                        "role": role,
                        "is_final": False,
                    },
                )
            except Exception:
                logger.exception("Error sending partial transcription for session %s", session.id)

            # Skip expensive _build_context when no hooks are registered —
            # partials stream continuously while the AI speaks, so this path
            # runs many times per second on the event loop.
            if not self._framework.hook_engine.has_hooks(HookTrigger.ON_PARTIAL_TRANSCRIPTION):
                return

            from roomkit.voice.events import PartialTranscriptionEvent

            partial_event = PartialTranscriptionEvent(
                session=session,
                text=text,
                confidence=0.0,
                is_stable=False,
                role=role,
            )
            context = await self._framework._build_context(room_id)
            await self._framework.hook_engine.run_async_hooks(
                room_id,
                HookTrigger.ON_PARTIAL_TRANSCRIPTION,
                partial_event,
                context,
                skip_event_filter=True,
            )
            return

        # Tool-call-as-text recovery: detect when the model speaks a tool
        # call (e.g. "call:send_to_agent{task:...}") instead of invoking it.
        if role == "assistant" and getattr(self, "_tool_recovery_enabled", False):
            recovered, remaining = self._try_recover_tool_call_from_text(session, text)  # ty: ignore[unresolved-attribute]
            if recovered:
                if not remaining or not remaining.strip():
                    return  # entire text was a tool call — suppress
                text = remaining  # continue with remaining speech

        from roomkit.telemetry.context import reset_span

        _, _tok = self._rt_span_ctx(session.id)
        try:
            context = await self._framework._build_context(room_id)

            from roomkit.voice.realtime.events import RealtimeTranscriptionEvent

            # Check and clear barge-in state for user transcriptions.
            was_barge_in = role == "user" and session.id in self._barge_in_active
            if was_barge_in and is_final:
                self._barge_in_active.discard(session.id)

            transcription_event = RealtimeTranscriptionEvent(
                session=session,
                text=text,
                role=role,  # ty: ignore[invalid-argument-type]
                is_final=is_final,
                was_barge_in=was_barge_in,
            )

            hook_result = await self._framework.hook_engine.run_sync_hooks(
                room_id,
                HookTrigger.ON_TRANSCRIPTION,
                transcription_event,
                context,
                skip_event_filter=True,
            )

            if not hook_result.allowed:
                logger.info("Transcription blocked by hook: %s", hook_result.reason)
                return

            # Use potentially modified text
            final_text = text
            if hook_result.event is not None and isinstance(
                hook_result.event, RealtimeTranscriptionEvent
            ):
                final_text = hook_result.event.text
            elif isinstance(hook_result.event, str):
                final_text = hook_result.event

            await self._send_client_message(
                session,
                {
                    "type": "transcription",
                    "text": final_text,
                    "role": role,
                    "is_final": is_final,
                },
            )

            # Track last assistant text for barge-in context
            if role == "assistant":
                self._last_assistant_text[session.id] = final_text

            if self._emit_transcription_events and final_text.strip():
                await self._emit_transcript_event(session, room_id, role, final_text, chain_depth)

        except ChannelNotFoundError:
            # Benign teardown race: the channel was detached from the room
            # between reading the binding and emitting the event. Expected when
            # the assistant's final utterance is the one that ends the session —
            # its transcript finalizes after end_session has torn the channel
            # down, so this trailing event has nowhere to land. Drop it quietly;
            # a full ERROR traceback here is noise, not a failure.
            logger.debug(
                "Channel detached during teardown; dropped trailing transcription "
                "for session %s (room=%s)",
                session.id,
                room_id,
            )
        except Exception:
            logger.exception(
                "Error processing transcription for session %s (room=%s, is_final=%s)",
                session.id,
                room_id,
                is_final,
            )
        finally:
            if _tok is not None:
                reset_span(_tok)

    async def _emit_transcript_event(
        self, session: VoiceSession, room_id: str, role: str, final_text: str, chain_depth: int
    ) -> None:
        """Store a final transcription as the room's RoomEvent, at *chain_depth*.

        The user's words open a chain; the model's answer carries the depth
        of what it heard plus one (RFC §12.4).

        User transcriptions are finalized at turn_complete, which often
        happens AFTER any tool_calls the agent fired mid-turn. Stamp user
        turns with the pipeline-VAD SPEECH_START timestamp (captured in
        _on_pipeline_speech_start) so they sort chronologically before the
        tool calls they triggered — matching what the user actually
        experienced (said the thing, then the agent reacted). Fall back to a
        small offset if VAD didn't fire (e.g., server-VAD mode without a
        local pipeline) so ordering is still reasonable.
        """
        assert self._framework is not None
        participant_id = session.participant_id if role == "user" else None
        logger.info(
            "Emitting transcription as RoomEvent: role=%s, text=%s",
            role,
            final_text,
        )
        created_at = None
        if role == "user":
            with self._state_lock:
                turn_start = self._user_turn_start_at.pop(session.id, None)
            created_at = turn_start or (datetime.now(UTC) - timedelta(seconds=2))
        await self._framework.send_event(
            room_id,
            self.channel_id,
            TextContent(body=final_text),
            chain_depth=chain_depth,
            participant_id=participant_id,
            metadata={
                "voice_session_id": session.id,
                "source": "realtime_voice",
                "role": role,
            },
            provider=self.provider_name,
            created_at=created_at,
        )

    def _session_answer_depth(self, session_id: str) -> AnswerDepth:
        """What the session's model heard last (RFC §12.4)."""
        with self._state_lock:
            return self._answer_depth.setdefault(session_id, AnswerDepth())
