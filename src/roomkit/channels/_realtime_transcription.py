"""Transcription processing for RealtimeVoiceChannel."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.models.enums import HookTrigger
from roomkit.models.event import TextContent

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
        _emit_transcription_events: Whether to emit transcriptions as RoomEvents.
        _framework: The RoomKit framework instance (or None).
        channel_id: Channel identifier.
        provider_name: Name of the realtime voice provider.

    Cross-mixin methods (implemented elsewhere in the MRO):
        _track_task: Schedule an async task with exception handling.
        _send_client_message: Send a JSON message to the client UI.
        _rt_span_ctx: Get the telemetry span context for a session.
    """

    _state_lock: threading.Lock
    _session_rooms: dict[str, str]
    _barge_in_active: set[str]
    _last_assistant_text: dict[str, str]
    _emit_transcription_events: bool
    _framework: RoomKit | None
    channel_id: str
    provider_name: str | None

    def _track_task(self, loop: Any, coro: Any, *, name: str) -> Any: ...

    async def _send_client_message(self, session: Any, message: dict[str, Any]) -> None: ...

    def _rt_span_ctx(self, session_id: str) -> tuple[Any, Any]: ...


class RealtimeTranscriptionMixin:
    """Transcription hooks and RoomEvent emission for RealtimeVoiceChannel.

    Host contract: :class:`RealtimeTranscriptionHost`.
    """

    _state_lock: threading.Lock
    _session_rooms: dict[str, str]
    _barge_in_active: set[str]
    _last_assistant_text: dict[str, str]
    _emit_transcription_events: bool
    _framework: RoomKit | None
    channel_id: str
    provider_name: str | None

    _track_task: Any  # see RealtimeTranscriptionHost — cross-mixin
    _send_client_message: Any  # see RealtimeTranscriptionHost — cross-mixin
    _rt_span_ctx: Any  # see RealtimeTranscriptionHost — cross-mixin

    def _on_provider_transcription(
        self, session: VoiceSession, text: str, role: str, is_final: bool
    ) -> Any:
        """Handle transcription from provider."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._track_task(
            loop,
            self._process_transcription(session, text, role, is_final),
            name=f"rt_transcription:{session.id}",
        )

    async def _process_transcription(
        self, session: VoiceSession, text: str, role: str, is_final: bool
    ) -> None:
        """Process a transcription: fire hooks, emit event, send to client.

        Partial transcriptions skip hooks and telemetry spans — they are
        forwarded to the client UI only.  Final transcriptions go through
        the full pipeline: hooks, client UI, and RoomEvent emission.
        """
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

            # Emit final transcriptions as RoomEvents
            if self._emit_transcription_events and final_text.strip():
                participant_id = session.participant_id if role == "user" else None
                logger.info(
                    "Emitting transcription as RoomEvent: role=%s, text=%s",
                    role,
                    final_text,
                )
                await self._framework.send_event(
                    room_id,
                    self.channel_id,
                    TextContent(body=final_text),
                    participant_id=participant_id,
                    metadata={
                        "voice_session_id": session.id,
                        "source": "realtime_voice",
                        "role": role,
                    },
                    provider=self.provider_name,
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
