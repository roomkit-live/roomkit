"""Voice channel mixin — announce the pipeline's recordings (RFC §17.6).

The audio pipeline starts a session's recording when the session becomes
active and stops it when the session ends. This mixin turns those two pipeline
callbacks into ``ON_RECORDING_STARTED`` / ``ON_RECORDING_STOPPED`` and their
framework events. ``VoiceChannel`` and ``RealtimeVoiceChannel`` record through
the same pipeline, so they announce through this one implementation; each
supplies only how it finds a session's room, its span and its task scheduler.
"""

from __future__ import annotations

import logging
from collections.abc import Coroutine, Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.channels._room_hooks import RoomHooks
from roomkit.models.enums import HookTrigger
from roomkit.telemetry.context import reset_span, set_current_span
from roomkit.voice.events import RecordingStartedEvent, RecordingStoppedEvent

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.voice.base import VoiceSession
    from roomkit.voice.pipeline.engine import AudioPipeline

logger = logging.getLogger("roomkit.voice")


@runtime_checkable
class RecordingHooksHost(Protocol):
    """Contract: what a host class provides for VoiceRecordingHooksMixin.

    Attributes:
        channel_id: Unique identifier for this channel instance.
        _framework: Reference to the RoomKit orchestrator (``None`` until
            the channel is registered).

    Methods:
        _recording_room: The room a session's recording reports to, or
            ``None`` when the session is no longer known. Called for the stop
            too, so it must still answer while the session is ending.
        _recording_span: The telemetry span to parent the hooks under.
        _schedule_recording_hook: Run a hook coroutine as a tracked task.
    """

    channel_id: str
    _framework: RoomKit | None

    def _recording_room(self, session: VoiceSession) -> str | None: ...

    def _recording_span(self, session: VoiceSession) -> str | None: ...

    def _schedule_recording_hook(self, coro: Coroutine[Any, Any, Any], *, name: str) -> None: ...


class VoiceRecordingHooksMixin:
    """Fire the recording hooks for a voice channel's pipeline recorder.

    Host contract: :class:`RecordingHooksHost`.
    """

    channel_id: str
    _framework: RoomKit | None

    def _recording_room(self, session: VoiceSession) -> str | None: ...

    def _recording_span(self, session: VoiceSession) -> str | None: ...

    def _schedule_recording_hook(self, coro: Coroutine[Any, Any, Any], *, name: str) -> None: ...

    def _wire_recording_hooks(self, pipeline: AudioPipeline) -> None:
        """Subscribe to *pipeline*'s recording start and stop."""
        pipeline.on_recording_started(self._on_pipeline_recording_started)
        pipeline.on_recording_stopped(self._on_pipeline_recording_stopped)

    def _on_pipeline_recording_started(self, session: VoiceSession, handle: Any) -> None:
        room_id = self._recording_room(session)
        if room_id is None or not self._framework:
            return
        self._schedule_recording_hook(
            self._fire_recording_started_hook(session, handle, room_id),
            name=f"recording_started:{session.id}",
        )

    def _on_pipeline_recording_stopped(self, session: VoiceSession, result: Any) -> None:
        room_id = self._recording_room(session)
        if room_id is None or not self._framework:
            return
        self._schedule_recording_hook(
            self._fire_recording_stopped_hook(session, result, room_id),
            name=f"recording_stopped:{session.id}",
        )

    @contextmanager
    def _recording_span_ctx(self, session: VoiceSession) -> Generator[None, None, None]:
        parent = self._recording_span(session)
        token = set_current_span(parent) if parent else None
        try:
            yield
        finally:
            if token is not None:
                reset_span(token)

    async def _fire_recording_started_hook(
        self, session: VoiceSession, handle: Any, room_id: str
    ) -> None:
        if not self._framework:
            return
        try:
            with self._recording_span_ctx(session):
                event = RecordingStartedEvent(session=session, id=handle.id, room_id=room_id)
                await RoomHooks(self._framework, room_id).fire(
                    HookTrigger.ON_RECORDING_STARTED, event
                )
                await self._emit_recording_started(session, handle.id, room_id)
        except Exception:
            logger.exception("Error firing ON_RECORDING_STARTED hook")

    async def _fire_recording_stopped_hook(
        self, session: VoiceSession, result: Any, room_id: str
    ) -> None:
        if not self._framework:
            return
        try:
            with self._recording_span_ctx(session):
                event = RecordingStoppedEvent(
                    session=session,
                    id=result.id,
                    urls=tuple(result.urls),
                    duration_seconds=result.duration_seconds,
                    room_id=room_id,
                )

                async def reported() -> None:
                    await self._emit_recording_stopped(
                        session, result.id, room_id, duration_seconds=result.duration_seconds
                    )

                # The framework event once the hooks have ended, the stop not
                # waiting for them (RFC §9.3).
                await RoomHooks(self._framework, room_id).fire(
                    HookTrigger.ON_RECORDING_STOPPED, event, then=reported
                )
        except Exception:
            logger.exception("Error firing ON_RECORDING_STOPPED hook")

    async def _emit_recording_started(
        self, session: VoiceSession, recording_id: str, room_id: str
    ) -> None:
        if not self._framework:
            return
        try:
            await self._framework._emit_framework_event(
                "recording_started",
                room_id=room_id,
                data={"session_id": session.id, "id": recording_id},
            )
        except Exception:
            logger.exception("Error emitting recording_started")

    async def _emit_recording_stopped(
        self,
        session: VoiceSession,
        recording_id: str,
        room_id: str,
        *,
        duration_seconds: float = 0.0,
    ) -> None:
        if not self._framework:
            return
        try:
            await self._framework._emit_framework_event(
                "recording_stopped",
                room_id=room_id,
                data={
                    "session_id": session.id,
                    "id": recording_id,
                    "duration_seconds": duration_seconds,
                },
            )
        except Exception:
            logger.exception("Error emitting recording_stopped")
