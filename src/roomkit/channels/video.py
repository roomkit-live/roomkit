"""Video channel for real-time video communication."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any

from roomkit.channels._video_hooks import VideoHooksMixin
from roomkit.channels.base import Channel
from roomkit.models.channel import ChannelCapabilities
from roomkit.models.enums import (
    ChannelCategory,
    ChannelDirection,
    ChannelMediaType,
    ChannelType,
    HookTrigger,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from roomkit.core.framework import RoomKit
    from roomkit.models.channel import ChannelBinding, ChannelOutput
    from roomkit.models.context import RoomContext
    from roomkit.models.delivery import InboundMessage
    from roomkit.models.event import RoomEvent
    from roomkit.recorder.base import ChannelRecordingConfig
    from roomkit.video.backends.base import VideoBackend
    from roomkit.video.base import VideoSession
    from roomkit.video.pipeline.config import VideoPipelineConfig
    from roomkit.video.recorder.base import (
        VideoRecorder,
        VideoRecordingConfig,
        VideoRecordingHandle,
    )
    from roomkit.video.video_frame import VideoFrame
    from roomkit.video.vision.base import VisionProvider, VisionResult

logger = logging.getLogger("roomkit.video")


class VideoChannel(VideoHooksMixin, Channel):
    """Real-time video communication channel.

    Wires a :class:`VideoBackend` into the RoomKit framework,
    managing session lifecycle, hook triggers, and optional
    :class:`VisionProvider` for periodic frame analysis.
    """

    channel_type = ChannelType.VIDEO
    category = ChannelCategory.TRANSPORT
    direction = ChannelDirection.BIDIRECTIONAL

    def __init__(
        self,
        channel_id: str,
        *,
        backend: VideoBackend,
        vision: VisionProvider | None = None,
        vision_interval_ms: int = 2000,
        pipeline: VideoPipelineConfig | None = None,
        recording: ChannelRecordingConfig | None = None,
    ) -> None:
        super().__init__(channel_id)
        self._backend = backend
        self._vision = vision
        self._vision_interval_ms = vision_interval_ms
        self._pipeline = pipeline
        self._recording = recording

        # Create video pipeline engine if config has processing stages
        from roomkit.video.pipeline.engine import VideoPipeline

        if pipeline is not None and (pipeline.decoder or pipeline.resizer):
            self._video_pipeline: VideoPipeline | None = VideoPipeline(pipeline)
        else:
            self._video_pipeline = None
        self._framework: RoomKit | None = None

        # Resolve recorder from pipeline
        self._recorder: VideoRecorder | None = pipeline.recorder if pipeline else None
        self._recording_config: VideoRecordingConfig | None = (
            pipeline.recording_config if pipeline else None
        )

        # Map session_id -> (room_id, binding) for routing
        self._session_bindings: dict[str, tuple[str, ChannelBinding]] = {}
        # Track scheduled fire-and-forget tasks for clean shutdown
        self._scheduled_tasks: set[asyncio.Task[Any]] = set()
        # Last vision result per session (for AI context injection)
        self._last_vision_results: dict[str, VisionResult] = {}
        # Timestamp of last vision analysis per session
        self._last_vision_ts: dict[str, float] = {}
        # Sessions where backend signalled ready before bind_session ran
        self._session_ready_pending: set[str] = set()
        # Cached event loop for cross-thread scheduling
        self._event_loop: asyncio.AbstractEventLoop | None = None
        # Active recording handles per session
        self._recording_handles: dict[str, VideoRecordingHandle] = {}
        # Room-level media recording taps
        self._media_taps: list[Callable[[VideoSession, VideoFrame], None]] = []

        # Wire backend callbacks
        backend.on_video_received(self._on_video_received)
        backend.on_session_ready(self._on_session_ready)
        backend.on_client_disconnected(self._on_backend_disconnected)

    # -- Framework integration --

    def set_framework(self, framework: RoomKit) -> None:
        """Set the framework reference for hook firing and event routing.

        Called automatically when the channel is registered with RoomKit.
        """
        self._framework = framework

    @property
    def provider_name(self) -> str | None:
        return self._backend.name

    @property
    def info(self) -> dict[str, Any]:
        return {
            "backend": self._backend.name,
            "vision": self._vision.name if self._vision else None,
            "vision_interval_ms": self._vision_interval_ms,
        }

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            media_types=[ChannelMediaType.VIDEO],
            supports_video=True,
        )

    def resolve_trace_room(self, session_id: str | None) -> str | None:
        """Resolve room_id from video session bindings."""
        if session_id is None:
            return None
        binding_info = self._session_bindings.get(session_id)
        return binding_info[0] if binding_info else None

    def add_media_tap(self, callback: Callable[[VideoSession, VideoFrame], None]) -> None:
        """Register a tap that receives every video frame (for room recording)."""
        self._media_taps.append(callback)

    # -------------------------------------------------------------------------
    # Task scheduling
    def _schedule(self, coro: Coroutine[Any, Any, Any], *, name: str) -> None:
        """Schedule *coro* as a fire-and-forget task.

        Works from both the event-loop thread and foreign threads
        (e.g. backend capture callbacks dispatched via call_soon_threadsafe).
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Foreign thread — dispatch via cached event loop
            cached = self._event_loop
            if cached is not None and cached.is_running():
                cached.call_soon_threadsafe(self._create_task, coro, name)
            else:
                coro.close()
            return
        self._event_loop = loop
        self._create_task(coro, name)

    def _create_task(self, coro: Coroutine[Any, Any, Any], name: str) -> None:
        """Create and track an asyncio task (must run on the event loop thread)."""
        task = asyncio.get_running_loop().create_task(coro, name=name)
        task.add_done_callback(self._task_done)
        self._scheduled_tasks.add(task)

    def _task_done(self, task: asyncio.Task[Any]) -> None:
        self._scheduled_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "Unhandled exception in scheduled task %s: %s",
                task.get_name(),
                exc,
                exc_info=exc,
            )

    # -------------------------------------------------------------------------
    # Session lifecycle
    def bind_session(
        self,
        session: VideoSession,
        room_id: str,
        binding: ChannelBinding,
    ) -> None:
        """Bind a video session to a room for event routing."""
        self._session_bindings[session.id] = (room_id, binding)
        # Dual-signal: check if backend already signalled ready
        was_ready_pending = session.id in self._session_ready_pending
        self._session_ready_pending.discard(session.id)
        # Emit framework event
        if self._framework:
            self._schedule(
                self._emit_session_event("video_session_started", session, room_id),
                name=f"video_session_started:{session.id}",
            )
        # Start recording if configured
        if self._recorder and self._recording_config:
            handle = self._recorder.start(session, self._recording_config)
            self._recording_handles[session.id] = handle
            logger.info("Recording started: %s for session %s", handle.id, session.id[:8])
        # Fire hook if backend was already ready
        if was_ready_pending and self._framework:
            self._schedule(
                self._fire_session_hook(HookTrigger.ON_VIDEO_SESSION_STARTED, session, room_id),
                name=f"video_session_hook:{session.id}",
            )

    def unbind_session(self, session: VideoSession) -> None:
        """Remove session binding and clean up state."""
        self._session_ready_pending.discard(session.id)
        # Stop recording
        handle = self._recording_handles.pop(session.id, None)
        if handle and self._recorder:
            result = self._recorder.stop(handle)
            logger.info(
                "Recording stopped: %s (%d frames, %.1fs)",
                result.id,
                result.frame_count,
                result.duration_seconds,
            )
        binding_info = self._session_bindings.pop(session.id, None)
        self._last_vision_results.pop(session.id, None)
        self._last_vision_ts.pop(session.id, None)
        if binding_info and self._framework:
            room_id = binding_info[0]
            self._schedule(
                self._emit_session_event("video_session_ended", session, room_id),
                name=f"video_session_ended:{session.id}",
            )
            self._schedule(
                self._fire_session_hook(HookTrigger.ON_VIDEO_SESSION_ENDED, session, room_id),
                name=f"video_session_ended_hook:{session.id}",
            )

    async def connect_session(
        self,
        session: Any,
        room_id: str,
        binding: ChannelBinding,
    ) -> None:
        """Accept a video session via process_inbound."""
        self.bind_session(session, room_id, binding)

    async def disconnect_session(self, session: Any, room_id: str) -> None:
        """Clean up a video session on remote disconnect."""
        self.unbind_session(session)
        await self._backend.disconnect(session)

    def update_binding(self, room_id: str, binding: ChannelBinding) -> None:
        """Update cached bindings for all sessions in a room."""
        for sid, (rid, _old) in self._session_bindings.items():
            if rid == room_id:
                self._session_bindings[sid] = (rid, binding)

    # -------------------------------------------------------------------------
    # Backend callbacks
    def _on_video_received(self, session: VideoSession, frame: VideoFrame) -> None:
        """Handle a video frame from the backend.

        If a pipeline is configured, the frame goes through decode → resize
        before reaching taps and vision.  Otherwise it's passed as-is.
        """
        binding_info = self._session_bindings.get(session.id)
        if binding_info is None:
            return

        # Run pipeline stages (decoder, resizer) if configured
        if self._video_pipeline is not None:
            processed = self._video_pipeline.process_inbound(session.id, frame)
            if processed is None:
                return  # frame dropped (e.g., decoder needs keyframe)
            frame = processed

        # Recorder tap runs on every frame
        rec_handle = self._recording_handles.get(session.id)
        if rec_handle and self._recorder:
            self._recorder.tap_frame(rec_handle, frame)
        # Room-level media taps
        for tap in self._media_taps:
            tap(session, frame)

        # Vision: from pipeline config, or direct on channel
        vision = (
            self._video_pipeline.config.vision
            if self._video_pipeline is not None
            else self._vision
        )
        if vision is None:
            return
        # Throttle before creating a task — avoids O(fps) task allocation
        if frame.timestamp_ms is not None:
            now_ms = frame.timestamp_ms
        else:
            now_ms = time.monotonic() * 1000.0
        last_ts = self._last_vision_ts.get(session.id, -float("inf"))
        if now_ms - last_ts < self._vision_interval_ms:
            return
        self._last_vision_ts[session.id] = now_ms
        self._schedule(
            self._analyze_frame(session, frame, binding_info[0]),
            name=f"vision:{session.id}",
        )

    def _on_session_ready(self, session: VideoSession) -> None:
        """Backend signals the video path is live."""
        binding_info = self._session_bindings.get(session.id)
        if binding_info is None:
            # Session not yet bound — record for dual-signal
            self._session_ready_pending.add(session.id)
            return
        room_id = binding_info[0]
        if self._framework:
            self._schedule(
                self._fire_session_hook(HookTrigger.ON_VIDEO_SESSION_STARTED, session, room_id),
                name=f"video_session_hook:{session.id}",
            )

    def _on_backend_disconnected(self, session: VideoSession) -> None:
        """Backend signals the client disconnected."""
        self.unbind_session(session)

    # -- Channel ABC implementation --
    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        from roomkit.models.event import EventSource
        from roomkit.models.event import RoomEvent as RoomEventModel

        return RoomEventModel(
            room_id=context.room.id,
            source=EventSource(
                channel_id=self.channel_id,
                channel_type=self.channel_type,
                participant_id=message.sender_id,
                external_id=message.external_id,
                provider=self.provider_name,
            ),
            content=message.content,
            idempotency_key=message.idempotency_key,
            metadata=message.metadata,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        from roomkit.models.channel import ChannelOutput as ChannelOutputModel

        # Video channel does not deliver text events — it handles
        # video frames via the backend, not the standard deliver path.
        # System events and internal-visibility events are skipped.
        return ChannelOutputModel.empty()

    async def close(self) -> None:
        # Cancel scheduled tasks
        for task in self._scheduled_tasks:
            task.cancel()
        if self._scheduled_tasks:
            await asyncio.gather(*self._scheduled_tasks, return_exceptions=True)
        self._scheduled_tasks.clear()
        # Stop active recordings
        if self._recorder:
            for handle in self._recording_handles.values():
                result = self._recorder.stop(handle)
                logger.info(
                    "Recording stopped (close): %s (%d frames, %.1fs)",
                    result.id,
                    result.frame_count,
                    result.duration_seconds,
                )
            self._recorder.close()
        self._recording_handles.clear()
        # Close vision provider
        if self._vision:
            await self._vision.close()
        # Close backend last
        await self._backend.close()
        self._session_bindings.clear()
        self._last_vision_results.clear()
        self._last_vision_ts.clear()
        self._session_ready_pending.clear()

    # -------------------------------------------------------------------------
    # Public helpers
    @property
    def backend(self) -> VideoBackend:
        """The video backend."""
        return self._backend

    @property
    def vision(self) -> VisionProvider | None:
        """The vision provider (if configured)."""
        return self._vision

    def get_last_vision_result(self, session_id: str) -> VisionResult | None:
        """Get the most recent vision analysis for a session."""
        return self._last_vision_results.get(session_id)
