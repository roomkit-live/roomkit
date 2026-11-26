"""RealtimeAudioVideoChannel — realtime speech-to-speech AI with video output."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from roomkit import _clock
from roomkit.channels._realtime_endings import cut_tasks
from roomkit.channels._video_hooks import VideoHooksMixin
from roomkit.channels._video_resources import _VideoResources
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.core.task_utils import CLOSE_WAIT_S
from roomkit.models.channel import ChannelCapabilities
from roomkit.models.enums import ChannelMediaType, ChannelType, HookTrigger
from roomkit.voice.realtime.provider import RealtimeAudioVideoProvider

if TYPE_CHECKING:
    from collections.abc import Callable

    from roomkit.video.base import VideoSession
    from roomkit.video.pipeline.config import VideoPipelineConfig
    from roomkit.video.video_frame import VideoFrame
    from roomkit.video.vision.base import VisionProvider, VisionResult
    from roomkit.voice.backends.base import VoiceBackend
    from roomkit.voice.base import VoiceSession
    from roomkit.voice.realtime.provider import RealtimeVoiceProvider

logger = logging.getLogger("roomkit.channels.realtime_av")


class RealtimeAudioVideoChannel(VideoHooksMixin, RealtimeVoiceChannel):
    """Realtime audio+video channel for providers that deliver both modalities.

    Extends :class:`RealtimeVoiceChannel` with video capabilities from
    :class:`VideoHooksMixin`.  Use this with providers like
    :class:`AnamRealtimeProvider` that produce synchronized audio and
    video from a cloud avatar pipeline.

    Audio flows through the normal realtime voice path (provider ↔ transport).
    Video frames from the provider are delivered to registered taps and
    optionally analysed by a :class:`VisionProvider`.

    Example::

        from roomkit.providers.anam import AnamConfig, AnamRealtimeProvider

        provider = AnamRealtimeProvider(AnamConfig(api_key="ak-..."))
        channel = RealtimeAudioVideoChannel(
            "avatar-1",
            provider=provider,
            transport=ws_transport,
            system_prompt="You are a helpful avatar.",
        )
        kit.register_channel(channel)
    """

    channel_type = ChannelType.REALTIME_AUDIO_VIDEO

    def __init__(
        self,
        channel_id: str,
        *,
        provider: RealtimeVoiceProvider,
        transport: VoiceBackend,
        video_pipeline: VideoPipelineConfig | None = None,
        vision: VisionProvider | None = None,
        vision_interval_ms: int = 2000,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            channel_id,
            provider=provider,
            transport=transport,
            **kwargs,
        )
        # Video-specific state
        self._vision = vision
        self._vision_interval_ms = vision_interval_ms
        self._last_vision_results: dict[str, VisionResult] = {}
        self._last_vision_ts: dict[str, float] = {}
        self._room_vision: dict[str, tuple[float, str, VisionResult]] = {}
        self._video_media_taps: list[Callable[[VideoSession, VideoFrame], None]] = []
        self._video_pipeline_config = video_pipeline

        # Video pipeline (decoder, resizer, transforms, filters, vision), owned
        # as the other video channels own theirs: each vision closed once.
        self._video_resources = _VideoResources(video_pipeline, vision)
        self._video_pipeline = self._video_resources.pipeline

        # Wire video callback from audio+video provider
        if isinstance(provider, RealtimeAudioVideoProvider):
            provider.on_video(self._on_provider_video)

    # -- Video tap API ---------------------------------------------------------

    def add_video_media_tap(self, callback: Callable[[VideoSession, VideoFrame], None]) -> None:
        """Register a tap that receives every video frame (for room recording)."""
        self._video_media_taps.append(callback)

    # -- Provider video callback -----------------------------------------------

    def _on_provider_video(self, session: VoiceSession, frame: VideoFrame) -> None:
        """Handle a video frame from the realtime A/V provider."""
        with self._state_lock:
            room_id = self._session_rooms.get(session.id)
            # A session whose teardown began takes no more video: its state is
            # reset before the provider disconnects, and a frame arriving
            # meanwhile would build it again, with no one left to reset it.
            ending = session.id in self._session_teardowns
        if room_id is None or ending:
            return

        # Run pipeline stages (decoder, resizer, transforms, filters) if configured
        if self._video_pipeline is not None:
            processed = self._video_pipeline.process_inbound(session.id, frame)
            if processed is None:
                return
            frame = processed
            self._dispatch_detections(session, room_id)

        # Deliver to all video taps (cast session as VideoSession-like)
        for tap in self._video_media_taps:
            tap(session, frame)  # ty: ignore[invalid-argument-type]

        # Vision analysis (throttled)
        vision = self._vision_provider
        if vision is None:
            return
        if frame.timestamp_ms is not None:
            now_ms = frame.timestamp_ms
        else:
            now_ms = _clock.monotonic() * 1000.0
        last_ts = self._last_vision_ts.get(session.id, -float("inf"))
        if now_ms - last_ts < self._vision_interval_ms:
            return
        self._last_vision_ts[session.id] = now_ms

        loop = self._event_loop
        if loop is not None and loop.is_running():
            self._track_task(
                loop,
                self._analyze_frame(session, frame, room_id),  # ty: ignore[invalid-argument-type]
                name=f"vision:{session.id}",
            )

    def _dispatch_detections(self, session: VoiceSession, room_id: str) -> None:
        """Fire ``ON_VIDEO_DETECTION`` for each event the filters emitted on a frame.

        Drained on every frame, hooks or not: undrained, the session's events
        pile up for as long as a filter keeps detecting.
        """
        assert self._video_pipeline is not None
        events = self._video_pipeline.drain_events(session.id)
        loop = self._event_loop
        if not events or loop is None or not loop.is_running():
            return
        for event in events:
            if event.data is None:
                continue
            event.data.session = session
            self._track_task(
                loop,
                self._fire_video_detection_hook(
                    session,  # ty: ignore[invalid-argument-type]
                    event.data,
                    room_id,
                ),
                name=f"detection:{session.id}",
            )

    # -- Session lifecycle overrides -------------------------------------------

    async def start_session(
        self,
        room_id: str,
        participant_id: str,
        connection: Any,
        *,
        metadata: dict[str, Any] | None = None,
        organization_id: str | None = None,
    ) -> VoiceSession:
        """Start a realtime A/V session, then fire video hooks.

        *organization_id* scopes the room read, as on the voice channel.
        """
        session = await super().start_session(
            room_id,
            participant_id,
            connection,
            metadata=metadata,
            organization_id=organization_id,
        )

        # Cache event loop for cross-thread video callback scheduling
        self._event_loop = asyncio.get_running_loop()

        # Fire video session events
        if self._framework:
            await self._emit_session_event(
                "video_session_started",
                session,
                room_id,
            )
            await self._fire_session_hook(
                HookTrigger.ON_VIDEO_SESSION_STARTED,
                session,
                room_id,
            )

        return session

    async def _before_session_teardown(self, session: VoiceSession) -> None:
        """Fire video hooks and clear video state; the base teardown follows.

        Rides the base channel's one-teardown-per-session arbitration, so a
        remote hangup that reaches the channel twice fires the video events
        once.
        """
        with self._state_lock:
            room_id = self._session_rooms.get(session.id, session.room_id)

        # Fire video session ended events before teardown
        if self._framework:
            await self._emit_session_event(
                "video_session_ended",
                session,
                room_id,
            )
            await self._fire_session_hook(
                HookTrigger.ON_VIDEO_SESSION_ENDED,
                session,
                room_id,
            )

        # Clean up video state, once no analysis is left to write it again
        await self._cancel_vision(session.id)
        if self._video_pipeline is not None:
            self._video_pipeline.reset(session.id)
        self._last_vision_results.pop(session.id, None)
        self._forget_session_vision(session.id)
        self._last_vision_ts.pop(session.id, None)

    async def _cancel_vision(self, session_id: str) -> None:
        """Cancel the session's vision analyses still under way.

        One finishing while the session ends (its provider disconnecting) would
        put it back in the pipeline and leave its view to the room after the
        reset (RFC §12.8.7).
        """
        name = f"vision:{session_id}"
        tasks = cut_tasks(task for task in self._scheduled_tasks if task.get_name() == name)
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=CLOSE_WAIT_S)
        if pending:
            logger.warning("Timed out cancelling vision for session %s", session_id)

    async def close(self) -> None:
        """End the sessions (parent), then close the video pipeline and vision.

        The pipeline outlives the sessions: each one's teardown resets its
        state in it, and the analyses still under way are cancelled first.
        """
        try:
            await super().close()
        finally:
            await self._video_resources.close()
            self._last_vision_results.clear()
            self._last_vision_ts.clear()
            self._room_vision.clear()
            self._video_media_taps.clear()

    # -- Capabilities ----------------------------------------------------------

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            media_types=[ChannelMediaType.AUDIO, ChannelMediaType.VIDEO, ChannelMediaType.TEXT],
            supports_audio=True,
            supports_video=True,
            custom={"realtime": True, "server_vad": True},
        )

    @property
    def info(self) -> dict[str, Any]:
        parent_info = super().info
        parent_info["vision"] = self._vision.name if self._vision else None
        parent_info["vision_interval_ms"] = self._vision_interval_ms
        return parent_info

    # -- Public helpers --------------------------------------------------------

    @property
    def vision(self) -> VisionProvider | None:
        """The vision provider (if configured)."""
        return self._vision

    def get_last_vision_result(self, session_id: str) -> VisionResult | None:
        """Get the most recent vision analysis for a session."""
        return self._last_vision_results.get(session_id)
