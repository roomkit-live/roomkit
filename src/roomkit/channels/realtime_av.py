"""RealtimeAudioVideoChannel — realtime speech-to-speech AI with video output."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from roomkit.channels._video_hooks import VideoHooksMixin
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
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


class RealtimeAudioVideoChannel(VideoHooksMixin, RealtimeVoiceChannel):  # type: ignore[misc]
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
        self._video_media_taps: list[Callable[[VideoSession, VideoFrame], None]] = []
        self._video_pipeline_config = video_pipeline

        # Build video pipeline if configured
        if video_pipeline is not None and (
            video_pipeline.decoder
            or video_pipeline.resizer
            or video_pipeline.transforms
            or video_pipeline.filters
        ):
            from roomkit.video.pipeline.engine import VideoPipeline

            self._video_pipeline: Any = VideoPipeline(video_pipeline)
        else:
            self._video_pipeline = None

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
        if room_id is None:
            return

        # Run pipeline stages (decoder, resizer) if configured
        if self._video_pipeline is not None:
            processed = self._video_pipeline.process_inbound(session.id, frame)
            if processed is None:
                return
            frame = processed

        # Deliver to all video taps (cast session as VideoSession-like)
        for tap in self._video_media_taps:
            tap(session, frame)  # type: ignore[arg-type]

        # Vision analysis (throttled)
        vision = self._vision_provider
        if vision is None:
            return
        if frame.timestamp_ms is not None:
            now_ms = frame.timestamp_ms
        else:
            now_ms = time.monotonic() * 1000.0
        last_ts = self._last_vision_ts.get(session.id, -float("inf"))
        if now_ms - last_ts < self._vision_interval_ms:
            return
        self._last_vision_ts[session.id] = now_ms

        loop = self._event_loop
        if loop is not None and loop.is_running():
            self._track_task(
                loop,
                self._analyze_frame(session, frame, room_id),  # type: ignore[arg-type]
                name=f"vision:{session.id}",
            )

    # -- Session lifecycle overrides -------------------------------------------

    async def start_session(
        self,
        room_id: str,
        participant_id: str,
        connection: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> VoiceSession:
        """Start a realtime A/V session, then fire video hooks."""
        session = await super().start_session(
            room_id,
            participant_id,
            connection,
            metadata=metadata,
        )

        # Cache event loop for cross-thread video callback scheduling
        import asyncio

        self._event_loop = asyncio.get_running_loop()

        # Fire video session events
        if self._framework:
            await self._emit_session_event(
                "video_session_started",
                session,
                room_id,  # type: ignore[arg-type]
            )
            await self._fire_session_hook(
                HookTrigger.ON_VIDEO_SESSION_STARTED,
                session,
                room_id,  # type: ignore[arg-type]
            )

        return session

    async def end_session(self, session: VoiceSession) -> None:
        """Fire video hooks, then clean up the realtime session."""
        with self._state_lock:
            room_id = self._session_rooms.get(session.id, session.room_id)

        # Fire video session ended events before teardown
        if self._framework:
            await self._emit_session_event(
                "video_session_ended",
                session,
                room_id,  # type: ignore[arg-type]
            )
            await self._fire_session_hook(
                HookTrigger.ON_VIDEO_SESSION_ENDED,
                session,
                room_id,  # type: ignore[arg-type]
            )

        # Clean up video state
        self._last_vision_results.pop(session.id, None)
        self._last_vision_ts.pop(session.id, None)

        await super().end_session(session)

    async def close(self) -> None:
        """Close vision, video state, then delegate to parent."""
        if self._vision:
            await self._vision.close()
        self._last_vision_results.clear()
        self._last_vision_ts.clear()
        self._video_media_taps.clear()
        await super().close()

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
