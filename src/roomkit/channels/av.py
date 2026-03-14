"""Combined audio+video channel for A/V backends (SIP, RTP)."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from roomkit.channels._video_hooks import VideoHooksMixin
from roomkit.channels.voice import VoiceChannel
from roomkit.models.channel import ChannelCapabilities
from roomkit.models.enums import ChannelMediaType, ChannelType, HookTrigger
from roomkit.video.backends.base import VideoBackend
from roomkit.video.pipeline.engine import VideoPipeline

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from roomkit.models.channel import ChannelBinding
    from roomkit.recorder.base import ChannelRecordingConfig
    from roomkit.video.avatar.base import AvatarProvider
    from roomkit.video.base import VideoSession
    from roomkit.video.pipeline.config import VideoPipelineConfig
    from roomkit.video.video_frame import VideoFrame
    from roomkit.video.vision.base import VisionProvider, VisionResult
    from roomkit.voice.backends.base import VoiceBackend
    from roomkit.voice.base import AudioChunk, VoiceSession
    from roomkit.voice.pipeline.config import AudioPipelineConfig
    from roomkit.voice.stt.base import STTProvider
    from roomkit.voice.tts.base import TTSProvider

logger = logging.getLogger("roomkit.av")


class AudioVideoChannel(VideoHooksMixin, VoiceChannel):
    """Combined audio+video channel for A/V backends.

    Extends :class:`VoiceChannel` with video capabilities from
    :class:`VideoHooksMixin`.  Use this with combined A/V backends
    like :class:`SIPVideoBackend` or :class:`RTPVideoBackend` that
    produce both audio and video through a single transport.

    Audio flows through the normal voice pipeline (STT/TTS/VAD).
    Video frames are delivered to registered taps and optionally
    analysed by a :class:`VisionProvider`.
    """

    channel_type = ChannelType.AUDIO_VIDEO

    def __init__(
        self,
        channel_id: str,
        *,
        stt: STTProvider | None = None,
        tts: TTSProvider | None = None,
        backend: VoiceBackend,
        pipeline: AudioPipelineConfig | None = None,
        video_pipeline: VideoPipelineConfig | None = None,
        vision: VisionProvider | None = None,
        vision_interval_ms: int = 2000,
        avatar: AvatarProvider | None = None,
        recording: ChannelRecordingConfig | None = None,
        **voice_kwargs: Any,
    ) -> None:
        super().__init__(
            channel_id,
            stt=stt,
            tts=tts,
            backend=backend,
            pipeline=pipeline,
            recording=recording,
            **voice_kwargs,
        )
        # Video-specific state
        self._vision = vision
        self._vision_interval_ms = vision_interval_ms
        self._last_vision_results: dict[str, VisionResult] = {}
        self._last_vision_ts: dict[str, float] = {}
        self._video_media_taps: list[Callable[[VideoSession, VideoFrame], None]] = []
        # Dual-signal: track sessions where backend signalled video ready
        # before bind_session completed
        self._session_ready_pending_video: set[str] = set()

        # Video pipeline (decoder, resizer, vision)
        if video_pipeline is not None and (
            video_pipeline.decoder
            or video_pipeline.resizer
            or video_pipeline.transforms
            or video_pipeline.filters
        ):
            self._video_pipeline: VideoPipeline | None = VideoPipeline(video_pipeline)
        else:
            self._video_pipeline = None
        self._video_pipeline_config = video_pipeline

        # Avatar: lip-synced video generation from TTS audio
        self._avatar = avatar

        # Wire video callbacks from the combined A/V backend
        if isinstance(backend, VideoBackend):
            backend.on_video_received(self._on_video_received)

    # -- Video tap API ---------------------------------------------------------

    def add_video_media_tap(self, callback: Callable[[VideoSession, VideoFrame], None]) -> None:
        """Register a tap that receives every video frame (for room recording)."""
        self._video_media_taps.append(callback)

    # -- Backend video callback ------------------------------------------------

    def _on_video_received(self, session: VideoSession, frame: VideoFrame) -> None:
        """Handle a video frame from the A/V backend.

        If a video pipeline is configured, the frame goes through
        decode → resize before reaching taps and vision.
        """
        binding_info = self._session_bindings.get(session.id)
        if binding_info is None:
            logger.debug("Video frame dropped (no binding): session=%s", session.id[:8])
            return

        # Run pipeline stages (decoder, resizer) if configured
        if self._video_pipeline is not None:
            processed = self._video_pipeline.process_inbound(session.id, frame)
            if processed is None:
                return
            frame = processed

        # Deliver to all video taps
        for tap in self._video_media_taps:
            tap(session, frame)

        # Vision: from pipeline config, or direct on channel
        vision = (
            self._video_pipeline_config.vision
            if self._video_pipeline_config is not None and self._video_pipeline_config.vision
            else self._vision
        )
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
        self._schedule(
            self._analyze_frame(session, frame, binding_info[0]),
            name=f"vision:{session.id}",
        )

    # -- Session lifecycle overrides -------------------------------------------

    def bind_session(
        self,
        session: VoiceSession,
        room_id: str,
        binding: ChannelBinding,
        *,
        backend: VoiceBackend | None = None,
    ) -> None:
        """Bind a voice session and fire video session events."""
        super().bind_session(session, room_id, binding, backend=backend)

        # Fire video session started event
        video_session = self._get_video_session(session.id)
        if video_session is None:
            return

        was_ready_pending = session.id in self._session_ready_pending_video
        self._session_ready_pending_video.discard(session.id)

        if self._framework:
            self._schedule(
                self._emit_session_event("video_session_started", video_session, room_id),
                name=f"video_session_started:{session.id}",
            )
        if was_ready_pending and self._framework:
            self._schedule(
                self._fire_session_hook(
                    HookTrigger.ON_VIDEO_SESSION_STARTED, video_session, room_id
                ),
                name=f"video_session_hook:{session.id}",
            )

    def unbind_session(self, session: VoiceSession) -> None:
        """Clean up video state before unbinding the voice session."""
        self._session_ready_pending_video.discard(session.id)
        binding_info = self._session_bindings.get(session.id)

        # The disconnect callback from combined A/V backends passes a
        # VideoSession directly (the backend pops it from its internal
        # map before firing callbacks, so _get_video_session would miss it).
        from roomkit.video.base import VideoSession as _VideoSession

        if isinstance(session, _VideoSession):
            video_session = session
        else:
            video_session = self._get_video_session(session.id)
        if video_session is not None and binding_info is not None:
            room_id = binding_info[0]
            if self._framework:
                self._schedule(
                    self._emit_session_event("video_session_ended", video_session, room_id),
                    name=f"video_session_ended:{session.id}",
                )
                self._schedule(
                    self._fire_session_hook(
                        HookTrigger.ON_VIDEO_SESSION_ENDED, video_session, room_id
                    ),
                    name=f"video_session_ended_hook:{session.id}",
                )

        self._last_vision_results.pop(session.id, None)
        self._last_vision_ts.pop(session.id, None)
        super().unbind_session(session)

    async def close(self) -> None:
        """Close avatar, vision, and video state, then delegate to parent."""
        if self._avatar:
            await self._avatar.close()
        if self._vision:
            await self._vision.close()
        self._last_vision_results.clear()
        self._last_vision_ts.clear()
        self._video_media_taps.clear()
        self._session_ready_pending_video.clear()
        await super().close()

    # -- Avatar: TTS audio → lip-synced video frames ---------------------------

    async def _wrap_outbound(
        self,
        session: VoiceSession,
        chunks: AsyncIterator[AudioChunk],
    ) -> AsyncIterator[AudioChunk]:
        """Wrap TTS outbound stream to feed avatar with audio chunks.

        Tees each audio chunk: one copy goes to the voice backend
        (speaker), the other feeds the avatar which generates
        lip-synced video frames sent to the video backend.
        """
        # If no avatar, delegate to parent (pipeline outbound processing)
        if self._avatar is None or not self._avatar.is_started:
            async for chunk in super()._wrap_outbound(session, chunks):
                yield chunk
            return

        async for chunk in super()._wrap_outbound(session, chunks):
            # Feed audio to avatar → get video frames
            if chunk.data:
                frames = self._avatar.feed_audio(chunk.data, chunk.sample_rate)
                for frame in frames:
                    # Send through video pipeline if configured
                    if self._video_pipeline is not None:
                        processed = self._video_pipeline.process_inbound(
                            session.id,
                            frame,
                        )
                        if processed is not None:
                            frame = processed
                    # Send to video taps
                    for tap in self._video_media_taps:
                        tap(session, frame)  # type: ignore[arg-type]
                    # Send to video backend
                    if isinstance(self._backend, VideoBackend):
                        await self._backend.send_video(session, frame)
            yield chunk

        # Flush avatar (mouth closing animation)
        for frame in self._avatar.flush():
            if isinstance(self._backend, VideoBackend):
                await self._backend.send_video(session, frame)

    # -- Capabilities ----------------------------------------------------------

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            media_types=[ChannelMediaType.AUDIO, ChannelMediaType.VIDEO, ChannelMediaType.TEXT],
            supports_audio=True,
            supports_video=True,
            supported_audio_formats=["wav", "mp3", "ogg", "webm"],
            max_audio_duration_seconds=3600,
        )

    @property
    def info(self) -> dict[str, Any]:
        parent_info = super().info
        parent_info["vision"] = self._vision.name if self._vision else None
        parent_info["vision_interval_ms"] = self._vision_interval_ms
        parent_info["avatar"] = self._avatar.name if self._avatar else None
        return parent_info

    # -- Video session lookup --------------------------------------------------

    def _get_video_session(self, session_id: str) -> VideoSession | None:
        """Look up the VideoSession from the combined backend."""
        if isinstance(self._backend, VideoBackend):
            return self._backend.get_video_session(session_id)
        return None

    # -- Public helpers --------------------------------------------------------

    @property
    def vision(self) -> VisionProvider | None:
        """The vision provider (if configured)."""
        return self._vision

    def get_last_vision_result(self, session_id: str) -> VisionResult | None:
        """Get the most recent vision analysis for a session."""
        return self._last_vision_results.get(session_id)
