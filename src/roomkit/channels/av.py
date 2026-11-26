"""Combined audio+video channel for A/V backends (SIP, RTP)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import queue as _queue_mod
import threading
import time
from typing import TYPE_CHECKING, Any

from roomkit.channels._video_hooks import VideoHooksMixin
from roomkit.channels._video_resources import _VideoResources
from roomkit.channels.voice import VoiceChannel
from roomkit.core.task_utils import cancel_and_wait
from roomkit.models.channel import ChannelCapabilities
from roomkit.models.enums import ChannelMediaType, ChannelType, HookTrigger
from roomkit.video.backends.av import AudioVideoBackend
from roomkit.video.bridge import VideoBridge, VideoBridgeConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from roomkit.models.channel import ChannelBinding
    from roomkit.recorder.base import ChannelRecordingConfig
    from roomkit.video.avatar.base import AvatarProvider
    from roomkit.video.backends.base import VideoBackend
    from roomkit.video.base import VideoSession
    from roomkit.video.pipeline.config import VideoPipelineConfig
    from roomkit.video.pipeline.encoder.base import VideoEncoderProvider
    from roomkit.video.video_frame import VideoFrame
    from roomkit.video.vision.base import VisionProvider, VisionResult
    from roomkit.voice.backends.base import VoiceBackend
    from roomkit.voice.base import VoiceSession
    from roomkit.voice.pipeline.config import AudioPipelineConfig
    from roomkit.voice.stt.base import STTProvider
    from roomkit.voice.tts.base import TTSProvider

logger = logging.getLogger("roomkit.av")


class AudioVideoChannel(VideoHooksMixin, VoiceChannel):
    """Combined audio+video channel for A/V backends.

    Extends :class:`VoiceChannel` with video capabilities from
    :class:`VideoHooksMixin`.  Use this with an :class:`AudioVideoBackend`
    like :class:`SIPVideoBackend` or :class:`RTPVideoBackend` that
    produces both audio and video through a single transport.

    Audio flows through the normal voice pipeline (STT/TTS/VAD).
    Video frames, from the backend's ``video`` facet, are delivered to
    registered taps and optionally analysed by a :class:`VisionProvider`.
    A call reports ``ON_SESSION_STARTED``; its video, when it has some,
    ``ON_VIDEO_SESSION_STARTED`` and ``ON_VIDEO_SESSION_ENDED``, each once
    (RFC §12.8.3).
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
        avatar_encoder: VideoEncoderProvider | None = None,
        recording: ChannelRecordingConfig | None = None,
        video_bridge: bool | VideoBridgeConfig | None = None,
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
        self._room_vision: dict[str, tuple[float, str, VisionResult]] = {}
        self._video_media_taps: list[Callable[[VideoSession, VideoFrame], None]] = []
        # Dual-signal: sessions whose video was reported live before
        # bind_session, and those whose video start was reported (so its end
        # is reported once, and only after it).
        self._session_ready_pending_video: set[str] = set()
        self._video_reported: set[str] = set()

        # Video pipeline (decoder, resizer, vision)
        self._video_resources = _VideoResources(video_pipeline, vision)
        self._video_pipeline = self._video_resources.pipeline
        self._video_pipeline_config = video_pipeline

        # Avatar: lip-synced video generation from TTS audio
        self._avatar = avatar
        self._avatar_encoder = avatar_encoder

        # Wire avatar as an outbound audio tap (side-effect, not in audio chain)
        self._outbound_audio_taps: list[Callable[..., None]] = (
            [self._feed_avatar_audio] if avatar is not None else []
        )

        # For async avatar providers (cloud), wire on_video callback
        if avatar is not None and avatar.is_async:
            avatar.on_video(self._on_async_avatar_frame)

        # Idle frame loop: sends avatar idle frames when TTS is not playing
        self._avatar_idle_tasks: dict[str, asyncio.Task[None]] = {}
        # Per-session audio queues for async avatar inference
        self._avatar_queues: dict[str, Any] = {}
        # Lock to serialize H.264 encoding (x264 is not thread-safe)
        self._encoder_lock = threading.Lock()

        # Video bridge for session-to-session forwarding
        if video_bridge is True:
            self._video_bridge: VideoBridge | None = VideoBridge()
        elif isinstance(video_bridge, VideoBridgeConfig):
            self._video_bridge = VideoBridge(video_bridge)
        else:
            self._video_bridge = None

        # Wire the video of the backend's calls
        self._video_backend: VideoBackend | None = (
            backend.video if isinstance(backend, AudioVideoBackend) else None
        )
        if self._video_backend is not None:
            self._video_backend.on_video_received(self._on_video_received)
            self._video_backend.on_session_ready(self._on_video_session_ready)
            self._video_backend.on_client_disconnected(self._on_video_session_ended)

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
        if self._video_resources.closed:
            return
        binding_info = self._session_bindings.get(session.id)
        if binding_info is None:
            logger.debug("Video frame dropped (no binding): session=%s", session.id[:8])
            return
        if frame.keyframe:
            logger.debug(
                "Video keyframe: session=%s seq=%d %d bytes",
                session.id[:8],
                frame.sequence,
                len(frame.data),
            )

        # Run pipeline stages (decoder, resizer) if configured
        if self._video_pipeline is not None:
            processed = self._video_pipeline.process_inbound(session.id, frame)
            if processed is None:
                return
            frame = processed

        if self._video_pipeline is not None:
            for fe in self._video_pipeline.drain_events(session.id):
                if fe.data is not None:
                    fe.data.session = session
                    self._schedule(
                        self._fire_video_detection_hook(session, fe.data, binding_info[0]),
                        name=f"detection:{session.id}",
                    )

        # Deliver to all video taps
        for tap in self._video_media_taps:
            tap(session, frame)

        # Forward to other sessions via video bridge
        if self._video_bridge is not None:
            if self._framework is None or not self._framework.hook_engine.has_hooks(
                HookTrigger.BEFORE_BRIDGE_VIDEO
            ):
                self._video_bridge.forward(session, frame)
            else:
                self._schedule(
                    self._fire_bridge_video_and_forward(session, frame),
                    name=f"bridge_video:{session.id}",
                )

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
        """Bind a voice session and fire video session events.

        A call whose video the video bridge has no room left for is refused
        before anything binds it, as one the audio bridge refuses (RFC §12.8.9).
        """
        # Look up the video session from the combined backend
        video_session = self._get_video_session(session.id)
        bridged = self._admit_video(video_session, room_id)
        try:
            super().bind_session(session, room_id, binding, backend=backend)
        except Exception:
            if bridged and self._video_bridge is not None:
                self._video_bridge.remove_session(session.id)
            raise
        if video_session is None:
            return
        if bridged and self._video_backend is not None:
            # Enable passthrough so the bridge forwards all frames
            # without blocking on keyframe availability — the remote
            # decoder handles its own recovery via RTCP.
            self._video_backend.set_video_passthrough(session.id, True)

        # Start avatar idle frame loop (shows reference image when not speaking)
        self._start_avatar_idle_loop(session)

        with self._state_lock:
            was_ready_pending = session.id in self._session_ready_pending_video
            self._session_ready_pending_video.discard(session.id)
        if was_ready_pending:
            self._report_video_started(video_session, room_id)

    def _admit_video(self, video_session: VideoSession | None, room_id: str) -> bool:
        """Register a call's video with the video bridge; whether it did. The
        bridge raises when it holds ``max_participants`` sessions of the room."""
        if video_session is None or self._video_bridge is None or self._video_backend is None:
            return False
        self._video_bridge.add_session(video_session, room_id, self._video_backend)
        return True

    def _on_video_session_ready(self, video_session: VideoSession) -> None:
        """The backend reports a call's video live: report it once bound."""
        with self._state_lock:
            binding_info = self._session_bindings.get(video_session.id)
            if binding_info is None:
                self._session_ready_pending_video.add(video_session.id)
        if binding_info is not None:
            self._report_video_started(video_session, binding_info[0])

    def _on_video_session_ended(self, video_session: VideoSession) -> None:
        """The backend reports a call's video ended (its call ended)."""
        with self._state_lock:
            self._session_ready_pending_video.discard(video_session.id)
            binding_info = self._session_bindings.get(video_session.id)
        if binding_info is not None:
            self._report_video_ended(video_session, binding_info[0])

    def _report_video_started(self, video_session: VideoSession, room_id: str) -> None:
        with self._state_lock:
            if video_session.id in self._video_reported:
                return
            self._video_reported.add(video_session.id)
        if self._framework:
            self._schedule(
                self._emit_session_event("video_session_started", video_session, room_id),
                name=f"video_session_started:{video_session.id}",
            )
            self._schedule(
                self._fire_session_hook(
                    HookTrigger.ON_VIDEO_SESSION_STARTED, video_session, room_id
                ),
                name=f"video_session_hook:{video_session.id}",
            )

    def _report_video_ended(self, video_session: VideoSession, room_id: str) -> None:
        with self._state_lock:
            if video_session.id not in self._video_reported:
                return
            self._video_reported.discard(video_session.id)
        if self._framework:
            self._schedule(
                self._emit_session_event("video_session_ended", video_session, room_id),
                name=f"video_session_ended:{video_session.id}",
            )
            self._schedule(
                self._fire_session_hook(
                    HookTrigger.ON_VIDEO_SESSION_ENDED, video_session, room_id
                ),
                name=f"video_session_ended_hook:{video_session.id}",
            )

    def unbind_session(self, session: VoiceSession) -> None:
        """Clean up video state before unbinding the voice session."""
        self._stop_avatar_idle_loop(session.id)
        # Signal avatar worker to stop
        q = self._avatar_queues.pop(session.id, None)
        if q is not None:
            q.put(None)
        with self._state_lock:
            self._session_ready_pending_video.discard(session.id)
        # Unregister from video bridge
        if self._video_bridge is not None:
            self._video_bridge.remove_session(session.id)
        binding_info = self._session_bindings.get(session.id)

        # A call unbound while its video is still live (the channel ends it):
        # its video ends here. One its backend already ended was reported then.
        video_session = self._get_video_session(session.id)
        if video_session is not None and binding_info is not None:
            self._report_video_ended(video_session, binding_info[0])

        if self._video_pipeline is not None:
            self._video_pipeline.reset(session.id)
        self._last_vision_results.pop(session.id, None)
        self._forget_session_vision(session.id)
        self._last_vision_ts.pop(session.id, None)
        super().unbind_session(session)

    async def close(self) -> None:
        """Close avatar, vision, video bridge, and video state, then delegate to parent."""
        for sid in list(self._avatar_idle_tasks):
            self._stop_avatar_idle_loop(sid)
        pool = getattr(self, "_avatar_pool", None)
        if pool is not None:
            pool.shutdown(wait=False)
        if self._avatar:
            await self._avatar.close()
        # Cancelled and awaited, but the one running this close (a handler of
        # a session's end that closes the kit), which would wait for itself.
        await cancel_and_wait(*self._scheduled_tasks, log_errors_to=logger)
        self._scheduled_tasks.clear()
        await self._video_resources.close()
        self._last_vision_results.clear()
        self._last_vision_ts.clear()
        self._room_vision.clear()
        self._video_media_taps.clear()
        self._session_ready_pending_video.clear()
        self._video_reported.clear()
        if self._video_bridge is not None:
            self._video_bridge.close()
        await super().close()

    # -- Avatar idle frame loop ------------------------------------------------

    def _start_avatar_idle_loop(self, session: VoiceSession) -> None:
        """Start sending idle frames for a session (reference image visible)."""
        if self._avatar is None or not self._avatar.is_started:
            return
        if self._avatar_encoder is None:
            return
        if session.id in self._avatar_idle_tasks:
            return
        task = asyncio.get_running_loop().create_task(
            self._avatar_idle_loop(session),
            name=f"avatar_idle:{session.id}",
        )
        self._avatar_idle_tasks[session.id] = task

    def _stop_avatar_idle_loop(self, session_id: str) -> None:
        """Cancel the idle frame loop for a session."""
        task = self._avatar_idle_tasks.pop(session_id, None)
        if task is not None:
            task.cancel()

    async def _avatar_idle_loop(self, session: VoiceSession) -> None:
        """Send idle frames at avatar fps while TTS is not playing.

        Automatically pauses when TTS is active (session in
        ``_playing_sessions``) and resumes when TTS finishes.
        """
        if self._avatar is None:
            return
        interval = 1.0 / self._avatar.fps
        logger.info(
            "Avatar idle loop started: session=%s, fps=%d", session.id[:8], self._avatar.fps
        )
        try:
            while True:
                # Skip idle frames while TTS is playing or avatar worker
                # is running — the worker handles video during speech.
                if (
                    session.id not in self._playing_sessions
                    and session.id not in self._avatar_queues
                ):
                    frame = self._avatar.get_idle_frame()
                    if frame is not None:
                        self._submit_avatar_frame(session, frame)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass
        logger.debug("Avatar idle loop stopped: session=%s", session.id[:8])

    # -- Avatar: TTS audio → lip-synced video frames ---------------------------

    def _feed_avatar_audio(self, session: VoiceSession, data: bytes, sample_rate: int) -> None:
        """Queue TTS audio for async avatar inference.

        This tap is called synchronously from the TTS outbound path.
        It MUST NOT block — all heavy work (inference, encoding) runs
        in a background thread via ``_avatar_worker``.
        """
        if self._avatar is None or not self._avatar.is_started:
            return
        if self._avatar_encoder is None:
            return

        q = self._avatar_queues.get(session.id)
        if q is None:
            q = _queue_mod.Queue()
            self._avatar_queues[session.id] = q
            pool = self._ensure_avatar_pool()
            pool.submit(self._avatar_worker, session, q, sample_rate)
        q.put(data)

    def _avatar_worker(
        self,
        session: VoiceSession,
        q: Any,
        sample_rate: int,
    ) -> None:
        """Background thread: drain audio queue → inference → encode → send.

        For sync providers: feed_audio() returns frames, encode + send.
        For async providers: feed_audio() sends to cloud, returns [].
        Video arrives via on_video callback (wired in __init__).

        Exits after 2 seconds of queue inactivity so the idle loop
        can resume.  A new worker is spawned on the next TTS response.
        """
        if self._avatar is None:
            return
        is_async = self._avatar.is_async
        logger.info("Avatar worker started: session=%s async=%s", session.id[:8], is_async)
        total_frames = 0
        while True:
            try:
                data = q.get(timeout=2.0)
            except Exception:
                # No audio for 2s — TTS likely ended
                if is_async:
                    self._avatar.end_turn()
                break
            if data is None:
                if is_async:
                    self._avatar.end_turn()
                break
            try:
                frames = self._avatar.feed_audio(data, sample_rate)
                for frame in frames:
                    self._send_avatar_frame_sync(session, frame)
                    total_frames += 1
            except Exception:
                logger.warning("Avatar worker error", exc_info=True)
        self._avatar_queues.pop(session.id, None)
        logger.info(
            "Avatar worker stopped: session=%s, total_frames=%d",
            session.id[:8],
            total_frames,
        )

    def _on_async_avatar_frame(self, frame: VideoFrame) -> None:
        """Handle a video frame from an async avatar provider (cloud).

        Encode and send to all active sessions.
        """
        for session_id in list(self._avatar_queues):
            binding_info = self._session_bindings.get(session_id)
            if binding_info is None:
                continue
            session = self._get_voice_session(session_id)
            if session is not None:
                self._send_avatar_frame_sync(session, frame)

    def _get_voice_session(self, session_id: str) -> VoiceSession | None:
        """Look up a VoiceSession by ID from the backend."""
        return getattr(self._backend, "get_session", lambda _: None)(session_id)

    def _ensure_avatar_pool(self) -> concurrent.futures.ThreadPoolExecutor:
        """Return or create the avatar thread pool."""
        pool = getattr(self, "_avatar_pool", None)
        if pool is None:
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
            self._avatar_pool = pool
        return pool

    def _submit_avatar_frame(self, session: VoiceSession, frame: VideoFrame) -> None:
        """Submit a video frame for encoding + RTP send in the thread pool."""
        pool = self._ensure_avatar_pool()
        pool.submit(self._send_avatar_frame_sync, session, frame)

    def _send_avatar_frame_sync(self, session: VoiceSession, frame: VideoFrame) -> None:
        """Encode and send a single avatar frame (synchronous, thread-safe)."""
        with self._encoder_lock:
            nals = self._avatar_encoder.encode(frame)  # ty: ignore[unresolved-attribute]
        if not nals:
            return

        is_key = any((nal[0] & 0x1F) == 5 for nal in nals if nal)

        try:
            vcs = getattr(self._backend, "_video_call_sessions", {}).get(session.id)
            if vcs is not None:
                rtp_session = getattr(vcs, "_session", None)
                if rtp_session is not None and hasattr(rtp_session, "send_frame_auto"):
                    rtp_session.send_frame_auto(nals, is_key)
                    return
                self._avatar_frame_count = getattr(self, "_avatar_frame_count", 0) + 1
                ts = self._avatar_frame_count * 3000
                vcs.send_frame(nals, ts, is_key)
        except Exception:
            logger.debug("Avatar frame send failed", exc_info=True)

    # -- Video bridge helpers --------------------------------------------------

    async def _fire_bridge_video_and_forward(
        self, session: VideoSession, frame: VideoFrame
    ) -> None:
        """Fire BEFORE_BRIDGE_VIDEO hook, then forward via video bridge."""
        if self._video_bridge is None or self._framework is None:
            return
        room_id = self._session_bindings.get(session.id, (None, None))[0]
        if room_id is None:
            return
        try:
            from roomkit.video.events import BridgeVideoEvent

            context = await self._framework._build_context(room_id)
            event = BridgeVideoEvent(session=session, frame=frame, room_id=room_id)
            result = await self._framework.hook_engine.run_sync_hooks(
                room_id,
                HookTrigger.BEFORE_BRIDGE_VIDEO,
                event,
                context,
                skip_event_filter=True,
            )
            if not result.allowed:
                return
            # A hook that returned a modified event forwards *its* frame
            # (RFC §12.8). ``set_bridge_filter`` stays the cheaper path for
            # per-frame work — audio-thread, no context build — but a hook
            # that does modify must not be silently ignored.
            if isinstance(result.event, BridgeVideoEvent):
                frame = result.event.frame
        except Exception:
            logger.exception("Error firing BEFORE_BRIDGE_VIDEO hook")
        self._video_bridge.forward(session, frame)

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
        """Look up the call's VideoSession from the backend's video facet."""
        if self._video_backend is None:
            return None
        return self._video_backend.get_session(session_id)

    # -- Public helpers --------------------------------------------------------

    @property
    def vision(self) -> VisionProvider | None:
        """The vision provider (if configured)."""
        return self._vision

    def get_last_vision_result(self, session_id: str) -> VisionResult | None:
        """Get the most recent vision analysis for a session."""
        return self._last_vision_results.get(session_id)
