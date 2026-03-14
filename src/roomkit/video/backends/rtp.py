"""RTP audio+video backend using aiortp.

Extends :class:`RTPVoiceBackend` with video transport. Each
:meth:`connect` call creates both an audio ``RTPSession`` and a video
``VideoRTPSession``.

Requires the ``aiortp`` optional dependency::

    pip install roomkit[rtp]

Usage::

    from roomkit.video.backends.rtp import RTPVideoBackend

    backend = RTPVideoBackend(
        local_addr=("0.0.0.0", 10000),
        remote_addr=("192.168.1.100", 20000),
        video_local_addr=("0.0.0.0", 10002),
        video_remote_addr=("192.168.1.100", 20002),
    )
    channel = AudioVideoChannel("av", backend=backend, ...)
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from roomkit.video.backends.base import VideoBackend
from roomkit.video.base import (
    VideoCapability,
    VideoChunk,
    VideoDisconnectCallback,
    VideoReceivedCallback,
    VideoSession,
    VideoSessionReadyCallback,
    VideoSessionState,
)
from roomkit.video.video_frame import VideoFrame
from roomkit.voice.backends.rtp import RTPVoiceBackend
from roomkit.voice.base import VoiceSession

logger = logging.getLogger("roomkit.video.rtp")


class RTPVideoBackend(RTPVoiceBackend, VideoBackend):  # type: ignore[misc]
    """RTP backend for combined audio + video transport.

    Extends :class:`RTPVoiceBackend` with a parallel video RTP path.
    Audio-only users should use :class:`RTPVoiceBackend` directly.

    Args:
        video_local_addr: ``(host, port)`` to bind video RTP.
        video_remote_addr: ``(host, port)`` to send video RTP to.
            May be ``None`` if supplied per-session via
            ``metadata["video_remote_addr"]`` in :meth:`connect`.
        video_payload_type: RTP payload type for video (default 96 = H.264).
        video_clock_rate: Video RTP clock rate (default 90000).
        **kwargs: Forwarded to :class:`RTPVoiceBackend`.
    """

    def __init__(
        self,
        *,
        video_local_addr: tuple[str, int] = ("0.0.0.0", 0),  # nosec B104
        video_remote_addr: tuple[str, int] | None = None,
        video_payload_type: int = 96,
        video_clock_rate: int = 90000,
        video_port_allocator: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self._video_local_addr = video_local_addr
        self._video_remote_addr = video_remote_addr
        self._video_payload_type = video_payload_type
        self._video_clock_rate = video_clock_rate
        self._video_port_allocator = video_port_allocator

        # Video session tracking
        self._video_sessions: dict[str, VideoSession] = {}
        self._video_rtp_sessions: dict[str, Any] = {}
        self._video_received_callback: VideoReceivedCallback | None = None
        self._video_taps: list[VideoReceivedCallback] = []
        self._video_session_ready_callbacks: list[VideoSessionReadyCallback] = []
        self._video_disconnect_callbacks: list[VideoDisconnectCallback] = []
        self._frame_sequences: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "RTP-AV"

    @property
    def video_capabilities(self) -> VideoCapability:
        """Video-specific capabilities."""
        return VideoCapability.NONE

    # -------------------------------------------------------------------------
    # Session lifecycle
    # -------------------------------------------------------------------------

    async def connect(  # type: ignore[override]
        self,
        room_id: str,
        participant_id: str,
        channel_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> VoiceSession:
        metadata = metadata or {}

        video_remote = metadata.pop("video_remote_addr", None) or self._video_remote_addr
        if video_remote is None:
            raise ValueError(
                "video_remote_addr must be provided either in RTPVideoBackend() "
                "constructor or via metadata['video_remote_addr'] in connect()"
            )

        # Audio session via parent
        voice_session = await super().connect(
            room_id, participant_id, channel_id, metadata=metadata
        )

        # Video RTP session
        video_rtp = await self._aiortp.VideoRTPSession.create(
            local_addr=self._video_local_addr,
            remote_addr=video_remote,
            payload_type=self._video_payload_type,
            clock_rate=self._video_clock_rate,
            port_allocator=self._video_port_allocator,
        )

        video_session = VideoSession(
            id=voice_session.id,
            room_id=room_id,
            participant_id=participant_id,
            channel_id=channel_id,
            state=VideoSessionState.ACTIVE,
            metadata={
                "payload_type": self._video_payload_type,
                "clock_rate": self._video_clock_rate,
                "video_remote_addr": video_remote,
                "backend": "rtp-av",
            },
        )

        self._video_sessions[voice_session.id] = video_session
        self._video_rtp_sessions[voice_session.id] = video_rtp
        self._frame_sequences[voice_session.id] = 0

        video_rtp.on_frame = self._make_video_handler(voice_session.id)

        logger.info(
            "Video RTP session created: session=%s, remote=%s, pt=%d",
            voice_session.id,
            video_remote,
            self._video_payload_type,
        )

        for cb in self._video_session_ready_callbacks:
            cb(video_session)

        return voice_session

    async def disconnect(self, session: VoiceSession) -> None:  # type: ignore[override]
        sid = session.id

        video_rtp = self._video_rtp_sessions.pop(sid, None)
        if video_rtp is not None:
            await video_rtp.close()

        video_session = self._video_sessions.pop(sid, None)
        self._frame_sequences.pop(sid, None)

        if video_session is not None:
            video_session.state = VideoSessionState.ENDED
            for cb in self._video_disconnect_callbacks:
                cb(video_session)

        await super().disconnect(session)

    async def close(self) -> None:
        for video_rtp in self._video_rtp_sessions.values():
            await video_rtp.close()
        self._video_rtp_sessions.clear()
        self._video_sessions.clear()
        self._frame_sequences.clear()
        await super().close()

    # -------------------------------------------------------------------------
    # Inbound video
    # -------------------------------------------------------------------------

    def _make_video_handler(self, session_id: str) -> Callable[..., None]:
        """Create an ``on_frame`` callback bound to *session_id*."""

        def _on_frame(nal_data: bytes, timestamp: int, is_keyframe: bool) -> None:
            if self._video_received_callback is None and not self._video_taps:
                return
            video_session = self._video_sessions.get(session_id)
            if video_session is None:
                return
            seq = self._frame_sequences.get(session_id, 0)
            self._frame_sequences[session_id] = seq + 1
            frame = VideoFrame(
                data=nal_data,
                codec="h264",
                timestamp_ms=timestamp / 90.0,
                keyframe=is_keyframe,
                sequence=seq,
            )
            if self._video_received_callback is not None:
                self._video_received_callback(video_session, frame)
            for tap in self._video_taps:
                tap(video_session, frame)

        return _on_frame

    # -------------------------------------------------------------------------
    # Outbound video
    # -------------------------------------------------------------------------

    async def send_video(
        self,
        session: VideoSession,
        video: bytes | AsyncIterator[VideoChunk],
    ) -> None:
        video_rtp = self._video_rtp_sessions.get(session.id)
        if video_rtp is None:
            logger.warning("send_video: no video session for %s", session.id)
            return

        if isinstance(video, bytes):
            video_rtp.send_frame([video], 0)
        else:
            async for chunk in video:
                ts = int((chunk.timestamp_ms or 0) * 90)
                video_rtp.send_frame([chunk.data], ts, chunk.keyframe)

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def on_video_received(self, callback: VideoReceivedCallback) -> None:
        self._video_received_callback = callback

    def add_video_tap(self, callback: VideoReceivedCallback) -> None:
        self._video_taps.append(callback)

    def on_session_ready(self, callback: VideoSessionReadyCallback) -> None:  # type: ignore[override]
        """Register callback for video session ready events."""
        self._video_session_ready_callbacks.append(callback)

    def on_client_disconnected(self, callback: VideoDisconnectCallback) -> None:  # type: ignore[override]
        self._video_disconnect_callbacks.append(callback)

    # -------------------------------------------------------------------------
    # Session queries
    # -------------------------------------------------------------------------

    def get_video_session(self, session_id: str) -> VideoSession | None:
        """Get the video session for a given session ID."""
        return self._video_sessions.get(session_id)

    def list_video_sessions(self, room_id: str) -> list[VideoSession]:
        """List active video sessions in a room."""
        return [s for s in self._video_sessions.values() if s.room_id == room_id]
