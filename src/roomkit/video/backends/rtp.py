"""RTP audio+video backend using aiortp.

Extends :class:`RTPVoiceBackend` with video transport. Each
:meth:`connect` call creates both an audio ``RTPSession`` and a video
``VideoRTPSession``.

Requires the ``aiortp`` optional dependency::

    pip install roomkit[rtp]

Usage::

    from roomkit.video.backends import RTPVideoBackend

    backend = RTPVideoBackend(
        local_addr=("0.0.0.0", 10000),
        remote_addr=("192.168.1.100", 20000),
        video_local_addr=("0.0.0.0", 10002),
        video_remote_addr=("192.168.1.100", 20002),
    )
    channel = AudioVideoChannel("av", backend=backend, ...)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from roomkit.video._annexb import nal_units
from roomkit.video.backends.av import AudioVideoBackend, CallVideoBackend
from roomkit.video.base import VideoChunk, VideoSession, VideoSessionState
from roomkit.video.video_frame import VideoFrame
from roomkit.voice.backends.rtp import RTPVoiceBackend
from roomkit.voice.base import VoiceSession

logger = logging.getLogger("roomkit.video.rtp")


class RTPVideoBackend(RTPVoiceBackend, AudioVideoBackend):
    """RTP backend for combined audio + video transport.

    Extends :class:`RTPVoiceBackend` with a parallel video RTP path, served
    by :attr:`video`. Audio-only users should use :class:`RTPVoiceBackend`
    directly.

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
        self._video_rtp_sessions: dict[str, Any] = {}
        self._frame_sequences: dict[str, int] = {}
        self._video = _RTPVideo(self)

    @property
    def name(self) -> str:
        return "RTP-AV"

    @property
    def video(self) -> _RTPVideo:
        """The video of each call, over its parallel video RTP session."""
        return self._video

    # -------------------------------------------------------------------------
    # Session lifecycle
    # -------------------------------------------------------------------------

    async def connect(
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

        self._video_rtp_sessions[voice_session.id] = video_rtp
        self._frame_sequences[voice_session.id] = 0

        video_rtp.on_frame = self._make_video_handler(voice_session.id)

        logger.info(
            "Video RTP session created: session=%s, remote=%s, pt=%d",
            voice_session.id,
            video_remote,
            self._video_payload_type,
        )

        # Its RTP socket is ready, as the audio's was in the parent's connect().
        self._video._started(video_session)

        return voice_session

    async def disconnect(self, session: VoiceSession) -> None:
        sid = session.id

        video_rtp = self._video_rtp_sessions.pop(sid, None)
        if video_rtp is not None:
            await video_rtp.close()

        self._frame_sequences.pop(sid, None)
        self._video._ended(sid)

        await super().disconnect(session)

    async def close(self) -> None:
        # Ending each call (disconnect()) ends its video, reported.
        await super().close()
        for video_rtp in self._video_rtp_sessions.values():
            await video_rtp.close()
        self._video_rtp_sessions.clear()
        self._video._forget()
        self._frame_sequences.clear()

    # -------------------------------------------------------------------------
    # Inbound video
    # -------------------------------------------------------------------------

    def _make_video_handler(self, session_id: str) -> Callable[..., None]:
        """Create an ``on_frame`` callback bound to *session_id*."""

        def _on_frame(nal_data: bytes, timestamp: int, is_keyframe: bool) -> None:
            if not self._video._has_consumers() or self._video.get_session(session_id) is None:
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
            self._video._deliver(session_id, frame)

        return _on_frame


class _RTPVideo(CallVideoBackend):
    """The video of an :class:`RTPVideoBackend`'s calls, over their video RTP."""

    _owner: RTPVideoBackend

    async def send_video(
        self,
        session: VideoSession,
        video: bytes | AsyncIterator[VideoChunk],
    ) -> None:
        video_rtp = self._owner._video_rtp_sessions.get(session.id)
        if video_rtp is None:
            logger.warning("send_video: no video session for %s", session.id)
            return

        if isinstance(video, bytes):
            video_rtp.send_frame(nal_units(video), 0)
        else:
            async for chunk in video:
                ts = int((chunk.timestamp_ms or 0) * 90)
                video_rtp.send_frame(nal_units(chunk.data), ts, chunk.keyframe)

    def send_video_sync(self, session: VideoSession, frame: VideoFrame) -> None:
        """Synchronously send a video frame via RTP.

        Called by the video bridge from the RTP receive thread.
        Schedules ``send_frame()`` on the event loop via
        ``call_soon_threadsafe`` to avoid cross-thread corruption.
        """
        video_rtp = self._owner._video_rtp_sessions.get(session.id)
        if video_rtp is None:
            return
        ts = int((frame.timestamp_ms or 0) * 90) & 0xFFFFFFFF  # ms → 90kHz RTP clock
        data = nal_units(frame.data)
        is_key = frame.keyframe
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(video_rtp.send_frame, data, ts, is_key)
                return
        except RuntimeError:
            pass
        video_rtp.send_frame(data, ts, is_key)
