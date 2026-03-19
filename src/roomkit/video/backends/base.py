"""VideoBackend abstract base class."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from roomkit.video.base import (
    VideoCapability,
    VideoChunk,
    VideoDisconnectCallback,
    VideoReceivedCallback,
    VideoSession,
    VideoSessionReadyCallback,
)

if TYPE_CHECKING:
    from roomkit.video.video_frame import VideoFrame

logger = logging.getLogger("roomkit.video.backend")


class VideoBackend(ABC):
    """Abstract base class for video transport backends.

    VideoBackend handles the transport layer for real-time video:

    - Managing video session connections
    - Streaming video frames to/from clients
    - Delivering raw inbound video frames via on_video_received

    The backend is framework-agnostic and a pure transport — all video
    intelligence (vision AI, overlays, etc.) is handled by the
    VideoPipeline or VisionProvider.

    Example::

        backend = WebRTCVideoBackend()

        # Register raw video callback
        backend.on_video_received(handle_video_frame)

        # Connect a participant
        session = await backend.connect("room-1", "user-1", "video-channel")

        # Stream video to the client
        await backend.send_video(session, video_chunks)

        # Disconnect
        await backend.disconnect(session)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name (e.g., 'webrtc', 'rtmp', 'srt')."""
        ...

    async def connect(
        self,
        room_id: str,
        participant_id: str,
        channel_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> VideoSession:
        """Create a new video session for a participant.

        Backends that initiate connections override this.  Backends that
        receive external connections (e.g. WebRTC offer) override
        :meth:`accept` instead.

        Args:
            room_id: The room to join.
            participant_id: The participant's ID.
            channel_id: The video channel ID.
            metadata: Optional session metadata.

        Returns:
            A VideoSession representing the connection.
        """
        raise NotImplementedError(f"{self.name} does not implement connect()")

    @abstractmethod
    async def disconnect(self, session: VideoSession) -> None:
        """End a video session.

        Args:
            session: The session to disconnect.
        """
        ...

    @abstractmethod
    async def send_video(
        self,
        session: VideoSession,
        video: bytes | AsyncIterator[VideoChunk],
    ) -> None:
        """Send video to a session.

        Args:
            session: The target session.
            video: Raw frame bytes or an async iterator of VideoChunks
                for streaming.
        """
        ...

    def send_video_sync(self, session: VideoSession, frame: VideoFrame) -> None:
        """Synchronously send a video frame to a session.

        Used by the video bridge for frame forwarding from callbacks
        where ``await`` is not available.  The default schedules the
        async ``send_video()`` on the event loop.

        Args:
            session: The target session.
            frame: The video frame to send.
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.send_video(session, frame.data))
        except RuntimeError:
            logger.warning(
                "send_video_sync: no event loop for session %s",
                session.id,
            )

    def request_keyframe(self, session: VideoSession) -> None:  # noqa: B027
        """Request a keyframe (PLI/FIR) from the remote endpoint.

        Called by the video bridge when a new participant joins so
        their decoder can start immediately.  Backends that support
        RTCP feedback should override this.

        Args:
            session: The session to request a keyframe from.
        """

    def set_video_passthrough(self, session_id: str, enabled: bool = True) -> None:  # noqa: B027
        """Enable passthrough (bridge) mode for a session.

        In passthrough mode the backend delivers every assembled frame
        to :meth:`on_video_received` regardless of keyframe status.
        Use this for SFU / bridge topologies where the remote decoder
        handles its own recovery.

        Args:
            session_id: The session to configure.
            enabled: ``True`` to enable passthrough.
        """

    def get_session(self, session_id: str) -> VideoSession | None:
        """Get a session by ID.

        Override for backends that track sessions internally.

        Args:
            session_id: The session ID to look up.

        Returns:
            The VideoSession if found, None otherwise.
        """
        return None

    def get_video_session(self, session_id: str) -> VideoSession | None:
        """Get a video session by ID.

        Alias for :meth:`get_session` that avoids return-type conflicts
        in combined A/V backends where the voice parent also defines
        ``get_session`` with a ``VoiceSession`` return type.
        """
        return self.get_session(session_id)

    def list_sessions(self, room_id: str) -> list[VideoSession]:
        """List all active sessions in a room.

        Override for backends that track sessions internally.

        Args:
            room_id: The room to list sessions for.

        Returns:
            List of active VideoSessions in the room.
        """
        return []

    async def close(self) -> None:  # noqa: B027
        """Release backend resources.

        Override in subclasses that need cleanup.
        """

    # -------------------------------------------------------------------------
    # Capabilities
    # -------------------------------------------------------------------------

    @property
    def capabilities(self) -> VideoCapability:
        """Declare supported capabilities.

        Override to enable features like simulcast, screen sharing, etc.
        By default, no optional capabilities are supported.

        Returns:
            Flags indicating supported capabilities.
        """
        return VideoCapability.NONE

    # -------------------------------------------------------------------------
    # Raw video delivery (pipeline integration)
    # -------------------------------------------------------------------------

    def on_video_received(self, callback: VideoReceivedCallback) -> None:  # noqa: B027
        """Register a callback for raw inbound video frames.

        The pipeline or channel calls this to receive every video frame
        produced by the transport.

        Args:
            callback: Function called with (session, video_frame).
        """

    def add_video_tap(self, callback: VideoReceivedCallback) -> None:  # noqa: B027
        """Register an additional video frame consumer (e.g. recording).

        Unlike :meth:`on_video_received` (single primary consumer), taps
        are additive — multiple taps can coexist.

        Args:
            callback: Function called with (session, video_frame).
        """

    def on_session_ready(self, callback: VideoSessionReadyCallback) -> None:  # noqa: B027
        """Register callback for when a session's video path becomes live.

        Fired when the transport is ready to send/receive video for a
        session (e.g. WebRTC connected, RTMP handshake complete).

        Args:
            callback: Function called with ``(session)``.
        """

    def on_client_disconnected(self, callback: VideoDisconnectCallback) -> None:  # noqa: B027
        """Register callback for client disconnection.

        Args:
            callback: Called with (session) when the client disconnects.
        """

    # -------------------------------------------------------------------------
    # Realtime transport methods
    # -------------------------------------------------------------------------

    async def accept(self, session: VideoSession, connection: Any) -> None:
        """Bind an external connection to a session.

        Backends that receive connections from external sources (e.g.
        WebRTC offer/answer, RTMP publish) override this.

        Args:
            session: The video session to bind.
            connection: Protocol-specific connection object.
        """
        raise NotImplementedError(f"{self.name} does not implement accept()")

    # -------------------------------------------------------------------------
    # Protocol trace
    # -------------------------------------------------------------------------

    def set_trace_emitter(  # noqa: B027
        self,
        emitter: Callable[..., Any] | None,
    ) -> None:
        """Set a callback for emitting protocol traces.

        Called by the owning channel when trace observers are registered.

        Args:
            emitter: The channel's ``emit_trace`` method, or ``None``
                to disable.
        """
