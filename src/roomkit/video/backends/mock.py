"""Mock video backend for testing."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from typing import Any
from uuid import uuid4

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


@dataclass
class MockVideoCall:
    """Record of a call made to MockVideoBackend."""

    method: str
    args: dict[str, Any] = field(default_factory=dict)


class MockVideoBackend(VideoBackend):
    """Mock video backend for testing.

    Tracks all method calls and provides helpers to simulate events.
    The backend is a pure transport — no video processing.

    Example::

        backend = MockVideoBackend()

        # Track calls
        session = await backend.connect("room-1", "user-1", "video-1")
        assert backend.calls[-1].method == "connect"

        # Simulate video frame received
        frame = VideoFrame(data=b"\\x00" * 100, codec="h264", width=640, height=480)
        await backend.simulate_video_received(session, frame)
    """

    def __init__(
        self,
        *,
        capabilities: VideoCapability = VideoCapability.NONE,
    ) -> None:
        self._sessions: dict[str, VideoSession] = {}
        self._video_received_callbacks: list[VideoReceivedCallback] = []
        self._session_ready_callbacks: list[VideoSessionReadyCallback] = []
        self._disconnect_callbacks: list[VideoDisconnectCallback] = []
        # Tracking
        self.calls: list[MockVideoCall] = []
        self.sent_video: list[tuple[str, bytes]] = []  # (session_id, data)
        self._capabilities = capabilities

    @property
    def name(self) -> str:
        return "MockVideoBackend"

    @property
    def capabilities(self) -> VideoCapability:
        return self._capabilities

    async def connect(
        self,
        room_id: str,
        participant_id: str,
        channel_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> VideoSession:
        session = VideoSession(
            id=uuid4().hex,
            room_id=room_id,
            participant_id=participant_id,
            channel_id=channel_id,
            state=VideoSessionState.ACTIVE,
            metadata=metadata or {},
        )
        self._sessions[session.id] = session
        self.calls.append(
            MockVideoCall(
                method="connect",
                args={
                    "room_id": room_id,
                    "participant_id": participant_id,
                    "channel_id": channel_id,
                    "metadata": metadata,
                },
            )
        )
        # Fire session ready — video path is live immediately for mock
        for cb in self._session_ready_callbacks:
            result = cb(session)
            if hasattr(result, "__await__"):
                await result
        return session

    async def disconnect(self, session: VideoSession) -> None:
        if session.id in self._sessions:
            self._sessions[session.id] = replace(session, state=VideoSessionState.ENDED)
        self.calls.append(MockVideoCall(method="disconnect", args={"session_id": session.id}))

    async def send_video(
        self,
        session: VideoSession,
        video: bytes | AsyncIterator[VideoChunk],
    ) -> None:
        if isinstance(video, bytes):
            self.sent_video.append((session.id, video))
        else:
            chunks: list[bytes] = []
            async for chunk in video:
                chunks.append(chunk.data)
            combined = b"".join(chunks)
            self.sent_video.append((session.id, combined))

        self.calls.append(MockVideoCall(method="send_video", args={"session_id": session.id}))

    def get_session(self, session_id: str) -> VideoSession | None:
        return self._sessions.get(session_id)

    def list_sessions(self, room_id: str) -> list[VideoSession]:
        return [s for s in self._sessions.values() if s.room_id == room_id]

    async def close(self) -> None:
        self._sessions.clear()
        self.sent_video.clear()
        self.calls.clear()
        self.calls.append(MockVideoCall(method="close"))

    # -------------------------------------------------------------------------
    # Raw video delivery
    # -------------------------------------------------------------------------

    def on_video_received(self, callback: VideoReceivedCallback) -> None:
        self._video_received_callbacks.append(callback)

    def on_session_ready(self, callback: VideoSessionReadyCallback) -> None:
        self._session_ready_callbacks.append(callback)

    def on_client_disconnected(self, callback: VideoDisconnectCallback) -> None:
        self._disconnect_callbacks.append(callback)

    # -------------------------------------------------------------------------
    # Test helpers
    # -------------------------------------------------------------------------

    async def simulate_video_received(self, session: VideoSession, frame: VideoFrame) -> None:
        """Simulate the backend receiving a raw video frame.

        Fires all registered on_video_received callbacks.
        """
        for callback in self._video_received_callbacks:
            result = callback(session, frame)
            if hasattr(result, "__await__"):
                await result

    async def simulate_session_ready(self, session: VideoSession) -> None:
        """Simulate the backend signalling that a session's video path is live.

        Fires all registered on_session_ready callbacks.
        """
        for cb in self._session_ready_callbacks:
            result = cb(session)
            if hasattr(result, "__await__"):
                await result

    async def simulate_client_disconnected(self, session: VideoSession) -> None:
        """Simulate the backend signalling that a client has disconnected.

        Fires all registered on_client_disconnected callbacks.
        """
        for cb in self._disconnect_callbacks:
            result = cb(session)
            if hasattr(result, "__await__"):
                await result
