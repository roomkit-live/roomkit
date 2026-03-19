"""Video bridge for direct session-to-session video forwarding."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from roomkit.video.backends.base import VideoBackend
    from roomkit.video.base import VideoSession
    from roomkit.video.video_frame import VideoFrame

logger = logging.getLogger("roomkit.video.bridge")

# Synchronous filter called before forwarding each frame.
# Returns the (possibly modified) frame, or None to drop it.
BridgeVideoFrameFilter = Callable[["VideoSession", "VideoFrame"], "VideoFrame | None"]

# Synchronous processor called for each (target_session, frame) pair.
# Returns a VideoFrame ready for send_video_sync.
BridgeVideoFrameProcessor = Callable[["VideoSession", "VideoFrame"], "VideoFrame"]


@dataclass
class VideoBridgeConfig:
    """Configuration for video bridging between video sessions.

    Args:
        enabled: Whether bridging is active.
        max_participants: Maximum sessions per room.
        forwarding_strategy: ``"forward"`` for direct frame forwarding.
            ``"composite"`` (grid layout) is deferred to a future release.
    """

    enabled: bool = True
    max_participants: int = 10
    forwarding_strategy: Literal["forward"] = "forward"
    keyframe_interval_s: float = 5.0
    """Request a keyframe from each source if none seen within this interval.
    Ensures decoder recovery after packet loss on the bridge→receiver path.
    Set to ``0`` to disable periodic keyframe requests."""


@dataclass
class _BridgedVideoSession:
    """A session registered for video bridging."""

    session: VideoSession
    room_id: str
    backend: VideoBackend


class VideoBridge:
    """Forwards video frames between video sessions in the same room.

    For 2-party rooms, video from session A is sent directly to
    session B and vice versa.  N-party compositing (grid layout)
    is deferred — only ``"forward"`` mode is supported for now.

    All public methods are thread-safe — ``forward()`` is called
    from video callback threads.
    """

    def __init__(self, config: VideoBridgeConfig | None = None) -> None:
        self._config = config or VideoBridgeConfig()
        # room_id -> {session_id -> _BridgedVideoSession}
        self._rooms: dict[str, dict[str, _BridgedVideoSession]] = {}
        self._lock = threading.Lock()
        self._frame_filter: BridgeVideoFrameFilter | None = None
        self._frame_processor: BridgeVideoFrameProcessor | None = None
        # Track last keyframe time per source session for periodic PLI
        self._last_keyframe_at: dict[str, float] = {}

    @property
    def config(self) -> VideoBridgeConfig:
        return self._config

    def set_frame_filter(self, fn: BridgeVideoFrameFilter | None) -> None:
        """Set a synchronous filter called before forwarding each frame.

        The filter receives ``(source_session, frame)`` and returns the
        frame (possibly modified) or ``None`` to drop it.  Runs in the
        video callback thread — must complete quickly.
        """
        self._frame_filter = fn

    def set_frame_processor(self, fn: BridgeVideoFrameProcessor | None) -> None:
        """Set a processor for outbound frames before sending.

        The processor receives ``(target_session, frame)`` and returns
        a ``VideoFrame`` ready for ``send_video_sync``.  Used for
        per-target transcoding or resolution adaptation.
        """
        self._frame_processor = fn

    def add_session(
        self,
        session: VideoSession,
        room_id: str,
        backend: VideoBackend,
    ) -> None:
        """Register a session for video bridging.

        Args:
            session: The video session to bridge.
            room_id: The room this session belongs to.
            backend: The backend used to send video to this session.

        Raises:
            RuntimeError: If the room has reached ``max_participants``.
        """
        with self._lock:
            room = self._rooms.setdefault(room_id, {})
            if session.id not in room and len(room) >= self._config.max_participants:
                raise RuntimeError(
                    f"Room {room_id} has reached max bridge participants "
                    f"({self._config.max_participants})"
                )
            # Collect existing sessions before adding the new one —
            # we'll request keyframes from them so the new participant's
            # decoder can start immediately.
            existing = list(room.values())
            room[session.id] = _BridgedVideoSession(
                session=session,
                room_id=room_id,
                backend=backend,
            )
            logger.info(
                "VideoBridge: added session %s to room %s (%d participants)",
                session.id,
                room_id,
                len(room),
            )

        # Request keyframes from all existing sessions (outside lock).
        # Their next keyframe will be forwarded to the new participant,
        # allowing its decoder to start without waiting for the natural
        # keyframe interval.
        for bs in existing:
            try:
                bs.backend.request_keyframe(bs.session)
            except Exception:
                logger.debug(
                    "Failed to request keyframe from %s", bs.session.id[:8], exc_info=True
                )

    def remove_session(self, session_id: str) -> None:
        """Unregister a session from video bridging.

        Args:
            session_id: The session to remove.
        """
        with self._lock:
            found_room_id: str | None = None
            for room_id, room in self._rooms.items():
                if session_id in room:
                    del room[session_id]
                    found_room_id = room_id
                    break
            if found_room_id is not None:
                if not self._rooms[found_room_id]:
                    del self._rooms[found_room_id]
                self._last_keyframe_at.pop(session_id, None)
                logger.info(
                    "VideoBridge: removed session %s from room %s",
                    session_id,
                    found_room_id,
                )

    def forward(self, session: VideoSession, frame: VideoFrame) -> None:
        """Forward a video frame from this session to others in the room.

        Sends the source frame directly to every other session.

        Args:
            session: The source session.
            frame: The video frame to forward.
        """
        if not self._config.enabled:
            return

        # Pre-forward filter (BEFORE_BRIDGE_VIDEO)
        if self._frame_filter is not None:
            filtered = self._frame_filter(session, frame)
            if filtered is None:
                return  # frame dropped
            frame = filtered

        # Track keyframes and request new ones when stale.
        # Packet loss on the bridge→receiver path can freeze decoders;
        # periodic PLI ensures recovery within keyframe_interval_s.
        if frame.keyframe:
            self._last_keyframe_at[session.id] = time.monotonic()
        elif self._config.keyframe_interval_s > 0:
            now = time.monotonic()
            last = self._last_keyframe_at.get(session.id, 0.0)
            if now - last >= self._config.keyframe_interval_s:
                self._last_keyframe_at[session.id] = now
                self._request_keyframe_for(session.id)

        with self._lock:
            targets = self._get_targets(session.id)

        if not targets:
            return

        for target in targets:
            self._send_to_target(session, target, frame)

    def get_participant_count(self, room_id: str) -> int:
        """Return the number of bridged sessions in a room."""
        with self._lock:
            room = self._rooms.get(room_id)
            return len(room) if room else 0

    def get_bridged_sessions(self, room_id: str) -> list[tuple[VideoSession, VideoBackend]]:
        """Return ``(session, backend)`` pairs for all sessions in a room."""
        with self._lock:
            room = self._rooms.get(room_id)
            if not room:
                return []
            return [(bs.session, bs.backend) for bs in room.values()]

    def close(self) -> None:
        """Remove all sessions and clean up."""
        with self._lock:
            self._rooms.clear()
            self._last_keyframe_at.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request_keyframe_for(self, session_id: str) -> None:
        """Request a keyframe from a source session's backend."""
        with self._lock:
            for room in self._rooms.values():
                bs = room.get(session_id)
                if bs is not None:
                    break
            else:
                return
        try:
            bs.backend.request_keyframe(bs.session)
        except Exception:
            logger.debug("Failed to request keyframe from %s", session_id[:8], exc_info=True)

    def _get_targets(self, source_id: str) -> list[_BridgedVideoSession]:
        """Get target sessions for forwarding (caller holds lock)."""
        for room in self._rooms.values():
            if source_id in room:
                return [bs for sid, bs in room.items() if sid != source_id]
        return []

    def _send_to_target(
        self,
        source: VideoSession,
        target: _BridgedVideoSession,
        frame: VideoFrame,
    ) -> None:
        try:
            if self._frame_processor is not None:
                frame = self._frame_processor(target.session, frame)
            target.backend.send_video_sync(target.session, frame)
        except Exception:
            logger.exception(
                "VideoBridge: failed to forward video from %s to %s",
                source.id,
                target.session.id,
            )
