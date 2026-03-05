"""Audio bridge for direct session-to-session audio forwarding."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame
    from roomkit.voice.backends.base import VoiceBackend
    from roomkit.voice.base import AudioChunk, VoiceSession

logger = logging.getLogger("roomkit.voice.bridge")

# Synchronous filter called before forwarding each frame.
# Returns the (possibly modified) frame, or None to drop it.
BridgeFrameFilter = Callable[["VoiceSession", "AudioFrame"], "AudioFrame | None"]

# Synchronous processor called for each (target_session, frame) pair.
# Returns an AudioChunk ready for send_audio_sync.  Used by VoiceChannel
# to run outbound pipeline (recorder tap, AEC reference, resampler).
BridgeFrameProcessor = Callable[["VoiceSession", "AudioFrame"], "AudioChunk"]


@dataclass
class AudioBridgeConfig:
    """Configuration for audio bridging between voice sessions.

    Args:
        enabled: Whether bridging is active.
        max_participants: Maximum sessions per room.
        mixing_strategy: ``"forward"`` for 2-party direct forwarding,
            ``"mix"`` for N-party additive mixing.
    """

    enabled: bool = True
    max_participants: int = 10
    mixing_strategy: Literal["forward", "mix"] = "forward"


@dataclass
class _BridgedSession:
    """A session registered for bridging."""

    session: VoiceSession
    room_id: str
    backend: VoiceBackend


class AudioBridge:
    """Forwards audio frames between voice sessions in the same room.

    For 2-party rooms (``mixing_strategy="forward"``), audio from session A
    is sent directly to session B and vice versa.  For N-party rooms
    (``mixing_strategy="mix"``), audio from each session is mixed with all
    other sessions' audio before forwarding.

    All public methods are thread-safe — ``forward()`` is called from audio
    callback threads.
    """

    def __init__(self, config: AudioBridgeConfig | None = None) -> None:
        self._config = config or AudioBridgeConfig()
        # room_id -> {session_id -> _BridgedSession}
        self._rooms: dict[str, dict[str, _BridgedSession]] = {}
        self._lock = threading.Lock()
        self._frame_filter: BridgeFrameFilter | None = None
        self._frame_processor: BridgeFrameProcessor | None = None

    @property
    def config(self) -> AudioBridgeConfig:
        return self._config

    def set_frame_filter(self, fn: BridgeFrameFilter | None) -> None:
        """Set a synchronous filter called before forwarding each frame.

        The filter receives ``(source_session, frame)`` and returns the
        frame (possibly modified) or ``None`` to drop it.  Runs in the
        audio callback thread — must complete in < 1ms.
        """
        self._frame_filter = fn

    def set_frame_processor(self, fn: BridgeFrameProcessor | None) -> None:
        """Set a processor for outbound frames before sending.

        The processor receives ``(target_session, frame)`` and returns
        an ``AudioChunk`` ready for ``send_audio_sync``.  Used by
        VoiceChannel to run the outbound pipeline (recorder tap, AEC
        reference, resampler) on bridged audio.
        """
        self._frame_processor = fn

    def add_session(
        self,
        session: VoiceSession,
        room_id: str,
        backend: VoiceBackend,
    ) -> None:
        """Register a session for audio bridging.

        Args:
            session: The voice session to bridge.
            room_id: The room this session belongs to.
            backend: The backend used to send audio to this session.

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
            room[session.id] = _BridgedSession(
                session=session,
                room_id=room_id,
                backend=backend,
            )
            logger.info(
                "Bridge: added session %s to room %s (%d participants)",
                session.id,
                room_id,
                len(room),
            )

    def remove_session(self, session_id: str) -> None:
        """Unregister a session from audio bridging.

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
                logger.info(
                    "Bridge: removed session %s from room %s",
                    session_id,
                    found_room_id,
                )

    def forward(self, session: VoiceSession, frame: AudioFrame) -> None:
        """Forward audio from this session to other sessions in the room.

        For ``"forward"`` strategy: sends directly to the other session
        (must be exactly 2 sessions).  For ``"mix"`` strategy: sends to
        all other sessions individually (mixing is future work for N>2).

        Args:
            session: The source session.
            frame: The processed audio frame to forward.
        """
        # Pre-forward filter (BEFORE_BRIDGE_AUDIO)
        if self._frame_filter is not None:
            filtered = self._frame_filter(session, frame)
            if filtered is None:
                return  # frame dropped
            frame = filtered

        with self._lock:
            targets = self._get_targets(session.id)

        if not targets:
            return

        for target in targets:
            try:
                if self._frame_processor is not None:
                    chunk = self._frame_processor(target.session, frame)
                else:
                    chunk = self._frame_to_chunk(frame)
                target.backend.send_audio_sync(target.session, chunk)
            except Exception:
                logger.exception(
                    "Bridge: failed to forward audio from %s to %s",
                    session.id,
                    target.session.id,
                )

    def get_participant_count(self, room_id: str) -> int:
        """Return the number of bridged sessions in a room."""
        with self._lock:
            room = self._rooms.get(room_id)
            return len(room) if room else 0

    def close(self) -> None:
        """Remove all sessions and clean up."""
        with self._lock:
            self._rooms.clear()

    def _get_targets(self, source_id: str) -> list[_BridgedSession]:
        """Get target sessions for forwarding (caller holds lock)."""
        for room in self._rooms.values():
            if source_id in room:
                return [bs for sid, bs in room.items() if sid != source_id]
        return []

    @staticmethod
    def _frame_to_chunk(frame: AudioFrame) -> AudioChunk:
        """Convert an AudioFrame to an AudioChunk for send_audio_sync."""
        from roomkit.voice.base import AudioChunk

        return AudioChunk(
            data=frame.data,
            sample_rate=frame.sample_rate,
            channels=frame.channels,
        )
