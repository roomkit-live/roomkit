"""Audio bridge for direct session-to-session audio forwarding."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame
    from roomkit.voice.backends.base import VoiceBackend
    from roomkit.voice.base import AudioChunk, VoiceSession
    from roomkit.voice.pipeline.mixer.base import MixerProvider
    from roomkit.voice.pipeline.resampler.base import ResamplerProvider

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
        mixer: Mixer provider for PCM frame mixing.  Defaults to
            :class:`~roomkit.voice.pipeline.mixer.numpy.NumpyMixerProvider`
            when NumPy is available, otherwise falls back to
            :class:`~roomkit.voice.pipeline.mixer.python.PythonMixerProvider`.
    """

    enabled: bool = True
    max_participants: int = 10
    mixing_strategy: Literal["forward", "mix"] = "forward"
    mixer: MixerProvider | None = None


@dataclass
class _BridgedSession:
    """A session registered for bridging."""

    session: VoiceSession
    room_id: str
    backend: VoiceBackend
    sample_rate: int = 16000
    """Native inbound sample rate for this session (from backend/metadata)."""
    output_sample_rate: int = 16000
    """Target outbound sample rate for sending audio to this session."""


@dataclass
class _MixingBuffer:
    """Per-room buffer holding the latest frame from each session."""

    frames: dict[str, AudioFrame] = field(default_factory=dict)
    """session_id -> latest AudioFrame."""


class AudioBridge:
    """Forwards audio frames between voice sessions in the same room.

    For 2-party rooms (``mixing_strategy="forward"``), audio from session A
    is sent directly to session B and vice versa.  For N-party rooms
    (``mixing_strategy="mix"``), audio from each session is mixed so that
    each participant hears all others but not themselves.

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
        # Mixing buffers for "mix" strategy
        self._mix_buffers: dict[str, _MixingBuffer] = {}
        # Mixer: explicit > auto-detect (numpy > python)
        self._mixer = self._config.mixer or _get_default_mixer()

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
        sample_rate = session.metadata.get("input_sample_rate", 16000)
        output_rate = session.metadata.get("output_sample_rate", sample_rate)
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
                sample_rate=sample_rate,
                output_sample_rate=output_rate,
            )
            if self._config.mixing_strategy == "mix":
                self._mix_buffers.setdefault(room_id, _MixingBuffer())
            logger.info(
                "Bridge: added session %s to room %s (%d participants, in=%d out=%d)",
                session.id,
                room_id,
                len(room),
                sample_rate,
                output_rate,
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
                # Clean up mixing buffer entry
                buf = self._mix_buffers.get(found_room_id)
                if buf is not None:
                    buf.frames.pop(session_id, None)
                if not self._rooms[found_room_id]:
                    del self._rooms[found_room_id]
                    self._mix_buffers.pop(found_room_id, None)
                logger.info(
                    "Bridge: removed session %s from room %s",
                    session_id,
                    found_room_id,
                )

    def forward(self, session: VoiceSession, frame: AudioFrame) -> None:
        """Forward audio from this session to other sessions in the room.

        For ``"forward"`` strategy: sends the source frame directly to
        every other session.  For ``"mix"`` strategy: stores the frame
        and sends each target a mix of all *other* sessions' latest
        frames (excluding the target's own audio).

        Args:
            session: The source session.
            frame: The processed audio frame to forward.
        """
        if not self._config.enabled:
            return

        # Pre-forward filter (BEFORE_BRIDGE_AUDIO)
        if self._frame_filter is not None:
            filtered = self._frame_filter(session, frame)
            if filtered is None:
                return  # frame dropped
            frame = filtered

        if self._config.mixing_strategy == "mix":
            self._forward_mix(session, frame)
        else:
            self._forward_direct(session, frame)

    def get_participant_count(self, room_id: str) -> int:
        """Return the number of bridged sessions in a room."""
        with self._lock:
            room = self._rooms.get(room_id)
            return len(room) if room else 0

    def get_bridged_sessions(self, room_id: str) -> list[tuple[VoiceSession, VoiceBackend]]:
        """Return ``(session, backend)`` pairs for all sessions in a room."""
        with self._lock:
            room = self._rooms.get(room_id)
            if not room:
                return []
            return [(bs.session, bs.backend) for bs in room.values()]

    def get_session_backend(self, session_id: str) -> VoiceBackend | None:
        """Return the backend registered for a bridged session, or ``None``."""
        with self._lock:
            for room in self._rooms.values():
                bs = room.get(session_id)
                if bs is not None:
                    return bs.backend
        return None

    def close(self) -> None:
        """Remove all sessions and clean up."""
        with self._lock:
            self._rooms.clear()
            self._mix_buffers.clear()

    # ------------------------------------------------------------------
    # Forward strategy: send source frame directly to each target
    # ------------------------------------------------------------------

    def _forward_direct(self, session: VoiceSession, frame: AudioFrame) -> None:
        with self._lock:
            targets = self._get_targets(session.id)

        if not targets:
            return

        for target in targets:
            self._send_to_target(session, target, frame)

    # ------------------------------------------------------------------
    # Mix strategy: store frame, send mixed audio to each target
    # ------------------------------------------------------------------

    def _forward_mix(self, session: VoiceSession, frame: AudioFrame) -> None:
        with self._lock:
            room_id = self._find_room_id(session.id)
            if room_id is None:
                return
            buf = self._mix_buffers.get(room_id)
            if buf is None:
                return
            # Store latest frame from this source
            buf.frames[session.id] = frame
            # Collect targets and their mix frames
            room = self._rooms.get(room_id, {})
            mix_jobs: list[tuple[_BridgedSession, list[AudioFrame]]] = []
            for sid, bs in room.items():
                if sid == session.id:
                    continue
                # Collect frames from all sessions except this target
                others = [f for s, f in buf.frames.items() if s != sid]
                if others:
                    mix_jobs.append((bs, others))

        for target, frames_to_mix in mix_jobs:
            # Resample all source frames to the target's output rate before
            # mixing so that samples are temporally aligned.
            target_rate = target.output_sample_rate
            resampled = [
                self._resample(f, target_rate) if f.sample_rate != target_rate else f
                for f in frames_to_mix
            ]
            mixed = resampled[0] if len(resampled) == 1 else self._mixer.mix(resampled)
            self._send_to_target(session, target, mixed)

    # ------------------------------------------------------------------
    # Shared send logic
    # ------------------------------------------------------------------

    def _send_to_target(
        self,
        source: VoiceSession,
        target: _BridgedSession,
        frame: AudioFrame,
    ) -> None:
        try:
            # Always resample to the target's output rate before processing
            target_rate = target.output_sample_rate
            if frame.sample_rate != target_rate:
                frame = self._resample(frame, target_rate)
            if self._frame_processor is not None:
                chunk = self._frame_processor(target.session, frame)
            else:
                chunk = self._frame_to_chunk(frame)
            target.backend.send_audio_sync(target.session, chunk)
        except Exception:
            logger.exception(
                "Bridge: failed to forward audio from %s to %s",
                source.id,
                target.session.id,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_room_id(self, session_id: str) -> str | None:
        """Find room for a session (caller holds lock)."""
        for room_id, room in self._rooms.items():
            if session_id in room:
                return room_id
        return None

    def _get_targets(self, source_id: str) -> list[_BridgedSession]:
        """Get target sessions for forwarding (caller holds lock)."""
        for room in self._rooms.values():
            if source_id in room:
                return [bs for sid, bs in room.items() if sid != source_id]
        return []

    @staticmethod
    def _resample(frame: AudioFrame, target_rate: int) -> AudioFrame:
        """Resample a frame to the target sample rate using the linear resampler."""
        return _get_resampler().resample(
            frame,
            target_rate=target_rate,
            target_channels=frame.channels,
            target_width=frame.sample_width,
        )

    @staticmethod
    def _frame_to_chunk(frame: AudioFrame) -> AudioChunk:
        """Convert an AudioFrame to an AudioChunk for send_audio_sync."""
        from roomkit.voice.base import AudioChunk

        return AudioChunk(
            data=frame.data,
            sample_rate=frame.sample_rate,
            channels=frame.channels,
        )


# ======================================================================
# Cached singletons (module-level, lazy)
# ======================================================================

_resampler_lock = threading.Lock()
_resampler_instance: ResamplerProvider | None = None


def _get_resampler() -> ResamplerProvider:
    """Return the best available resampler: NumPy if installed, else pure Python.

    Uses double-checked locking so the fast path (already initialized) is
    lock-free, while concurrent first calls are serialized.
    """
    global _resampler_instance  # noqa: PLW0603
    if _resampler_instance is not None:
        return _resampler_instance
    with _resampler_lock:
        if _resampler_instance is not None:
            return _resampler_instance
        try:
            from roomkit.voice.pipeline.resampler.numpy import NumpyResamplerProvider

            _resampler_instance = NumpyResamplerProvider()
        except ImportError:
            from roomkit.voice.pipeline.resampler.linear import LinearResamplerProvider

            _resampler_instance = LinearResamplerProvider()
        return _resampler_instance


def _get_default_mixer() -> MixerProvider:
    """Return the best available mixer: NumPy if installed, else pure Python."""
    try:
        from roomkit.voice.pipeline.mixer.numpy import NumpyMixerProvider

        return NumpyMixerProvider()
    except ImportError:
        from roomkit.voice.pipeline.mixer.python import PythonMixerProvider

        return PythonMixerProvider()
