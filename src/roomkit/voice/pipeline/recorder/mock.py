"""Mock audio recorder for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit.voice.pipeline.recorder.base import (
    AudioRecorder,
    RecordingConfig,
    RecordingHandle,
    RecordingResult,
)

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame
    from roomkit.voice.base import VoiceSession


class MockAudioRecorder(AudioRecorder):
    """Mock audio recorder that tracks calls."""

    def __init__(self) -> None:
        self.started: list[tuple[str, RecordingConfig]] = []
        self.stopped: list[RecordingHandle] = []
        self.inbound_frames: list[tuple[str, AudioFrame]] = []
        self.outbound_frames: list[tuple[str, AudioFrame]] = []
        self.reset_count = 0
        self.closed = False
        self._next_id = 0

    @property
    def name(self) -> str:
        return "MockAudioRecorder"

    def start(self, session: VoiceSession, config: RecordingConfig) -> RecordingHandle:
        self._next_id += 1
        handle = RecordingHandle(
            id=f"rec_{self._next_id}",
            session_id=session.id,
            path=f"/tmp/recording_{self._next_id}.wav",  # nosec B108
        )
        self.started.append((session.id, config))
        return handle

    def stop(self, handle: RecordingHandle) -> RecordingResult:
        self.stopped.append(handle)
        return RecordingResult(
            id=handle.id,
            urls=[handle.path] if handle.path else [],
            duration_seconds=1.0,
            size_bytes=32000,
        )

    def tap_inbound(self, handle: RecordingHandle, frame: AudioFrame) -> None:
        self.inbound_frames.append((handle.id, frame))

    def tap_outbound(self, handle: RecordingHandle, frame: AudioFrame) -> None:
        self.outbound_frames.append((handle.id, frame))

    def reset(self) -> None:
        self.reset_count += 1

    def close(self) -> None:
        self.closed = True
