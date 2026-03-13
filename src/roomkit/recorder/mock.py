"""Mock media recorder for testing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from roomkit.recorder.base import (
    MediaRecorder,
    MediaRecordingConfig,
    MediaRecordingHandle,
    MediaRecordingResult,
    RecordingTrack,
)


@dataclass
class MockDataChunk:
    """A recorded data chunk for test inspection."""

    track_id: str
    data: bytes
    timestamp_ms: float | None


class MockMediaRecorder(MediaRecorder):
    """In-memory media recorder for testing.

    Stores all tracks and data chunks so tests can inspect what was recorded.
    """

    @property
    def name(self) -> str:
        return "mock"

    def __init__(self) -> None:
        self.tracks: list[RecordingTrack] = []
        self.chunks: list[MockDataChunk] = []
        self.handles: list[MediaRecordingHandle] = []
        self.results: list[MediaRecordingResult] = []
        self.closed: bool = False

    def on_recording_start(self, config: MediaRecordingConfig) -> MediaRecordingHandle:
        handle = MediaRecordingHandle(
            id=uuid4().hex[:12],
            room_id="",
            state="recording",
            started_at=datetime.now(UTC),
        )
        self.handles.append(handle)
        return handle

    def on_recording_stop(self, handle: MediaRecordingHandle) -> MediaRecordingResult:
        handle.state = "stopped"
        result = MediaRecordingResult(
            id=handle.id,
            duration_seconds=0.0,
            tracks=list(self.tracks),
            format="mp4",
            size_bytes=sum(len(c.data) for c in self.chunks),
        )
        self.results.append(result)
        return result

    def on_track_added(self, handle: MediaRecordingHandle, track: RecordingTrack) -> None:
        self.tracks.append(track)

    def on_track_removed(self, handle: MediaRecordingHandle, track: RecordingTrack) -> None:
        self.tracks = [t for t in self.tracks if t.id != track.id]

    def on_data(
        self,
        handle: MediaRecordingHandle,
        track: RecordingTrack,
        data: bytes,
        timestamp_ms: float | None,
    ) -> None:
        self.chunks.append(MockDataChunk(track_id=track.id, data=data, timestamp_ms=timestamp_ms))

    def close(self) -> None:
        self.closed = True
