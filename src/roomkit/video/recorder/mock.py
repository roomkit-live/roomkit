"""Mock video recorder for testing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from roomkit.video.recorder.base import (
    VideoRecorder,
    VideoRecordingConfig,
    VideoRecordingHandle,
    VideoRecordingResult,
)

if TYPE_CHECKING:
    from roomkit.video.base import VideoSession
    from roomkit.video.video_frame import VideoFrame


@dataclass
class MockRecordingEntry:
    """Tracked recording in MockVideoRecorder."""

    handle: VideoRecordingHandle
    frames: list[VideoFrame] = field(default_factory=list)


class MockVideoRecorder(VideoRecorder):
    """Mock video recorder that tracks frames in memory."""

    def __init__(self) -> None:
        self.recordings: dict[str, MockRecordingEntry] = {}
        self._counter = 0

    @property
    def name(self) -> str:
        return "MockVideoRecorder"

    def start(self, session: VideoSession, config: VideoRecordingConfig) -> VideoRecordingHandle:
        self._counter += 1
        rec_id = f"rec_{self._counter}"
        handle = VideoRecordingHandle(
            id=rec_id,
            session_id=session.id,
            started_at=datetime.now(UTC),
            path=f"{config.storage or 'mock'}/{rec_id}.{config.format}",
        )
        self.recordings[rec_id] = MockRecordingEntry(handle=handle)
        return handle

    def stop(self, handle: VideoRecordingHandle) -> VideoRecordingResult:
        handle.state = "stopped"
        entry = self.recordings.get(handle.id)
        frame_count = len(entry.frames) if entry else 0
        return VideoRecordingResult(
            id=handle.id,
            url=handle.path,
            frame_count=frame_count,
            duration_seconds=frame_count / 15.0,
            size_bytes=frame_count * 921_600,
        )

    def tap_frame(self, handle: VideoRecordingHandle, frame: VideoFrame) -> None:
        entry = self.recordings.get(handle.id)
        if entry is not None:
            entry.frames.append(frame)
