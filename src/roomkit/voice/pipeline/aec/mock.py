"""Mock AEC provider for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit.voice.pipeline.aec.base import AECProvider

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame


class MockAECProvider(AECProvider):
    """Mock AEC provider that passes frames through unchanged."""

    def __init__(self) -> None:
        self.frames: list[AudioFrame] = []
        self.reference_frames: list[AudioFrame] = []
        self.streams: list[str] = []
        self.reference_streams: list[str] = []
        self.reset_streams: list[str] = []
        self.reset_count = 0
        self.closed = False

    @property
    def name(self) -> str:
        return "MockAECProvider"

    def process(self, frame: AudioFrame, stream: str) -> AudioFrame:
        self.frames.append(frame)
        self.streams.append(stream)
        return frame

    def feed_reference(self, frame: AudioFrame, stream: str) -> None:
        self.reference_frames.append(frame)
        self.reference_streams.append(stream)

    def reset(self, stream: str) -> None:
        self.reset_count += 1
        self.reset_streams.append(stream)

    def close(self) -> None:
        self.closed = True
