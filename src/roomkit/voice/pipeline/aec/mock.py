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
        self.reset_count = 0
        self.closed = False

    @property
    def name(self) -> str:
        return "MockAECProvider"

    def process(self, frame: AudioFrame) -> AudioFrame:
        self.frames.append(frame)
        return frame

    def feed_reference(self, frame: AudioFrame) -> None:
        self.reference_frames.append(frame)

    def reset(self) -> None:
        self.reset_count += 1

    def close(self) -> None:
        self.closed = True
