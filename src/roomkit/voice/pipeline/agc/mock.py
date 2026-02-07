"""Mock AGC provider for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit.voice.pipeline.agc.base import AGCProvider

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame


class MockAGCProvider(AGCProvider):
    """Mock AGC provider that passes frames through unchanged."""

    def __init__(self) -> None:
        self.frames: list[AudioFrame] = []
        self.reset_count = 0
        self.closed = False

    @property
    def name(self) -> str:
        return "MockAGCProvider"

    def process(self, frame: AudioFrame) -> AudioFrame:
        self.frames.append(frame)
        return frame

    def reset(self) -> None:
        self.reset_count += 1

    def close(self) -> None:
        self.closed = True
