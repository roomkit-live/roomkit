"""Mock denoiser provider for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit.voice.pipeline.denoiser.base import DenoiserProvider

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame


class MockDenoiserProvider(DenoiserProvider):
    """Mock denoiser that passes frames through unchanged.

    Tracks processed frames for test assertions.
    """

    def __init__(self) -> None:
        self.frames: list[AudioFrame] = []
        self.reset_count = 0
        self.closed = False

    @property
    def name(self) -> str:
        return "MockDenoiserProvider"

    def process(self, frame: AudioFrame) -> AudioFrame:
        self.frames.append(frame)
        return frame

    def reset(self) -> None:
        self.reset_count += 1

    def close(self) -> None:
        self.closed = True
