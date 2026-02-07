"""Mock resampler provider for testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from roomkit.voice.pipeline.resampler.base import ResamplerProvider

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame


@dataclass
class ResampleCall:
    """Record of a single resample() invocation."""

    frame: AudioFrame
    target_rate: int
    target_channels: int
    target_width: int


class MockResamplerProvider(ResamplerProvider):
    """Mock resampler that passes frames through unchanged and records calls."""

    def __init__(self) -> None:
        self.calls: list[ResampleCall] = []
        self.reset_count: int = 0
        self.closed: bool = False

    @property
    def name(self) -> str:
        return "mock"

    def resample(
        self,
        frame: AudioFrame,
        target_rate: int,
        target_channels: int,
        target_width: int,
    ) -> AudioFrame:
        self.calls.append(
            ResampleCall(
                frame=frame,
                target_rate=target_rate,
                target_channels=target_channels,
                target_width=target_width,
            )
        )
        return frame

    def reset(self) -> None:
        self.reset_count += 1

    def close(self) -> None:
        self.closed = True
