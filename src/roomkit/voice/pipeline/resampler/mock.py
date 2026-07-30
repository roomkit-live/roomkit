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
    stream: str


class MockResamplerProvider(ResamplerProvider):
    """Mock resampler that passes frames through unchanged and records calls.

    Records the stream key it was given, so a test can assert the pipeline
    threads it rather than assuming it does.
    """

    def __init__(self) -> None:
        self.calls: list[ResampleCall] = []
        self.reset_count: int = 0
        self.reset_streams: list[str | None] = []
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
        stream: str,
    ) -> AudioFrame:
        self.calls.append(
            ResampleCall(
                frame=frame,
                target_rate=target_rate,
                target_channels=target_channels,
                target_width=target_width,
                stream=stream,
            )
        )
        return frame

    def reset(self, stream: str | None = None) -> None:
        self.reset_count += 1
        self.reset_streams.append(stream)

    def close(self) -> None:
        self.closed = True
