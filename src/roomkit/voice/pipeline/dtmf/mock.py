"""Mock DTMF detector for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit.voice.pipeline.dtmf.base import DTMFDetector, DTMFEvent

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame


class MockDTMFDetector(DTMFDetector):
    """Mock DTMF detector that returns a preconfigured sequence of events."""

    def __init__(self, events: list[DTMFEvent | None] | None = None) -> None:
        self._events = events or []
        self._index = 0
        self.frames: list[AudioFrame] = []
        self.reset_count = 0
        self.closed = False

    @property
    def name(self) -> str:
        return "MockDTMFDetector"

    def process(self, frame: AudioFrame) -> DTMFEvent | None:
        self.frames.append(frame)
        if self._index < len(self._events):
            event = self._events[self._index]
            self._index += 1
            return event
        return None

    def reset(self) -> None:
        self._index = 0
        self.reset_count += 1

    def close(self) -> None:
        self.closed = True
