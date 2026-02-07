"""Mock VAD provider for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit.voice.pipeline.vad.base import VADEvent, VADProvider

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame


class MockVADProvider(VADProvider):
    """Mock VAD provider that returns a preconfigured sequence of events.

    Example:
        from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType

        events = [
            VADEvent(type=VADEventType.SPEECH_START),
            None,
            VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio"),
        ]
        vad = MockVADProvider(events=events)
    """

    def __init__(self, events: list[VADEvent | None] | None = None) -> None:
        self._events = events or []
        self._index = 0
        self.frames: list[AudioFrame] = []
        self.reset_count = 0
        self.closed = False

    @property
    def name(self) -> str:
        return "MockVADProvider"

    def process(self, frame: AudioFrame) -> VADEvent | None:
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
