"""Mock capture source for tests and examples — no audio device involved."""

from __future__ import annotations

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.capture.base import DEFAULT_BACKLOG_SECONDS, AudioCaptureSource


class MockCaptureSource(AudioCaptureSource):
    """Capture source driven by explicit ``feed()`` calls.

    Frames are dispatched on the calling thread, which stands in for the
    capture thread.  Tests that need to exercise the catch-up path can feed
    from a second thread while a subscriber is replaying.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        block_duration_ms: int = 20,
        backlog_seconds: float = DEFAULT_BACKLOG_SECONDS,
        max_backlog_bytes: int | None = None,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            channels=channels,
            sample_width=2,
            block_duration_ms=block_duration_ms,
            backlog_seconds=backlog_seconds,
            max_backlog_bytes=max_backlog_bytes,
        )
        self.started = False
        """True between ``start()`` and ``stop()``."""

        self.start_count = 0
        """How many times ``start()`` actually acquired the device."""

    @property
    def block_bytes(self) -> int:
        """Byte length of one nominal block at this source's format."""
        samples = int(self._sample_rate * self._block_duration_ms / 1000)
        return samples * self._channels * self._sample_width

    def start(self) -> None:
        if self.started:
            return
        self.started = True
        self.start_count += 1

    def stop(self) -> None:
        self.started = False

    def feed(self, data: bytes | AudioFrame) -> None:
        """Dispatch one frame, as a capture device would."""
        frame = (
            data
            if isinstance(data, AudioFrame)
            else AudioFrame(
                data=data,
                sample_rate=self._sample_rate,
                channels=self._channels,
                sample_width=self._sample_width,
            )
        )
        self._dispatch(frame)

    def feed_blocks(self, count: int, *, fill: int = 0) -> list[bytes]:
        """Dispatch ``count`` blocks of identical filler, returning what was sent.

        Each block carries a distinct byte value derived from its index, so
        tests can assert on exact ordering rather than on totals alone.
        """
        sent: list[bytes] = []
        for i in range(count):
            payload = bytes([(fill + i) % 256]) * self.block_bytes
            sent.append(payload)
            self.feed(payload)
        return sent
