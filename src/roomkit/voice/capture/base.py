"""Audio capture source ABC — device ownership detached from voice sessions.

A ``VoiceBackend`` acquires its capture device when a session starts and
releases it when that session ends.  That is the right lifetime for a call and
the wrong one for anything that must listen *before* a session exists — a wake
word, a level meter, a hotkey that arms on speech.  Such a consumer would hold
the device, hand it back so the session can take it, and the reacquisition
lands precisely while the person is still speaking.

An ``AudioCaptureSource`` owns the device instead, and a session becomes one
subscriber among several.  See RFC Section 12.12.

Subclasses implement device acquisition only: ``start()``, ``stop()``, and a
call to :meth:`AudioCaptureSource._dispatch` for each captured frame.  The ring
buffer, the marks, the fan-out and the catch-up protocol live here.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic

from roomkit.voice.audio_frame import AudioFrame

logger = logging.getLogger("roomkit.voice.capture")

CaptureFrameCallback = Callable[[AudioFrame], None]
"""Called once per captured frame, synchronously on the capture thread."""

DEFAULT_BACKLOG_SECONDS = 10.0
"""How much audio a source retains for replay, unless configured otherwise."""

_SLOW_CALLBACK_RATIO = 0.5
"""Fraction of a block a subscriber callback may take before it is flagged."""

_SLOW_LOG_INTERVAL_SECONDS = 5.0
"""Minimum spacing between slow-subscriber warnings, per subscriber."""


@dataclass(frozen=True)
class CaptureMark:
    """A position in a source's backlog.

    Opaque: the fields are an implementation detail and MUST NOT be
    interpreted by callers.  Obtain one from :meth:`AudioCaptureSource.mark`
    and hand it back to :meth:`AudioCaptureSource.subscribe`.
    """

    sequence: int
    """Index of the next frame the source will capture."""

    source_id: int
    """Identity of the issuing source, so a foreign mark is rejected."""


class CaptureSubscription:
    """Handle on one subscriber's attachment to a capture source."""

    def __init__(
        self,
        source: AudioCaptureSource,
        subscriber: _Subscriber,
        name: str | None,
    ) -> None:
        self._source = source
        self._subscriber = subscriber
        self.name = name
        """Diagnostic label, as passed to ``subscribe()``."""

        self.replayed_bytes = 0
        """Bytes delivered from the backlog — audio captured before subscribing."""

        self.truncated = False
        """True when the mark had already been evicted and replay is partial."""

    @property
    def active(self) -> bool:
        """False once :meth:`unsubscribe` has been called."""
        return self._subscriber.active

    def unsubscribe(self) -> None:
        """Detach.  Idempotent, and never stops the source itself."""
        self._source._remove_subscriber(self._subscriber)


class _Subscriber:
    """Internal per-subscriber state.  Not part of the public surface."""

    __slots__ = ("active", "callback", "name", "pending", "slow_count", "slow_logged_at")

    def __init__(self, callback: CaptureFrameCallback, name: str | None) -> None:
        self.callback = callback
        self.name = name
        self.active = True
        self.pending: deque[AudioFrame] | None = None
        """Non-None while catching up: the fan-out queues here instead of calling."""
        self.slow_count = 0
        self.slow_logged_at = 0.0

    @property
    def label(self) -> str:
        return self.name or "<unnamed>"


class AudioCaptureSource(ABC):
    """A continuous source of ``AudioFrame`` shared by several consumers.

    Lifetime is explicit: ``start()`` and ``stop()`` are the only things that
    acquire and release the device.  Dropping to zero subscribers does not stop
    capture, and gaining one does not start it — which is what makes it safe
    for a wake-word detector to detach for the duration of a session.

    Frames are emitted **raw**.  Echo cancellation is per-session (it needs the
    reference signal of what that session plays) and belongs downstream, in the
    backend, after fan-out.  A subscriber still attached while a session is
    playing hears the far end unattenuated.  Resampling is likewise the
    subscriber's concern: a source has one format.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
        block_duration_ms: int = 20,
        backlog_seconds: float = DEFAULT_BACKLOG_SECONDS,
        max_backlog_bytes: int | None = None,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")
        if sample_width <= 0:
            raise ValueError("sample_width must be positive")
        if block_duration_ms <= 0:
            raise ValueError("block_duration_ms must be positive")
        if backlog_seconds <= 0:
            raise ValueError("backlog_seconds must be positive")
        if max_backlog_bytes is not None and max_backlog_bytes <= 0:
            raise ValueError("max_backlog_bytes must be positive")

        self._sample_rate = sample_rate
        self._channels = channels
        self._sample_width = sample_width
        self._block_duration_ms = block_duration_ms

        by_duration = int(backlog_seconds * sample_rate * channels * sample_width)
        self._max_ring_bytes = min(by_duration, max_backlog_bytes or by_duration)
        self._slow_threshold_s = (block_duration_ms / 1000) * _SLOW_CALLBACK_RATIO

        self._lock = threading.Lock()
        self._ring: deque[tuple[int, AudioFrame]] = deque()
        self._ring_bytes = 0
        self._sequence = 0
        self._subscribers: tuple[_Subscriber, ...] = ()

    # -------------------------------------------------------------------------
    # Format
    # -------------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        """Sample rate of every frame this source emits."""
        return self._sample_rate

    @property
    def channels(self) -> int:
        """Channel count of every frame this source emits."""
        return self._channels

    @property
    def sample_width(self) -> int:
        """Bytes per sample of every frame this source emits."""
        return self._sample_width

    @property
    def block_duration_ms(self) -> int:
        """Nominal duration of one captured block, in milliseconds."""
        return self._block_duration_ms

    @property
    def input_latency_ms(self) -> float | None:
        """Device input latency, when the implementation can report one.

        Used to seed AEC stream delay.  ``None`` means unknown.
        """
        return None

    # -------------------------------------------------------------------------
    # Lifecycle — implemented by subclasses
    # -------------------------------------------------------------------------

    @abstractmethod
    def start(self) -> None:
        """Acquire the device and begin capturing.  Idempotent."""

    @abstractmethod
    def stop(self) -> None:
        """Release the device and stop capturing.  Idempotent."""

    def close(self) -> None:
        """Stop capturing and drop every subscriber.  Idempotent."""
        self.stop()
        with self._lock:
            subscribers = self._subscribers
            self._subscribers = ()
            self._ring.clear()
            self._ring_bytes = 0
        for sub in subscribers:
            sub.active = False

    # -------------------------------------------------------------------------
    # Marks and subscription
    # -------------------------------------------------------------------------

    def mark(self) -> CaptureMark:
        """Name the current position in the backlog, for later replay."""
        with self._lock:
            return CaptureMark(sequence=self._sequence, source_id=id(self))

    def subscribe(
        self,
        callback: CaptureFrameCallback,
        *,
        since: CaptureMark | None = None,
        name: str | None = None,
    ) -> CaptureSubscription:
        """Attach a consumer, optionally replaying the backlog from ``since``.

        The callback runs **synchronously on the capture thread**.  It MUST NOT
        perform unbounded work: enqueue the frame and return.  The source
        guarantees no isolation between subscribers — one slow subscriber
        degrades capture for all of them.

        Ordering is total: every replayed frame is delivered before every live
        one, including frames captured during the replay itself.

        Args:
            callback: Invoked once per frame.
            since: Replay from this mark before going live.  A mark whose
                position has been evicted replays what remains and sets
                ``truncated`` on the returned subscription.
            name: Diagnostic label used in logs and slow-subscriber warnings.

        Returns:
            The subscription handle.

        Raises:
            ValueError: If ``since`` was issued by a different source.
        """
        if since is not None and since.source_id != id(self):
            raise ValueError("CaptureMark was issued by a different capture source")

        subscriber = _Subscriber(callback, name)
        subscription = CaptureSubscription(self, subscriber, name)

        with self._lock:
            backlog: list[AudioFrame] = []
            if since is not None:
                backlog, subscription.truncated = self._slice_locked(since)
                # Live frames queue here until the replay has caught up.
                subscriber.pending = deque()
            self._subscribers = (*self._subscribers, subscriber)

        if since is None:
            return subscription

        if subscription.truncated:
            logger.warning(
                "Capture mark for subscriber %s had already been evicted; "
                "replaying %d frames instead of the full backlog",
                subscriber.label,
                len(backlog),
            )
        self._replay(subscriber, subscription, backlog)
        logger.info(
            "Capture subscriber %s caught up: %d bytes replayed%s",
            subscriber.label,
            subscription.replayed_bytes,
            " (truncated)" if subscription.truncated else "",
        )
        return subscription

    # -------------------------------------------------------------------------
    # Frame ingress — called by subclasses from the capture thread
    # -------------------------------------------------------------------------

    def _dispatch(self, frame: AudioFrame) -> None:
        """Record a captured frame and fan it out.  Call from the capture thread."""
        with self._lock:
            self._ring.append((self._sequence, frame))
            self._ring_bytes += len(frame.data)
            self._sequence += 1
            self._trim_locked()

            # Deciding queue-vs-live under the same lock as the ring write is
            # what keeps replay strictly ahead of live for a catching-up
            # subscriber.  The callbacks themselves run outside the lock.
            live: list[_Subscriber] = []
            for sub in self._subscribers:
                if sub.pending is not None:
                    sub.pending.append(frame)
                else:
                    live.append(sub)

        for sub in live:
            self._invoke(sub, frame)

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _trim_locked(self) -> None:
        while self._ring and self._ring_bytes > self._max_ring_bytes:
            _, evicted = self._ring.popleft()
            self._ring_bytes -= len(evicted.data)

    def _slice_locked(self, mark: CaptureMark) -> tuple[list[AudioFrame], bool]:
        if not self._ring:
            # Nothing retained: truncated only if frames existed and were lost.
            return [], mark.sequence < self._sequence
        oldest = self._ring[0][0]
        truncated = mark.sequence < oldest
        start = max(mark.sequence, oldest)
        return [frame for seq, frame in self._ring if seq >= start], truncated

    def _replay(
        self,
        subscriber: _Subscriber,
        subscription: CaptureSubscription,
        backlog: list[AudioFrame],
    ) -> None:
        """Drain the backlog, then the queue, then switch to live delivery.

        Replay must not hold the fan-out lock: seconds of audio delivered under
        it would stall the capture thread.  So live frames queue while this
        runs, and the switch to direct delivery happens under the lock once the
        queue is empty — the same shape as the realtime channel's pre-connect
        flush.
        """
        for frame in backlog:
            if not subscriber.active:
                return
            subscription.replayed_bytes += len(frame.data)
            self._invoke(subscriber, frame)

        while True:
            with self._lock:
                pending = subscriber.pending
                if pending is None:
                    return
                if not pending:
                    # Atomically switch subsequent fan-out to direct delivery.
                    subscriber.pending = None
                    return
                batch = list(pending)
                pending.clear()

            for frame in batch:
                if not subscriber.active:
                    return
                self._invoke(subscriber, frame)

    def _invoke(self, subscriber: _Subscriber, frame: AudioFrame) -> None:
        started = monotonic()
        try:
            subscriber.callback(frame)
        except Exception:
            logger.exception("Capture subscriber %s raised; frame dropped", subscriber.label)
        self._check_slow(subscriber, monotonic() - started)

    def _check_slow(self, subscriber: _Subscriber, elapsed_s: float) -> None:
        """Make the no-unbounded-work contract observable.

        Without this the symptom of a slow subscriber is "the audio crackles",
        never "this subscriber is slow".
        """
        if elapsed_s <= self._slow_threshold_s:
            return
        subscriber.slow_count += 1
        now = monotonic()
        if now - subscriber.slow_logged_at < _SLOW_LOG_INTERVAL_SECONDS:
            return
        subscriber.slow_logged_at = now
        logger.warning(
            "Capture subscriber %s took %.1fms on a %dms block (%d slow callbacks so "
            "far); a subscriber must enqueue the frame and return",
            subscriber.label,
            elapsed_s * 1000,
            self._block_duration_ms,
            subscriber.slow_count,
        )

    def _remove_subscriber(self, subscriber: _Subscriber) -> None:
        with self._lock:
            if not subscriber.active:
                return
            subscriber.active = False
            self._subscribers = tuple(s for s in self._subscribers if s is not subscriber)
