"""Tests for AudioCaptureSource — ring, marks, fan-out and catch-up.

See RFC Section 12.12.
"""

from __future__ import annotations

import logging
import threading
import time

import pytest

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.capture import CaptureMark, MockCaptureSource


def _source(**kwargs: object) -> MockCaptureSource:
    source = MockCaptureSource(**kwargs)  # type: ignore[arg-type]
    source.start()
    return source


class TestLifecycle:
    def test_start_is_idempotent(self) -> None:
        source = MockCaptureSource()
        source.start()
        source.start()
        assert source.started is True
        assert source.start_count == 1

    def test_stop_is_idempotent(self) -> None:
        source = _source()
        source.stop()
        source.stop()
        assert source.started is False

    def test_losing_every_subscriber_does_not_stop_capture(self) -> None:
        """Lifetime is explicit: a detector may detach without ending capture."""
        source = _source()
        subscription = source.subscribe(lambda frame: None)
        subscription.unsubscribe()

        assert source.started is True
        # And the source keeps filling its ring for the next subscriber.
        mark = source.mark()
        sent = source.feed_blocks(2)
        received: list[bytes] = []
        source.subscribe(lambda frame: received.append(frame.data), since=mark)
        assert received == sent

    def test_close_deactivates_subscriptions(self) -> None:
        source = _source()
        subscription = source.subscribe(lambda frame: None)
        source.close()

        assert subscription.active is False
        assert source.started is False


class TestFanOut:
    def test_every_subscriber_receives_every_frame(self) -> None:
        source = _source()
        first: list[bytes] = []
        second: list[bytes] = []
        source.subscribe(lambda f: first.append(f.data))
        source.subscribe(lambda f: second.append(f.data))

        sent = source.feed_blocks(3)

        assert first == sent
        assert second == sent

    def test_unsubscribe_stops_delivery(self) -> None:
        source = _source()
        received: list[bytes] = []
        subscription = source.subscribe(lambda f: received.append(f.data))

        source.feed_blocks(1)
        subscription.unsubscribe()
        source.feed_blocks(1, fill=100)

        assert len(received) == 1
        assert subscription.active is False

    def test_unsubscribe_is_idempotent(self) -> None:
        source = _source()
        subscription = source.subscribe(lambda f: None)
        subscription.unsubscribe()
        subscription.unsubscribe()
        assert subscription.active is False

    def test_a_raising_subscriber_does_not_break_the_others(self) -> None:
        source = _source()
        survivor: list[bytes] = []

        def explode(frame: AudioFrame) -> None:
            raise RuntimeError("boom")

        source.subscribe(explode, name="broken")
        source.subscribe(lambda f: survivor.append(f.data), name="fine")

        sent = source.feed_blocks(2)

        assert survivor == sent

    def test_subscribing_without_a_mark_delivers_only_live_audio(self) -> None:
        source = _source()
        source.feed_blocks(3)  # before anyone is listening

        received: list[bytes] = []
        source.subscribe(lambda f: received.append(f.data))
        live = source.feed_blocks(1, fill=200)

        assert received == live


class TestRing:
    def test_ring_evicts_oldest_first_under_its_byte_bound(self) -> None:
        source = _source(max_backlog_bytes=MockCaptureSource().block_bytes * 3)
        mark = source.mark()
        sent = source.feed_blocks(6)

        received: list[bytes] = []
        subscription = source.subscribe(lambda f: received.append(f.data), since=mark)

        assert received == sent[-3:]
        assert subscription.truncated is True

    def test_backlog_seconds_bounds_the_ring(self) -> None:
        # 100 ms of 20 ms blocks = 5 blocks retained.
        source = _source(backlog_seconds=0.1)
        mark = source.mark()
        sent = source.feed_blocks(8)

        received: list[bytes] = []
        source.subscribe(lambda f: received.append(f.data), since=mark)

        assert received == sent[-5:]


class TestReplay:
    def test_mark_then_subscribe_replays_the_exact_bytes_in_order(self) -> None:
        source = _source()
        source.feed_blocks(2, fill=1)  # before the mark — must not be replayed
        mark = source.mark()
        phrase = source.feed_blocks(4, fill=50)

        received: list[bytes] = []
        subscription = source.subscribe(lambda f: received.append(f.data), since=mark)

        assert received == phrase
        assert subscription.replayed_bytes == sum(len(b) for b in phrase)
        assert subscription.truncated is False

    def test_replay_is_followed_by_live_audio_on_the_same_subscriber(self) -> None:
        source = _source()
        mark = source.mark()
        backlog = source.feed_blocks(2, fill=10)

        received: list[bytes] = []
        source.subscribe(lambda f: received.append(f.data), since=mark)
        live = source.feed_blocks(2, fill=90)

        assert received == backlog + live

    def test_a_mark_with_nothing_behind_it_replays_nothing(self) -> None:
        source = _source()
        mark = source.mark()

        received: list[bytes] = []
        subscription = source.subscribe(lambda f: received.append(f.data), since=mark)

        assert received == []
        assert subscription.replayed_bytes == 0
        assert subscription.truncated is False

    def test_stale_mark_warns_rather_than_raising(self, caplog: pytest.LogCaptureFixture) -> None:
        """Raising here would discard the utterance the backlog exists to keep."""
        source = _source(max_backlog_bytes=MockCaptureSource().block_bytes * 2)
        mark = source.mark()
        source.feed_blocks(5)

        with caplog.at_level(logging.WARNING, logger="roomkit.voice.capture"):
            subscription = source.subscribe(lambda f: None, since=mark, name="wakeword")

        assert subscription.truncated is True
        assert "already been evicted" in caplog.text
        assert "wakeword" in caplog.text

    def test_a_mark_from_another_source_is_rejected(self) -> None:
        source = _source()
        foreign = MockCaptureSource().mark()

        with pytest.raises(ValueError, match="different capture source"):
            source.subscribe(lambda f: None, since=foreign)

    def test_live_frames_never_interleave_with_the_replay(self) -> None:
        """Frames captured mid-replay queue behind it instead of jumping ahead."""
        source = _source()
        mark = source.mark()
        backlog = source.feed_blocks(4, fill=10)
        live = b"\xff" * source.block_bytes

        received: list[bytes] = []
        injected = threading.Event()

        def subscriber(frame: AudioFrame) -> None:
            received.append(frame.data)
            if len(received) == 1:
                # Simulate the capture thread delivering while replay runs.
                feeder = threading.Thread(target=source.feed, args=(live,))
                feeder.start()
                feeder.join()
                injected.set()

        subscription = source.subscribe(subscriber, since=mark)

        assert injected.is_set(), "the mid-replay frame was never fed"
        assert received == [*backlog, live]
        # The queued live frame is not backlog, so it is not counted as replayed.
        assert subscription.replayed_bytes == sum(len(b) for b in backlog)

    def test_delivery_goes_direct_once_the_queue_drains(self) -> None:
        source = _source()
        mark = source.mark()
        source.feed_blocks(2)

        received: list[bytes] = []
        source.subscribe(lambda f: received.append(f.data), since=mark)
        before = len(received)
        tail = source.feed_blocks(1, fill=7)

        assert received[before:] == tail


class TestSubscriberContract:
    def test_a_slow_subscriber_is_reported(self, caplog: pytest.LogCaptureFixture) -> None:
        """Otherwise the only symptom is crackling audio, with no named cause."""
        source = _source(block_duration_ms=20)

        def slow(frame: AudioFrame) -> None:
            time.sleep(0.05)

        source.subscribe(slow, name="wakeword")

        with caplog.at_level(logging.WARNING, logger="roomkit.voice.capture"):
            source.feed_blocks(1)

        assert "wakeword" in caplog.text
        assert "enqueue the frame and return" in caplog.text

    def test_a_prompt_subscriber_is_not_reported(self, caplog: pytest.LogCaptureFixture) -> None:
        source = _source(block_duration_ms=20)
        source.subscribe(lambda f: None, name="fast")

        with caplog.at_level(logging.WARNING, logger="roomkit.voice.capture"):
            source.feed_blocks(3)

        assert caplog.text == ""


class TestFormat:
    def test_frames_carry_the_source_format(self) -> None:
        source = _source(sample_rate=48000, channels=2)
        received: list[AudioFrame] = []
        source.subscribe(received.append)

        source.feed_blocks(1)

        assert received[0].sample_rate == 48000
        assert received[0].channels == 2
        assert received[0].sample_width == 2

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"sample_rate": 0},
            {"channels": 0},
            {"block_duration_ms": 0},
            {"backlog_seconds": 0},
            {"max_backlog_bytes": 0},
        ],
    )
    def test_invalid_configuration_is_rejected(self, kwargs: dict[str, int]) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            MockCaptureSource(**kwargs)  # type: ignore[arg-type]

    def test_mark_is_opaque_but_comparable(self) -> None:
        source = _source()
        first = source.mark()
        source.feed_blocks(1)
        second = source.mark()

        assert isinstance(first, CaptureMark)
        assert first != second
