"""AudioPace: at most a burst of audio at once, then a bounded catch-up speed
(RFC §12.2, RMK-581)."""

from __future__ import annotations

import pytest

from roomkit.voice.stt._pacing import AudioPace


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_a_backlog_goes_out_as_a_burst_then_at_the_catch_up_speed() -> None:
    clock = _Clock()
    pace = AudioPace(burst=3.0, speed=2.0, clock=clock)

    assert [pace.delay(1.0) for _ in range(3)] == [0.0, 0.0, 0.0]
    # Past the burst, a second of audio waits half a second at twice real time.
    assert pace.delay(1.0) == pytest.approx(0.5)
    clock.now += 0.5
    assert pace.delay(1.0) == pytest.approx(0.5)


def test_live_audio_never_waits() -> None:
    clock = _Clock()
    pace = AudioPace(burst=3.0, speed=2.0, clock=clock)
    pace.delay(3.0)  # a burst spent on a backlog

    delays = []
    for _ in range(50):
        clock.now += 0.1
        delays.append(pace.delay(0.1))

    assert delays[-40:] == [0.0] * 40


def test_an_idle_stream_saves_no_more_than_one_burst() -> None:
    clock = _Clock()
    pace = AudioPace(burst=3.0, speed=2.0, clock=clock)
    clock.now += 60.0

    assert pace.delay(4.0) == pytest.approx(0.5)
