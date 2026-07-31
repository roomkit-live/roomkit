"""The conference mixer: the windowed N→1 sum a realtime provider hears.

Deterministic through ``_mix_once`` — the ticker is a clock around that same
call, and racing wall-clock ticks in a test proves cadence, not mixing.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass

from roomkit.channels._conference_mixer import (
    _RING_WINDOWS,
    ConferenceMixer,
    mix_window,
)
from roomkit.voice.audio_frame import AudioFrame

RATE = 16000
WINDOW_SAMPLES = 320  # 20 ms at 16 kHz
WINDOW_BYTES = WINDOW_SAMPLES * 2
ROOM = "room-1"


@dataclass
class _Lane:
    room_id: str
    track_id: str


class _Capture:
    def __init__(self) -> None:
        self.sent: list[tuple[str, bytes]] = []

    async def __call__(self, room_id: str, data: bytes) -> None:
        self.sent.append((room_id, data))


def pcm(value: int, samples: int = WINDOW_SAMPLES) -> bytes:
    return value.to_bytes(2, "little", signed=True) * samples


def sample_of(data: bytes) -> int:
    return int.from_bytes(data[:2], "little", signed=True)


def make_mixer(capture: _Capture, *, input_sample_rate: int = RATE) -> ConferenceMixer:
    mixer = ConferenceMixer(send=capture)
    mixer.configure(input_sample_rate=input_sample_rate)
    return mixer


def feed_quietly(mixer: ConferenceMixer, lane: _Lane, data: bytes) -> None:
    """Feed a frame and stop the room's clock, so only _mix_once consumes."""
    mixer.feed(lane, AudioFrame(data=data, sample_rate=RATE))
    room = mixer._rooms[lane.room_id]
    if room.ticker is not None:
        room.ticker.cancel()
        room.ticker = None


class TestMixWindow:
    def test_a_lone_window_passes_through_untouched(self) -> None:
        window = pcm(1234)
        assert mix_window([window]) is window

    def test_two_windows_sum_with_sqrt2_headroom(self) -> None:
        mixed = mix_window([pcm(1000), pcm(2000)])
        assert sample_of(mixed) == int(3000 / math.sqrt(2))

    def test_three_windows_sum_with_sqrt3_headroom(self) -> None:
        mixed = mix_window([pcm(1000), pcm(2000), pcm(3000)])
        assert sample_of(mixed) == int(6000 / math.sqrt(3))

    def test_a_hot_sum_clamps_to_int16(self) -> None:
        mixed = mix_window([pcm(30000), pcm(30000), pcm(30000)])
        assert sample_of(mixed) == 32767

    def test_a_hot_negative_sum_clamps_to_int16(self) -> None:
        mixed = mix_window([pcm(-30000), pcm(-30000), pcm(-30000)])
        assert sample_of(mixed) == -32768


class TestMixing:
    async def test_one_track_reaches_the_provider_unscaled(self) -> None:
        capture = _Capture()
        mixer = make_mixer(capture)
        feed_quietly(mixer, _Lane(ROOM, "t-1"), pcm(1000))
        assert await mixer._mix_once(ROOM)
        assert capture.sent == [(ROOM, pcm(1000))]

    async def test_two_tracks_mix_into_one_window(self) -> None:
        capture = _Capture()
        mixer = make_mixer(capture)
        feed_quietly(mixer, _Lane(ROOM, "t-1"), pcm(1000))
        feed_quietly(mixer, _Lane(ROOM, "t-2"), pcm(2000))
        await mixer._mix_once(ROOM)
        [(_, data)] = capture.sent
        assert sample_of(data) == int(3000 / math.sqrt(2))

    async def test_a_track_short_of_a_window_sits_out_the_mix(self) -> None:
        capture = _Capture()
        mixer = make_mixer(capture)
        feed_quietly(mixer, _Lane(ROOM, "t-1"), pcm(1000))
        feed_quietly(mixer, _Lane(ROOM, "t-2"), pcm(2000, samples=10))
        await mixer._mix_once(ROOM)
        # One contributor: passthrough, no headroom — the short track's bytes
        # keep their place for the next window.
        assert capture.sent == [(ROOM, pcm(1000))]

    async def test_a_silence_only_window_is_not_forwarded(self) -> None:
        capture = _Capture()
        mixer = make_mixer(capture)
        feed_quietly(mixer, _Lane(ROOM, "t-1"), pcm(1000))
        await mixer._mix_once(ROOM)
        assert await mixer._mix_once(ROOM)  # rings now empty
        assert len(capture.sent) == 1

    async def test_the_ring_drops_oldest_and_counts_windows(self) -> None:
        capture = _Capture()
        mixer = make_mixer(capture)
        lane = _Lane(ROOM, "t-1")
        for value in range(1, 9):  # 8 windows into a 5-window ring
            feed_quietly(mixer, lane, pcm(value * 100))
        assert mixer.dropped_windows(ROOM) == 8 - _RING_WINDOWS
        await mixer._mix_once(ROOM)
        # The oldest surviving window is the 4th fed, not the 1st.
        assert sample_of(capture.sent[0][1]) == 400

    async def test_the_mix_is_resampled_to_the_provider_rate(self) -> None:
        capture = _Capture()
        mixer = make_mixer(capture, input_sample_rate=24000)
        feed_quietly(mixer, _Lane(ROOM, "t-1"), pcm(1000))
        await mixer._mix_once(ROOM)
        [(_, data)] = capture.sent
        assert len(data) == WINDOW_BYTES * 24000 // RATE

    async def test_a_frame_at_the_wrong_rate_is_dropped(self) -> None:
        capture = _Capture()
        mixer = make_mixer(capture)
        lane = _Lane(ROOM, "t-1")
        feed_quietly(mixer, lane, pcm(1000))
        mixer.feed(lane, AudioFrame(data=pcm(2000), sample_rate=8000))
        await mixer._mix_once(ROOM)
        assert capture.sent == [(ROOM, pcm(1000))]

    async def test_a_failing_send_does_not_stop_the_clock(self) -> None:
        async def refuse(room_id: str, data: bytes) -> None:
            raise RuntimeError("provider gone")

        mixer = ConferenceMixer(send=refuse)
        mixer.configure(input_sample_rate=RATE)
        feed_quietly(mixer, _Lane(ROOM, "t-1"), pcm(1000))
        assert await mixer._mix_once(ROOM)


class TestLifecycle:
    async def test_an_inactive_mixer_ignores_frames(self) -> None:
        capture = _Capture()
        mixer = ConferenceMixer(send=capture)
        mixer.feed(_Lane(ROOM, "t-1"), AudioFrame(data=pcm(1000), sample_rate=RATE))
        assert mixer._rooms == {}

    async def test_deactivate_drops_rooms_and_further_frames(self) -> None:
        capture = _Capture()
        mixer = make_mixer(capture)
        feed_quietly(mixer, _Lane(ROOM, "t-1"), pcm(1000))
        mixer.deactivate()
        assert mixer._rooms == {}
        mixer.feed(_Lane(ROOM, "t-1"), AudioFrame(data=pcm(1000), sample_rate=RATE))
        assert mixer._rooms == {}

    async def test_dropping_the_last_track_forgets_the_room(self) -> None:
        capture = _Capture()
        mixer = make_mixer(capture)
        feed_quietly(mixer, _Lane(ROOM, "t-1"), pcm(1000))
        mixer.drop_track(ROOM, "t-1")
        assert ROOM not in mixer._rooms

    async def test_the_ticker_mixes_on_its_own_clock(self) -> None:
        capture = _Capture()
        mixer = make_mixer(capture)
        mixer.feed(_Lane(ROOM, "t-1"), AudioFrame(data=pcm(1000) * 3, sample_rate=RATE))
        await asyncio.sleep(0.1)
        assert capture.sent
        mixer.forget_room(ROOM)
