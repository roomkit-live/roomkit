"""N→1 audio: the mix a conference's speech-to-speech provider hears.

A realtime provider takes one audio stream and a conference has N. The mixer
closes that gap the way RFC section 12.10.12 arbitrates it: every subscribed
audio track contributes to a windowed additive mix — int32 sum, headroom
scaled by the square root of the number of contributors, clamped back to
16-bit — and a window to which no track contributed is never forwarded.

The tap is the lane. Frames arrive here after the lane's pipeline has
normalized them to the contract format, so every track contributes at one
rate and the mix needs no per-track conversion; what it needs is one resample
of each mixed window to the provider's declared input rate.

The clock is the mixer's own. Tracks deliver frames on their own cadence and
SFUs suppress silent tracks entirely, so mixing on arrival stalls on the
quietest participant; a wall-clock ticker per room reads whatever each ring
buffer holds every window and moves on. The rings are deliberately shallow —
a conversational loop wants recent audio, not complete audio — so overflow
drops the oldest audio and counts it, the same stay-close-to-live policy as
the lanes' own backlog.

Private to the channel: RFC section 12.10.6 defines no mixer provider
interface, and the pipeline's MixerProvider implements a different headroom
rule for a different consumer (the in-process AudioBridge).
"""

from __future__ import annotations

import array
import asyncio
import logging
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from roomkit.core.task_utils import log_task_exception
from roomkit.voice.audio_frame import AudioFrame

try:  # NumPy accelerates the mix; the pure-Python path mixes without it.
    import numpy as _np
except ImportError:  # pragma: no cover - exercised where numpy is absent
    _np = None

if TYPE_CHECKING:
    from roomkit.channels._conference_lane import ConferenceLane
    from roomkit.voice.pipeline.resampler.base import ResamplerProvider

logger = logging.getLogger("roomkit.channels.conference")

WINDOW_MS = 20
"""One mixing window, and the ticker's cadence."""

_RING_WINDOWS = 5
"""How many windows a track may buffer ahead of the mix.

100 ms: enough to absorb frame jitter, small enough that a stalled send costs
recency rather than latency — the ring drops its oldest audio and the
provider stays near-live. Deliberately not the lanes' ``max_queued_frames``,
which at its default would be two seconds of added latency.
"""

_IDLE_TICKS = 250
"""Consecutive empty windows before a room's ticker stands down (5 s).

The next frame fed restarts it; what a quiet meeting pays for the mixer is a
dictionary lookup per frame, not a clock that never stops.
"""

# What carries a mixed window to the provider: (room_id, pcm bytes).
MixedSender = Callable[[str, bytes], Awaitable[None]]


def _best_resampler() -> ResamplerProvider:
    """The best available resampler, NumPy where installed.

    Mirrors the pipeline's own default rather than importing its private
    factory across the subsystem boundary.
    """
    try:
        from roomkit.voice.pipeline.resampler.numpy import NumpyResamplerProvider

        return NumpyResamplerProvider()
    except ImportError:  # pragma: no cover - exercised where numpy is absent
        from roomkit.voice.pipeline.resampler.linear import LinearResamplerProvider

        return LinearResamplerProvider()


def mix_window(windows: list[bytes]) -> bytes:
    """Mix equal-length 16-bit PCM windows additively (RFC 12.7.5).

    int32 accumulation, headroom scaled by ``1/sqrt(contributors)``, clamped
    to int16. A lone window passes through untouched: nothing was summed, so
    nothing needs headroom. Both paths truncate toward zero, so the mix is
    bit-identical with and without NumPy.
    """
    if len(windows) == 1:
        return windows[0]
    scale = 1.0 / math.sqrt(len(windows))
    if _np is not None:
        acc = _np.frombuffer(windows[0], dtype=_np.int16).astype(_np.int32)
        for window in windows[1:]:
            acc = acc + _np.frombuffer(window, dtype=_np.int16).astype(_np.int32)
        scaled = _np.clip(acc * scale, -32768.0, 32767.0)
        return scaled.astype(_np.int16).tobytes()
    first = array.array("h")
    first.frombytes(windows[0])
    acc_list = list(first)
    for window in windows[1:]:
        other = array.array("h")
        other.frombytes(window)
        for index, sample in enumerate(other):
            acc_list[index] += sample
    mixed = array.array("h", (max(-32768, min(32767, int(sample * scale))) for sample in acc_list))
    return mixed.tobytes()


class _TrackRing:
    """One track's audio waiting for the mix: bounded, drop-oldest."""

    __slots__ = ("buffer", "dropped_bytes")

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.dropped_bytes = 0

    def append(self, data: bytes, cap: int) -> None:
        self.buffer.extend(data)
        overflow = len(self.buffer) - cap
        if overflow > 0:
            del self.buffer[:overflow]
            self.dropped_bytes += overflow

    def take(self, window_bytes: int) -> bytes | None:
        """One full window, or ``None`` while the track has less than that.

        Less than a window is not padded out: a track that contributed
        nothing this window is silence in the sum either way, and its bytes
        keep their place in the next one.
        """
        if len(self.buffer) < window_bytes:
            return None
        window = bytes(self.buffer[:window_bytes])
        del self.buffer[:window_bytes]
        return window


@dataclass
class _RoomMix:
    """One room's share of the mix: its rings, its clock, its rate."""

    rings: dict[str, _TrackRing] = field(default_factory=dict)
    ticker: asyncio.Task[None] | None = None
    sample_rate: int | None = None
    """The contract rate, learned from the first frame fed.

    Every lane's pipeline normalizes to one internal rate, so one value
    covers the room; a frame at any other rate is a bug upstream and is
    dropped rather than mixed as noise.
    """


class ConferenceMixer:
    """The windowed N→1 mix behind a conference's realtime provider.

    Inert until :meth:`configure` names a provider input rate, which is what
    lets the lane tap (:meth:`feed`) be wired unconditionally: a channel with
    no speech-to-speech configured pays a comparison per frame and nothing
    else, and a provider plugged mid-meeting starts hearing the lanes that
    were already open.
    """

    def __init__(self, *, send: MixedSender) -> None:
        self._send = send
        self._rooms: dict[str, _RoomMix] = {}
        self._input_sample_rate: int | None = None
        self._resampler = _best_resampler()

    @property
    def active(self) -> bool:
        return self._input_sample_rate is not None

    def configure(self, *, input_sample_rate: int) -> None:
        """Start mixing, resampling each window to the provider's rate."""
        self._input_sample_rate = input_sample_rate

    def deactivate(self) -> None:
        """Stop mixing and drop everything buffered. The lanes are untouched."""
        self._input_sample_rate = None
        for room_id in list(self._rooms):
            self.forget_room(room_id)

    def feed(self, lane: ConferenceLane, frame: AudioFrame) -> None:
        """The lane tap: one processed frame joins its track's ring.

        Synchronous and cheap by contract — it runs on the lane's task inside
        the frame path, and anything slow here would hold that lane's own VAD
        and STT behind the mix.
        """
        if self._input_sample_rate is None:
            return
        room = self._rooms.get(lane.room_id)
        if room is None:
            room = self._rooms[lane.room_id] = _RoomMix()
        if room.sample_rate is None:
            room.sample_rate = frame.sample_rate
        elif room.sample_rate != frame.sample_rate:
            logger.warning(
                "A frame for track %s arrived at %d Hz in a room mixing at %d Hz; "
                "dropping it rather than mixing it as noise",
                lane.track_id,
                frame.sample_rate,
                room.sample_rate,
            )
            return
        ring = room.rings.get(lane.track_id)
        if ring is None:
            ring = room.rings[lane.track_id] = _TrackRing()
        ring.append(frame.data, self._window_bytes(room.sample_rate) * _RING_WINDOWS)
        if room.ticker is None or room.ticker.done():
            room.ticker = asyncio.create_task(self._tick(lane.room_id))
            room.ticker.add_done_callback(log_task_exception)

    def drop_track(self, room_id: str, track_id: str) -> None:
        """Forget one track's ring; the room follows when it was the last."""
        room = self._rooms.get(room_id)
        if room is None:
            return
        room.rings.pop(track_id, None)
        if not room.rings:
            self.forget_room(room_id)

    def forget_room(self, room_id: str) -> None:
        """Drop a room's rings and stop its clock, when the room goes away."""
        room = self._rooms.pop(room_id, None)
        if room is None:
            return
        if room.ticker is not None:
            room.ticker.cancel()
        self._resampler.reset(self._stream_key(room_id))

    def dropped_windows(self, room_id: str) -> int:
        """How much audio the room's rings have discarded, in whole windows."""
        room = self._rooms.get(room_id)
        if room is None or room.sample_rate is None:
            return 0
        dropped = sum(ring.dropped_bytes for ring in room.rings.values())
        return dropped // self._window_bytes(room.sample_rate)

    @staticmethod
    def _window_bytes(sample_rate: int) -> int:
        return int(sample_rate * WINDOW_MS / 1000) * 2

    @staticmethod
    def _stream_key(room_id: str) -> str:
        return f"rt-mix:{room_id}"

    async def _tick(self, room_id: str) -> None:
        """The room's mixing clock: read the rings every window, send what they hold.

        Absolute deadlines, re-anchored when a send overruns: a slow provider
        costs the mix recency (the rings drop their oldest audio) rather than
        a burst of catch-up windows.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time()
        idle = 0
        while idle < _IDLE_TICKS:
            deadline += WINDOW_MS / 1000
            delay = deadline - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                deadline = loop.time()
            if not await self._mix_once(room_id):
                return
            room = self._rooms.get(room_id)
            idle = 0 if room is None or any(r.buffer for r in room.rings.values()) else idle + 1

    async def _mix_once(self, room_id: str) -> bool:
        """Mix and send one window. Says whether the clock should keep running."""
        room = self._rooms.get(room_id)
        if room is None or self._input_sample_rate is None:
            return False
        if room.sample_rate is None:
            return True
        window_bytes = self._window_bytes(room.sample_rate)
        windows = []
        for ring in room.rings.values():
            window = ring.take(window_bytes)
            if window is not None:
                windows.append(window)
        if not windows:
            # No track contributed: the window is silence, and RFC 12.7.5
            # says a silence-only mix is not forwarded.
            return True
        mixed = mix_window(windows)
        data = self._to_provider_rate(mixed, room.sample_rate, room_id)
        try:
            await self._send(room_id, data)
        except Exception:
            logger.warning(
                "A mixed window for room %s could not be sent to the realtime provider",
                room_id,
                exc_info=True,
            )
        return True

    def _to_provider_rate(self, mixed: bytes, sample_rate: int, room_id: str) -> bytes:
        target = self._input_sample_rate
        if target is None or target == sample_rate:
            return mixed
        resampled = self._resampler.resample(
            AudioFrame(data=mixed, sample_rate=sample_rate),
            target,
            1,
            2,
            self._stream_key(room_id),
        )
        return resampled.data
