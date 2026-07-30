"""Audio helpers for conference lane tests.

A lane segments with a VAD, so a test that wants a transcription has to deliver
something a VAD will call speech and then enough silence to close the utterance.
Frames of zeros — the obvious thing to reach for — produce nothing at all, which
is correct behaviour and a confusing test failure.

Shared by the channel, gating and lane suites so the shape of "one utterance"
is written once.
"""

from __future__ import annotations

import struct

from roomkit.channels.conference import ConferenceChannel
from roomkit.conference.mock import MockConferenceBackend
from roomkit.conference.models import ConferenceTrack
from roomkit.voice.audio_frame import AudioFrame

SAMPLE_RATE = 16_000
FRAME_MS = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000

SPEECH_FRAMES = 15
"""300 ms of speech — past EnergyVADProvider's 200 ms minimum."""

SILENCE_FRAMES = 30
"""600 ms of silence — past EnergyVADProvider's 500 ms end-of-speech threshold."""


def speech_frame() -> AudioFrame:
    """A frame loud enough for the energy VAD to call it speech."""
    samples = [8000, -8000] * (SAMPLES_PER_FRAME // 2)
    return AudioFrame(
        data=struct.pack(f"<{SAMPLES_PER_FRAME}h", *samples), sample_rate=SAMPLE_RATE
    )


def silence_frame() -> AudioFrame:
    return AudioFrame(data=b"\x00\x00" * SAMPLES_PER_FRAME, sample_rate=SAMPLE_RATE)


async def say(
    backend: MockConferenceBackend,
    track: ConferenceTrack,
    *,
    speech: int = SPEECH_FRAMES,
    silence: int = SILENCE_FRAMES,
) -> int:
    """Deliver one utterance's worth of frames. Returns how many were sent."""
    for _ in range(speech):
        await backend.simulate_audio(track, speech_frame())
    for _ in range(silence):
        await backend.simulate_audio(track, silence_frame())
    return speech + silence


async def drain(channel: ConferenceChannel, *track_ids: str) -> None:
    """Wait until the named lanes have processed everything they were given."""
    for track_id in track_ids:
        lane = channel.active_lanes.get(track_id)
        if lane is not None:
            await lane.drain()


async def drain_recordings(channel: ConferenceChannel) -> None:
    """Wait until every open recording has written the frames it was given.

    A frame is queued in the callback and written on another thread, so
    "the recorder has it" is only true a moment later. Tests that assert on
    what a recorder received wait here first — the recording equivalent of
    :func:`drain`.
    """
    if channel._recorder is not None:
        await channel._recorder.drain()
