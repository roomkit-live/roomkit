"""A voice session's audio level hooks fire at most once per 100 ms (RFC §12.5).

The throttle measures that interval on the media clock (``roomkit._clock``):
the tests read it from a clock they set, so frames and chunks handed at once
fire one level, and one more once the clock moved past 100 ms.
"""

from __future__ import annotations

import asyncio

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit, VoiceChannel, _clock
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import VoiceSession
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider
from tests.media_clock import SetClock

_FRAME = b"\x01\x00" * 320  # 20 ms at 16 kHz mono

# The clock's readings, in ms, at each of which the session hands two
# frames (input) or speaks two words (output), and how many levels have
# fired once the work of that reading ran: within 100 ms of the last level,
# none; past it, one.
_READINGS_MS = (0, 60, 99, 110, 160, 220)
_LEVELS_SO_FAR = [1, 1, 1, 2, 2, 3]


async def _drain() -> None:
    """Let the tasks the reading's frames or words scheduled run to their end."""
    for _ in range(20):
        await asyncio.sleep(0)


async def _hand(
    direction: str, channel: VoiceChannel, backend: MockVoiceBackend, session: VoiceSession
) -> None:
    if direction == "input":
        for _ in range(2):
            await backend.simulate_audio_received(session, AudioFrame(data=_FRAME))
    else:
        await channel.say(session, "two words")


@pytest.mark.parametrize(
    ("direction", "trigger"),
    [
        ("input", HookTrigger.ON_INPUT_AUDIO_LEVEL),
        ("output", HookTrigger.ON_OUTPUT_AUDIO_LEVEL),
    ],
)
async def test_a_session_s_levels_fire_once_per_100_ms_of_the_media_clock(
    direction: str, trigger: HookTrigger
) -> None:
    backend = MockVoiceBackend()
    kit = RoomKit(voice=backend)
    channel = VoiceChannel(
        "voice-1",
        stt=MockSTTProvider(transcripts=[]),
        tts=MockTTSProvider(),
        backend=backend,
        pipeline=AudioPipelineConfig(),
    )
    kit.register_channel(channel)
    levels: list[str] = []

    @kit.hook(trigger, execution=HookExecution.ASYNC)
    async def record(event: object, ctx: object) -> None:
        levels.append(direction)

    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice-1")
    session = await kit.join(room.id, "voice-1", participant_id="alice")
    clock = SetClock()
    so_far: list[int] = []
    with _clock.installed(clock):
        for at_ms in _READINGS_MS:
            # Readings well away from 0.0, which the throttle takes for never.
            clock.now = 1000.0 + at_ms / 1000
            await _hand(direction, channel, backend, session)
            await _drain()
            so_far.append(len(levels))
    await kit.close()

    assert so_far == _LEVELS_SO_FAR
