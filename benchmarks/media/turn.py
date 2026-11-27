"""The latency the turn detector adds on its fast path (RFC §12.3.12, RK-12.3.12-022).

The RFC's figure is 200 ms at most. What the framework adds around a detector
is measured with a detector that answers at once (``MockTurnDetector``: every
turn complete): from the transcript's ``ON_TRANSCRIPTION`` hook to the routed
message's ``BEFORE_BROADCAST`` hook, both synchronous, with the detector and
without one. A real detector's own latency (a model) is its provider's to
measure. Each sample is one turn of one session (a scripted VAD opens and
closes the speech, a mock STT answers its transcript), the next turn fed once
the previous one reached the room.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from roomkit import HookResult, HookTrigger, RoomEvent, RoomKit, VoiceChannel
from roomkit.voice import (
    AudioFrame,
    AudioPipelineConfig,
    MockTurnDetector,
    MockVADProvider,
    VADEvent,
    VADEventType,
)
from roomkit.voice.backends import MockVoiceBackend
from roomkit.voice.stt import MockSTTProvider

SAMPLE_RATE = 16000
TURNS = 200
CASES = ("with-detector", "without-detector")
FIGURE_MS = 200.0
_FRAME = AudioFrame(data=b"\x01\x00" * (SAMPLE_RATE // 50), sample_rate=SAMPLE_RATE)
_TURN_BOUND_S = 5.0


async def _turn_latencies(turns: int, *, detector: bool) -> list[float]:
    """Milliseconds from each turn's transcript to its routed message."""
    events: list[VADEvent | None] = []
    for _ in range(turns):
        events += [
            VADEvent(type=VADEventType.SPEECH_START),
            VADEvent(type=VADEventType.SPEECH_END, audio_bytes=_FRAME.data * 5),
        ]
    pipeline = AudioPipelineConfig(
        vad=MockVADProvider(events=events),
        turn_detector=MockTurnDetector() if detector else None,
    )
    backend = MockVoiceBackend()
    stt = MockSTTProvider(transcripts=[f"Turn {i}." for i in range(turns)])
    kit = RoomKit(voice=backend)
    kit.register_channel(VoiceChannel("voice", stt=stt, backend=backend, pipeline=pipeline))
    transcribed: list[float] = []
    routed: list[float] = []
    arrived = asyncio.Event()

    @kit.hook(HookTrigger.ON_TRANSCRIPTION, name="transcribed")
    async def on_transcription(event: Any, ctx: Any) -> HookResult:
        transcribed.append(time.perf_counter())
        return HookResult.allow()

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="routed")
    async def before_broadcast(event: RoomEvent, ctx: Any) -> HookResult:
        if event.source.channel_id == "voice":
            routed.append(time.perf_counter())
            arrived.set()
        return HookResult.allow()

    try:
        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice")
        session = await kit.join(room.id, "voice", participant_id="caller")
        for _ in range(turns):
            arrived.clear()
            await backend.simulate_audio_received(session, _FRAME)  # speech starts
            await backend.simulate_audio_received(session, _FRAME)  # speech ends
            await asyncio.wait_for(arrived.wait(), _TURN_BOUND_S)
    finally:
        await kit.close()
    if len(transcribed) != turns or len(routed) != turns:
        raise RuntimeError(
            f"{turns} turns fed, {len(transcribed)} transcribed, {len(routed)} routed"
        )
    return [(r - t) * 1000 for t, r in zip(transcribed, routed, strict=True)]


def measure(turns: int = TURNS) -> Callable[[], dict[str, list[float]]]:
    """One pass: the turns of one session with the detector, then without."""

    def run() -> dict[str, list[float]]:
        return {
            case: asyncio.run(_turn_latencies(turns, detector=case == "with-detector"))
            for case in CASES
        }

    return run
