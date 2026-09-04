"""RoomKit -- The voice test bench: a simulated phone and a hook timeline.

Plays a synthetic utterance into a VoiceChannel through ScenarioVoiceBackend
(20 ms frames, paced like a real transport), lets the mock STT -> AI -> TTS
turn run, waits on the hook timeline instead of sleeping, then writes what
the bot said to a WAV file and prints the turn's latencies.

Everything is mock, so it runs without keys, models or audio devices.

Run with:
    uv run python examples/voice_scenario_backend.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import HookTrigger, RoomKit, ScenarioVoiceBackend, VoiceChannel, VoiceTrace
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider, VADEvent, VADEventType
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.testing import silence, tone
from roomkit.voice.tts.mock import MockTTSProvider


async def main() -> None:
    setup_logging("voice_scenario_backend")

    stt = MockSTTProvider(transcripts=["what time do you open tomorrow"])
    tts = MockTTSProvider()
    backend = ScenarioVoiceBackend()

    # The caller's utterance: 300 ms of tone then 100 ms of silence, twenty
    # 20 ms frames. At the mock level the VAD is scripted: speech starts on
    # the first frame and ends on the last one, with the whole clip attached.
    utterance = tone(300) + silence(100)
    vad_events: list[VADEvent | None] = [VADEvent(type=VADEventType.SPEECH_START)]
    vad_events += [None] * 18
    vad_events.append(VADEvent(type=VADEventType.SPEECH_END, audio_bytes=utterance.data))

    kit = RoomKit(stt=stt, tts=tts, voice=backend)
    # Before the first session, so ON_SESSION_STARTED is on the timeline too.
    trace = VoiceTrace(kit)
    kit.register_channel(
        VoiceChannel(
            "voice",
            stt=stt,
            tts=tts,
            backend=backend,
            pipeline=AudioPipelineConfig(vad=MockVADProvider(events=vad_events)),
        )
    )
    kit.register_channel(
        AIChannel(
            "agent",
            provider=MockAIProvider(responses=["We open at nine tomorrow."]),
            system_prompt="Answer in one sentence.",
        )
    )
    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice")
    await kit.attach_channel(room.id, "agent")
    session = await kit.join(room.id, "voice", participant_id="caller")

    # Paced like a phone line: twenty frames take about 400 ms of wall clock.
    frames = await backend.play(session, utterance)
    print(f"played {frames} frames ({utterance.duration_ms:.0f} ms)")

    heard = await trace.wait_for(HookTrigger.ON_TRANSCRIPTION, timeout=5)
    print(f"heard: {heard.payload.text!r}")
    await trace.wait_for(HookTrigger.AFTER_TTS, timeout=5)

    speech_end = trace.first(HookTrigger.ON_SPEECH_END)
    before_tts = trace.first(HookTrigger.BEFORE_TTS)
    assert speech_end is not None
    assert before_tts is not None
    print(f"stt latency: {trace.elapsed_ms(speech_end, heard):.1f} ms")
    print(f"response latency: {trace.elapsed_ms(speech_end, before_tts):.1f} ms")

    out = Path(tempfile.mkdtemp(prefix="roomkit-bench-")) / "bot.wav"
    backend.write_capture(session, out)
    print(f"bot audio: {out} ({backend.captured(session).duration_ms:.1f} ms)")
    print("timeline:", " -> ".join(t.value for t in trace.sequence()))

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
