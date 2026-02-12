"""RoomKit -- Audio level VU meter with local mic/speakers.

Demonstrates the ON_INPUT_AUDIO_LEVEL and ON_OUTPUT_AUDIO_LEVEL hooks
using LocalAudioBackend.  Mic audio is captured, processed through the
pipeline, transcribed, and echoed back via TTS ("parrot mode").

A text-based VU meter prints continuously for both input (mic) and
output (speaker) directions.

Prerequisites:
    pip install roomkit[local-audio]

Run with:
    uv run python examples/audio_level_vu_meter.py

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import logging
import signal

from roomkit import (
    ChannelBinding,
    ChannelCategory,
    ChannelType,
    HookExecution,
    HookResult,
    HookTrigger,
    MockAIProvider,
    RoomKit,
    VoiceChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.pipeline import (
    AudioPipelineConfig,
    MockVADProvider,
    VADEvent,
    VADEventType,
)
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

logging.basicConfig(level=logging.WARNING)

# Number of parrot turns to simulate before looping stops.
NUM_TURNS = 50

# Phrases the parrot will cycle through.
PHRASES = [
    "Hello, I am a parrot!",
    "Polly wants a cracker!",
    "Pretty bird, pretty bird!",
    "Squawk squawk!",
    "Who's a good bird?",
]


def _vu_bar(level_db: float, width: int = 40) -> str:
    """Render a text VU meter bar from a dB level (-60..0)."""
    clamped = max(-60.0, min(0.0, level_db))
    ratio = (clamped + 60.0) / 60.0  # 0.0 = silence, 1.0 = max
    filled = int(ratio * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _build_vad_events(num_turns: int) -> list[VADEvent | None]:
    """Build a repeating VAD event sequence for multiple turns.

    Each turn: SPEECH_START → 48 None frames (~1s) → SPEECH_END.
    Between turns: 100 None frames (~2s for TTS playback).
    """
    events: list[VADEvent | None] = []
    for _ in range(num_turns):
        events.append(VADEvent(type=VADEventType.SPEECH_START))
        events.extend([None] * 48)
        events.append(
            VADEvent(
                type=VADEventType.SPEECH_END,
                audio_bytes=b"\x00" * 640,
                duration_ms=1000.0,
            )
        )
        # Idle frames while TTS plays back
        events.extend([None] * 100)
    # Extra tail so mic keeps running after the last turn
    events.extend([None] * 5000)
    return events


async def main() -> None:
    kit = RoomKit()

    # --- Backend: local mic/speakers ----------------------------------------
    backend = LocalAudioBackend(
        input_sample_rate=16000,
        output_sample_rate=16000,
        channels=1,
        block_duration_ms=20,
    )

    # --- Pipeline: mock VAD with repeating turn cycles ----------------------
    vad_events = _build_vad_events(NUM_TURNS)
    pipeline = AudioPipelineConfig(vad=MockVADProvider(events=vad_events))

    # --- STT + TTS (mock, cycling phrases) ----------------------------------
    transcripts = [PHRASES[i % len(PHRASES)] for i in range(NUM_TURNS)]
    stt = MockSTTProvider(transcripts=transcripts)
    tts = MockTTSProvider()

    # --- AI: parrot mode (echoes back whatever was said) --------------------
    ai_provider = MockAIProvider(responses=list(transcripts))

    # --- Channels -----------------------------------------------------------
    voice = VoiceChannel("voice", stt=stt, tts=tts, backend=backend, pipeline=pipeline)
    kit.register_channel(voice)

    ai = AIChannel("ai", provider=ai_provider)
    kit.register_channel(ai)

    await kit.create_room(room_id="vu-demo")
    await kit.attach_channel("vu-demo", "voice")
    await kit.attach_channel("vu-demo", "ai", category=ChannelCategory.INTELLIGENCE)

    # --- Audio level hooks (VU meter) ---------------------------------------

    @kit.hook(
        HookTrigger.ON_INPUT_AUDIO_LEVEL,
        execution=HookExecution.ASYNC,
        name="input_vu",
    )
    async def on_input_level(event, ctx):
        bar = _vu_bar(event.level_db)
        print(f"  IN  {event.level_db:+6.1f} dB  {bar}")

    @kit.hook(
        HookTrigger.ON_OUTPUT_AUDIO_LEVEL,
        execution=HookExecution.ASYNC,
        name="output_vu",
    )
    async def on_output_level(event, ctx):
        bar = _vu_bar(event.level_db)
        print(f"  OUT {event.level_db:+6.1f} dB  {bar}")

    # --- Log transcription and TTS for visibility ---------------------------

    @kit.hook(HookTrigger.ON_TRANSCRIPTION, name="log_stt")
    async def on_transcription(text, ctx):
        print(f"\n  STT: {text}")
        return HookResult.allow()

    @kit.hook(HookTrigger.BEFORE_TTS, name="log_tts")
    async def before_tts(text, ctx):
        print(f"  TTS: {text}\n")
        return HookResult.allow()

    # --- Start voice session ------------------------------------------------
    session = await backend.connect("vu-demo", "local-user", "voice")
    binding = ChannelBinding(room_id="vu-demo", channel_id="voice", channel_type=ChannelType.VOICE)
    voice.bind_session(session, "vu-demo", binding)

    print("Audio Level VU Meter (Parrot Mode)")
    print("=" * 60)
    print("Speak into your microphone to see input levels.")
    print("The parrot echoes back via TTS (output levels).")
    print(f"Configured for {NUM_TURNS} turns. Press Ctrl+C to stop.\n")

    await backend.start_listening(session)

    # --- Keep running until Ctrl+C ------------------------------------------
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()

    # --- Cleanup ------------------------------------------------------------
    print("\nStopping...")
    await backend.stop_listening(session)
    await backend.disconnect(session)
    await kit.close()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
