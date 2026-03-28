"""RoomKit -- Console display demo with audio meters and colored logs.

Demonstrates the RoomKitConsole module which provides a polished terminal
experience for voice agent development.  It auto-registers hooks for audio
levels, voice state, transcription, and more — rendering a real-time
status bar with waveform meters at the bottom of the terminal.

Prerequisites:
    pip install roomkit[console,local-audio]

Run with:
    uv run python examples/voice_console_demo.py

Press Ctrl+C to stop.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio

from shared import run_until_stopped

from roomkit import ChannelCategory, HookResult, HookTrigger, RoomKit, VoiceChannel
from roomkit.channels.ai import AIChannel
from roomkit.console import RoomKitConsole
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider, VADEvent, VADEventType
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

NUM_TURNS = 20

PHRASES = [
    "Hello, I am your voice assistant!",
    "How can I help you today?",
    "That's an interesting question.",
    "Let me think about that for a moment.",
    "I'm happy to assist with anything you need.",
]


def _build_vad_events(num_turns: int) -> list[VADEvent | None]:
    """Build a repeating VAD event sequence for multiple turns."""
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
        events.extend([None] * 100)
    events.extend([None] * 5000)
    return events


async def main() -> None:
    kit = RoomKit()

    # --- Console: enable rich display -----------------------------------------
    console = RoomKitConsole(kit)

    # --- Backend: local mic/speakers ------------------------------------------
    backend = LocalAudioBackend(
        input_sample_rate=16000,
        output_sample_rate=16000,
        channels=1,
        block_duration_ms=20,
    )

    # --- Pipeline: mock VAD ---------------------------------------------------
    vad_events = _build_vad_events(NUM_TURNS)
    pipeline = AudioPipelineConfig(vad=MockVADProvider(events=vad_events))

    # --- STT + TTS (mock) -----------------------------------------------------
    transcripts = [PHRASES[i % len(PHRASES)] for i in range(NUM_TURNS)]
    stt = MockSTTProvider(transcripts=transcripts)
    tts = MockTTSProvider()

    # --- AI: echo mode --------------------------------------------------------
    ai_provider = MockAIProvider(responses=list(transcripts))

    # --- Channels -------------------------------------------------------------
    voice = VoiceChannel("voice", stt=stt, tts=tts, backend=backend, pipeline=pipeline)
    kit.register_channel(voice)

    ai = AIChannel("ai", provider=ai_provider)
    kit.register_channel(ai)

    await kit.create_room(room_id="console-demo")
    await kit.attach_channel("console-demo", "ai", category=ChannelCategory.INTELLIGENCE)

    # --- Hook: allow transcriptions through -----------------------------------

    @kit.hook(HookTrigger.ON_TRANSCRIPTION, name="allow_stt")
    async def on_transcription(event, ctx):
        return HookResult.allow()

    @kit.hook(HookTrigger.BEFORE_TTS, name="allow_tts")
    async def before_tts(text, ctx):
        return HookResult.allow()

    # --- Start voice session --------------------------------------------------
    await kit.attach_channel("console-demo", "voice")

    await run_until_stopped(kit, cleanup=console.stop)


if __name__ == "__main__":
    asyncio.run(main())
