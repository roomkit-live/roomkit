"""RoomKit -- Proactive voice: say() and play().

Demonstrates how to proactively speak text or play pre-rendered audio
to a participant using VoiceChannel.say() and VoiceChannel.play().

Use cases:
- Greet a caller when they connect
- Play announcements or pre-recorded prompts
- Deliver system messages without going through the AI pipeline

say()  — synthesizes text via TTS and sends to the participant.
play() — sends pre-rendered audio (bytes or async chunks) directly.

Both methods support barge-in (the participant can interrupt) and
integrate with the audio pipeline (AEC reference, outbound processing).

This example uses mock providers so it runs without external deps.

Run with:
    uv run python examples/voice_say_play.py
"""

from __future__ import annotations

import asyncio
import logging

from roomkit import (
    HookExecution,
    HookTrigger,
    RoomKit,
    VoiceChannel,
)
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice_say_play")


async def main() -> None:
    # --- Providers ------------------------------------------------------------
    backend = MockVoiceBackend()
    tts = MockTTSProvider()
    stt = MockSTTProvider()

    kit = RoomKit(voice=backend)

    # --- Voice channel --------------------------------------------------------
    voice = VoiceChannel("voice", stt=stt, tts=tts, backend=backend)
    kit.register_channel(voice)

    # --- Room -----------------------------------------------------------------
    await kit.create_room(room_id="demo")
    await kit.attach_channel("demo", "voice")

    # --- Hooks: observe TTS events --------------------------------------------
    @kit.hook(HookTrigger.BEFORE_TTS, name="log_before_tts")
    async def before_tts(text, ctx):
        from roomkit import HookResult

        print(f"  [hook] BEFORE_TTS: {text!r}")
        return HookResult.allow()

    @kit.hook(HookTrigger.AFTER_TTS, execution=HookExecution.ASYNC, name="log_after_tts")
    async def after_tts(text, ctx):
        print(f"  [hook] AFTER_TTS: {text!r}")

    # --- Connect a participant ------------------------------------------------
    session = await kit.connect_voice("demo", "user-1", "voice")
    print(f"Session connected: {session.id}\n")

    # =========================================================================
    # say() — text to speech
    # =========================================================================
    print("--- say(): greeting ---")
    await voice.say(session, "Hello! Welcome to the demo.")
    print(f"  TTS calls: {len(tts.calls)}")
    print(f"  Audio sent: {len(backend.sent_audio)} stream(s)")
    print(f"  Transcriptions: {backend.sent_transcriptions}")

    # say() with a custom voice override
    print("\n--- say(): custom voice ---")
    await voice.say(session, "This uses a different voice.", voice="narrator")
    print(f"  Voice used: {tts.calls[-1]['voice']}")

    # =========================================================================
    # play() — pre-rendered audio
    # =========================================================================
    print("\n--- play(): raw bytes ---")
    await voice.play(session, b"\x00\x00" * 8000)  # 0.5s of silence at 16kHz
    print(f"  Audio sent: {len(backend.sent_audio)} stream(s)")

    # play() with a transcript for UI display
    print("\n--- play(): bytes with transcript ---")
    await voice.play(session, b"\x00\x00" * 8000, text="[pre-recorded greeting]")
    print(f"  Transcriptions: {backend.sent_transcriptions[-1]}")

    # play() with an async chunk iterator (wraps through pipeline if configured)
    async def announcement_chunks():
        """Simulate pre-rendered audio arriving in chunks."""
        for i in range(3):
            data = f"chunk-{i}".encode()
            # Pad to even length for PCM alignment
            if len(data) % 2:
                data += b"\x00"
            yield AudioChunk(data=data, sample_rate=16000, is_final=(i == 2))

    print("\n--- play(): async chunk iterator ---")
    await voice.play(session, announcement_chunks(), text="Attention please")
    print(f"  Audio sent: {len(backend.sent_audio)} stream(s)")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n--- Summary ---")
    print(f"  Total TTS synth calls: {len(tts.calls)}")
    print(f"  Total audio streams sent: {len(backend.sent_audio)}")
    print(f"  Total transcriptions sent: {len(backend.sent_transcriptions)}")

    await kit.close()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
