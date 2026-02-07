"""RoomKit -- Local microphone/speaker voice demo.

Talk to an AI assistant using your system microphone and speakers.
Audio is captured from the mic, processed through the pipeline (VAD),
transcribed via STT, sent to the AI, and the response is spoken back
through your speakers.

Prerequisites:
    pip install roomkit[local-audio]

Run with:
    uv run python examples/voice_local_audio.py

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
    HookTrigger,
    MockAIProvider,
    RoomKit,
)
from roomkit.channels.ai import AIChannel
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider, VADEvent, VADEventType
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("voice_local_audio")


async def main() -> None:
    kit = RoomKit()

    # --- Backend: local mic/speakers ------------------------------------------
    backend = LocalAudioBackend(
        input_sample_rate=16000,
        output_sample_rate=24000,
        channels=1,
        block_duration_ms=20,
        # input_device=None,   # use default mic
        # output_device=None,  # use default speakers
    )

    # --- Pipeline: VAD detects speech start/end -------------------------------
    # In a real setup, replace MockVADProvider with a real VAD provider
    # (e.g. Silero VAD). The mock here produces a fixed event sequence
    # for demonstration purposes.
    vad = MockVADProvider(
        events=[
            VADEvent(type=VADEventType.SPEECH_START),
            None,
            None,
            None,
            VADEvent(
                type=VADEventType.SPEECH_END,
                audio_bytes=b"demo-audio",
                duration_ms=2000.0,
            ),
        ]
    )
    pipeline = AudioPipelineConfig(vad=vad)

    # --- STT + TTS (mock for demo, swap with real providers) ------------------
    stt = MockSTTProvider(transcripts=["Hello, can you hear me?"])
    tts = MockTTSProvider()

    # --- AI provider ----------------------------------------------------------
    ai_provider = MockAIProvider(
        responses=["Yes, I can hear you! This is the local audio demo working."]
    )

    # --- Channels -------------------------------------------------------------
    from roomkit import VoiceChannel

    voice = VoiceChannel(
        "voice",
        stt=stt,
        tts=tts,
        backend=backend,
        pipeline=pipeline,
    )
    kit.register_channel(voice)

    ai = AIChannel("ai", provider=ai_provider)
    kit.register_channel(ai)

    # --- Room -----------------------------------------------------------------
    await kit.create_room(room_id="local-demo")
    await kit.attach_channel("local-demo", "voice")
    await kit.attach_channel("local-demo", "ai", category=ChannelCategory.INTELLIGENCE)

    # --- Hooks: log what's happening ------------------------------------------
    @kit.hook(HookTrigger.ON_SPEECH_START, execution=HookExecution.ASYNC)
    async def on_speech_start(session, ctx):
        logger.info("Speech started")

    @kit.hook(HookTrigger.ON_SPEECH_END, execution=HookExecution.ASYNC)
    async def on_speech_end(session, ctx):
        logger.info("Speech ended")

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(text, ctx):
        from roomkit import HookResult

        logger.info("Transcription: %s", text)
        return HookResult.allow()

    @kit.hook(HookTrigger.BEFORE_TTS)
    async def before_tts(text, ctx):
        from roomkit import HookResult

        logger.info("AI says: %s", text)
        return HookResult.allow()

    # --- Start voice session --------------------------------------------------
    session = await backend.connect("local-demo", "local-user", "voice")
    binding = ChannelBinding(
        room_id="local-demo", channel_id="voice", channel_type=ChannelType.VOICE
    )
    voice.bind_session(session, "local-demo", binding)

    logger.info("Starting mic capture... speak into your microphone!")
    logger.info("Press Ctrl+C to stop.\n")

    await backend.start_listening(session)

    # --- Keep running until Ctrl+C --------------------------------------------
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()

    # --- Cleanup --------------------------------------------------------------
    logger.info("\nStopping...")
    await backend.stop_listening(session)
    await backend.disconnect(session)
    await kit.close()
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
