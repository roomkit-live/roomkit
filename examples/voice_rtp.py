"""RoomKit -- RTP voice backend demo.

Send and receive voice audio over RTP, with mock STT/TTS/AI providers.
Useful for testing RTP integration with a PBX, SIP gateway, or tools
like ffmpeg/GStreamer that can send/receive RTP streams.

Prerequisites:
    pip install roomkit[rtp]

Run with:
    uv run python examples/voice_rtp.py

Environment variables (all optional):
    RTP_LOCAL_PORT   - Local port to bind RTP (default: 10000)
    RTP_REMOTE_HOST  - Remote host to send RTP to (default: 127.0.0.1)
    RTP_REMOTE_PORT  - Remote port to send RTP to (default: 20000)

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import logging
import os
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
from roomkit.voice.backends.rtp import RTPVoiceBackend
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider, VADEvent, VADEventType
from roomkit.voice.pipeline.dtmf import MockDTMFDetector
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("voice_rtp")


async def main() -> None:
    kit = RoomKit()

    # --- Configuration from env vars ------------------------------------------
    local_port = int(os.environ.get("RTP_LOCAL_PORT", "10000"))
    remote_host = os.environ.get("RTP_REMOTE_HOST", "127.0.0.1")
    remote_port = int(os.environ.get("RTP_REMOTE_PORT", "20000"))

    # --- Backend: RTP ---------------------------------------------------------
    backend = RTPVoiceBackend(
        local_addr=("0.0.0.0", local_port),
        remote_addr=(remote_host, remote_port),
        payload_type=0,  # PCMU (G.711 mu-law)
        clock_rate=8000,
    )

    # --- Pipeline: VAD + DTMF -------------------------------------------------
    # Mock VAD produces a fixed speech start/end sequence for demo purposes.
    # Replace with a real VAD provider (e.g. Silero) in production.
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
    dtmf = MockDTMFDetector()
    pipeline = AudioPipelineConfig(vad=vad, dtmf=dtmf)

    # --- STT + TTS (mock for demo) --------------------------------------------
    stt = MockSTTProvider(transcripts=["Hello from the RTP demo!"])
    tts = MockTTSProvider()

    # --- AI provider ----------------------------------------------------------
    ai_provider = MockAIProvider(
        responses=["I received your message over RTP. The voice pipeline is working!"]
    )

    # --- Channels -------------------------------------------------------------
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
    await kit.create_room(room_id="rtp-demo")
    await kit.attach_channel("rtp-demo", "voice")
    await kit.attach_channel("rtp-demo", "ai", category=ChannelCategory.INTELLIGENCE)

    # --- Hooks: log pipeline events -------------------------------------------
    @kit.hook(HookTrigger.ON_SPEECH_START, execution=HookExecution.ASYNC)
    async def on_speech_start(session, ctx):
        logger.info("Speech started: session=%s", session.id)

    @kit.hook(HookTrigger.ON_SPEECH_END, execution=HookExecution.ASYNC)
    async def on_speech_end(session, ctx):
        logger.info("Speech ended: session=%s", session.id)

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(text, ctx):
        logger.info("Transcription: %s", text)
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_DTMF, execution=HookExecution.ASYNC)
    async def on_dtmf(event, ctx):
        logger.info("DTMF digit: %s (duration=%sms)", event.digit, event.duration_ms)

    @kit.hook(HookTrigger.BEFORE_TTS)
    async def before_tts(text, ctx):
        logger.info("AI says: %s", text)
        return HookResult.allow()

    # --- Start RTP session ----------------------------------------------------
    session = await backend.connect("rtp-demo", "rtp-caller", "voice")
    binding = ChannelBinding(
        room_id="rtp-demo", channel_id="voice", channel_type=ChannelType.VOICE
    )
    voice.bind_session(session, "rtp-demo", binding)

    logger.info(
        "RTP listening on 0.0.0.0:%d, sending to %s:%d",
        local_port,
        remote_host,
        remote_port,
    )
    logger.info("Press Ctrl+C to stop.\n")

    # --- Keep running until Ctrl+C --------------------------------------------
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()

    # --- Cleanup --------------------------------------------------------------
    logger.info("\nStopping...")
    await backend.disconnect(session)
    await kit.close()
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
