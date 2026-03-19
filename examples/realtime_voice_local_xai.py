"""RoomKit — Speech-to-speech with xAI Grok Realtime using local mic/speakers.

Talk to Grok using your system microphone — AI audio plays through your
speakers.  xAI handles turn detection server-side.  WebRTC AEC strips
speaker echo so the mic stays open during playback.

Requirements:
    pip install roomkit websockets sounddevice numpy aec-audio-processing

Run with:
    XAI_API_KEY=xai-... uv run python examples/realtime_voice_local_xai.py

Environment variables:
    XAI_API_KEY         (required) xAI API key
    XAI_MODEL           Model name (default: grok-3-fast)
    XAI_VOICE           Voice preset: eve | ara | rex | sal | leo (default: eve)
    SYSTEM_PROMPT       Custom system prompt

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.providers.xai.config import XAIRealtimeConfig
from roomkit.providers.xai.realtime import XAIRealtimeProvider
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.pipeline.aec.webrtc import WebRTCAECProvider
from roomkit.voice.pipeline.config import AudioPipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("realtime_voice_local_xai")


async def main() -> None:
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        logger.error("Set XAI_API_KEY to run this example.")
        logger.error("  XAI_API_KEY=xai-... uv run python examples/realtime_voice_local_xai.py")
        return

    kit = RoomKit()

    # --- xAI Realtime provider (speech-to-speech) ---
    config = XAIRealtimeConfig(
        api_key=api_key,
        model=os.environ.get("XAI_MODEL", "grok-3-fast"),
        voice=os.environ.get("XAI_VOICE", "eve"),
    )
    provider = XAIRealtimeProvider(config)

    # --- WebRTC AEC (echo cancellation + noise suppression) ---
    sample_rate = 24000
    block_ms = 20

    aec = WebRTCAECProvider(sample_rate=sample_rate, enable_ns=True)
    pipeline = AudioPipelineConfig(aec=aec)

    transport = LocalAudioBackend(
        input_sample_rate=sample_rate,
        output_sample_rate=sample_rate,
        block_duration_ms=block_ms,
        mute_mic_during_playback=False,
        aec=aec,
        pipeline=pipeline,
    )

    # --- Realtime voice channel ---
    channel = RealtimeVoiceChannel(
        "voice",
        provider=provider,
        transport=transport,
        system_prompt=os.environ.get(
            "SYSTEM_PROMPT",
            "You are a friendly voice assistant powered by Grok. Be concise and helpful.",
        ),
        voice=config.voice,
        input_sample_rate=sample_rate,
        output_sample_rate=sample_rate,
    )
    kit.register_channel(channel)

    # --- Room ---
    await kit.create_room(room_id="local-demo")
    await kit.attach_channel("local-demo", "voice")

    # --- Start session ---
    session = await channel.start_session(
        "local-demo",
        "local-user",
        connection=None,
    )

    logger.info(
        "xAI Grok Realtime session started (voice=%s, model=%s)", config.voice, config.model
    )
    logger.info("Speak into your microphone! Press Ctrl+C to stop.\n")

    # --- Keep running until Ctrl+C ---
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()

    # --- Cleanup ---
    logger.info("\nStopping...")
    await channel.end_session(session)
    await kit.close()
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
