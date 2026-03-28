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
    XAI_MODEL           Model name (default: grok-2-audio)
    XAI_VOICE           Voice preset: eve | ara | rex | sal | leo (default: eve)
    SYSTEM_PROMPT       Custom system prompt

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import (
    build_aec,
    build_pipeline,
    require_env,
    run_until_stopped,
    setup_console,
    setup_logging,
)

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.providers.xai.config import XAIRealtimeConfig
from roomkit.providers.xai.realtime import XAIRealtimeProvider
from roomkit.voice.backends.local import LocalAudioBackend

logger = setup_logging("realtime_voice_local_xai")


async def main() -> None:
    env = require_env("XAI_API_KEY")

    kit = RoomKit()

    # --- Console dashboard (set CONSOLE=1 to enable) ---
    console_cleanup = setup_console(kit)

    # --- xAI Realtime provider (speech-to-speech) ---
    config = XAIRealtimeConfig(
        api_key=env["XAI_API_KEY"],
        model=os.environ.get("XAI_MODEL", "grok-2-audio"),
        voice=os.environ.get("XAI_VOICE", "eve"),
    )
    provider = XAIRealtimeProvider(config)

    # --- Audio pipeline (AEC + noise suppression) ---
    sample_rate = 24000
    block_ms = 20

    aec = build_aec(sample_rate, block_ms, default="webrtc")
    pipeline = build_pipeline(aec=aec)

    transport = LocalAudioBackend(
        input_sample_rate=sample_rate,
        output_sample_rate=sample_rate,
        block_duration_ms=block_ms,
        mute_mic_during_playback=False,
        aec=aec,
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
        pipeline=pipeline,
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
    async def cleanup() -> None:
        if console_cleanup:
            await console_cleanup()
        await channel.end_session(session)

    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    asyncio.run(main())
