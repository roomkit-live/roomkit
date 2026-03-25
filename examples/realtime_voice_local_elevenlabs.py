"""RoomKit — Speech-to-speech with ElevenLabs Conversational AI using local mic/speakers.

Talk to an ElevenLabs agent using your system microphone — AI audio plays
through your speakers.  ElevenLabs handles STT, LLM, TTS, and turn
detection entirely server-side.

Requirements:
    pip install roomkit[realtime-elevenlabs,local-audio]

Setup:
    Create an agent on the ElevenLabs dashboard (https://elevenlabs.io/conversational-ai)
    and copy the agent ID.

Run with:
    ELEVENLABS_API_KEY=... ELEVENLABS_AGENT_ID=... \
        uv run python examples/realtime_voice_local_elevenlabs.py

Environment variables:
    ELEVENLABS_API_KEY      (required) ElevenLabs API key
    ELEVENLABS_AGENT_ID     (required) Agent ID from the dashboard
    ELEVENLABS_VOICE_ID     Override the agent's default voice
    SYSTEM_PROMPT           Override the agent's default system prompt
    ELEVENLABS_LANGUAGE     Language code (e.g. en, fr, ja)

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from typing import Any

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.providers.elevenlabs.config import ElevenLabsRealtimeConfig
from roomkit.providers.elevenlabs.realtime import ElevenLabsRealtimeProvider
from roomkit.voice.backends.local import LocalAudioBackend

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("realtime_voice_local_elevenlabs")


async def main() -> None:
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    agent_id = os.environ.get("ELEVENLABS_AGENT_ID")
    if not api_key or not agent_id:
        print("Set ELEVENLABS_API_KEY and ELEVENLABS_AGENT_ID to run this example.")
        print(
            "  ELEVENLABS_API_KEY=... ELEVENLABS_AGENT_ID=... "
            "uv run python examples/realtime_voice_local_elevenlabs.py"
        )
        return

    kit = RoomKit()

    # --- ElevenLabs Conversational AI provider ---
    config = ElevenLabsRealtimeConfig(api_key=api_key, agent_id=agent_id)
    provider = ElevenLabsRealtimeProvider(config)

    # --- Local audio transport (mic/speakers) ---
    # ElevenLabs uses 16 kHz PCM by default
    sample_rate = 16000
    transport = LocalAudioBackend(
        input_sample_rate=sample_rate,
        output_sample_rate=sample_rate,
        block_duration_ms=20,
    )

    # --- Provider config overrides (passed via session metadata) ---
    # Note: use ELEVENLABS_LANGUAGE, not LANGUAGE (which is a system locale var)
    provider_config: dict[str, str] = {}
    language = os.environ.get("ELEVENLABS_LANGUAGE")
    if language:
        provider_config["language"] = language

    # --- Realtime voice channel ---
    channel = RealtimeVoiceChannel(
        "voice",
        provider=provider,
        transport=transport,
        system_prompt=os.environ.get("SYSTEM_PROMPT"),
        voice=os.environ.get("ELEVENLABS_VOICE_ID"),
        input_sample_rate=sample_rate,
        output_sample_rate=sample_rate,
    )
    kit.register_channel(channel)

    # --- Room ---
    await kit.create_room(room_id="local-demo")
    await kit.attach_channel("local-demo", "voice")

    # --- Start session (provider_config passed via metadata) ---
    metadata: dict[str, Any] = {}
    if provider_config:
        metadata["provider_config"] = provider_config

    session = await channel.start_session(
        "local-demo",
        "local-user",
        connection=None,
        metadata=metadata or None,
    )

    logger.info("ElevenLabs Conversational AI session started")
    logger.info("Speak into your microphone! Press Ctrl+C to stop.\n")

    # --- Keep running until Ctrl+C ---
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()

    logger.info("Stopping...")
    await channel.end_session(session)
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
