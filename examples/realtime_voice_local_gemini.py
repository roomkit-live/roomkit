"""RoomKit — Speech-to-speech with Gemini Live using local mic/speakers.

Talk to Gemini using your system microphone — AI audio plays through
your speakers.  No browser, no WebSocket, no server required.

Requirements:
    pip install roomkit[realtime-gemini,local-audio]

Run with:
    GOOGLE_API_KEY=... uv run python examples/realtime_voice_local_gemini.py

Environment variables:
    GOOGLE_API_KEY      (required) Google API key
    GEMINI_MODEL        Model name (default: gemini-2.5-flash-native-audio-preview-12-2025)
    GEMINI_VOICE        Voice preset (default: Aoede)
    SYSTEM_PROMPT       Custom system prompt
    MUTE_MIC            Mute mic during playback: 1 | 0 (default: 0, use with AEC)

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.providers.gemini.realtime import GeminiLiveProvider
from roomkit.voice.realtime.local_transport import LocalAudioTransport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("realtime_voice_local_gemini")


async def main() -> None:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Set GOOGLE_API_KEY to run this example.")
        print("  GOOGLE_API_KEY=... uv run python examples/realtime_voice_local_gemini.py")
        return

    kit = RoomKit()

    # --- Gemini Live provider (speech-to-speech) ---
    provider = GeminiLiveProvider(
        api_key=api_key,
        model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025"),
    )

    # --- Local audio transport: mic for input, speakers for output ---
    #
    # Tip: set PipeWire echo-cancel as default source/sink for full-duplex:
    #   wpctl set-default <echo-cancel-source-id>
    #   wpctl set-default <echo-cancel-sink-id>
    # Then MUTE_MIC=0 (default) works great — AEC strips speaker echo.
    # Without AEC, set MUTE_MIC=1 to prevent echo triggering barge-in.
    mute_mic = os.environ.get("MUTE_MIC", "0") == "1"

    transport = LocalAudioTransport(
        input_sample_rate=16000,
        output_sample_rate=24000,
        mute_mic_during_playback=mute_mic,
    )

    # --- Realtime voice channel ---
    channel = RealtimeVoiceChannel(
        "voice",
        provider=provider,
        transport=transport,
        system_prompt=os.environ.get(
            "SYSTEM_PROMPT",
            "You are a friendly voice assistant. Be concise and helpful.",
        ),
        voice=os.environ.get("GEMINI_VOICE", "Aoede"),
    )
    kit.register_channel(channel)

    # --- Room ---
    await kit.create_room(room_id="local-demo")
    await kit.attach_channel("local-demo", "voice")

    # --- Start session (connection=None for local transport) ---
    session = await channel.start_session(
        "local-demo",
        "local-user",
        connection=None,
    )

    logger.info("Gemini Live session started — speak into your microphone!")
    logger.info("Press Ctrl+C to stop.\n")

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
