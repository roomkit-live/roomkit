"""RoomKit — Speech-to-speech with OpenAI Realtime using local mic/speakers.

Talk to GPT Realtime 1.5 using your system microphone — AI audio plays
through your speakers.  OpenAI handles turn detection server-side.

Requirements:
    pip install roomkit[realtime-openai,local-audio]
    System (optional): libspeexdsp for Speex AEC (apt install libspeexdsp1)
    System (optional): librnnoise for noise suppression

Run with:
    OPENAI_API_KEY=... uv run python examples/realtime_voice_local_openai.py

Environment variables:
    OPENAI_API_KEY      (required) OpenAI API key
    OPENAI_MODEL        Model name (default: gpt-realtime-1.5)
    OPENAI_VOICE        Voice preset (default: alloy)
    SYSTEM_PROMPT       Custom system prompt
    AEC                 Echo cancellation: webrtc | speex | 1 (=webrtc) | 0
                        (default: webrtc)
    DENOISE             Enable RNNoise noise suppression: 1 | 0 (default: 0)
    MUTE_MIC            Mute mic during playback: 1 | 0 (default: auto,
                        off with AEC)
    DEBUG_AUDIO         Save pipeline stage WAVs to ./debug_audio/: 1 | 0
                        (default: 0)

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
    build_debug_taps,
    build_denoiser,
    build_pipeline,
    run_until_stopped,
    setup_logging,
)

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.providers.openai.realtime import OpenAIRealtimeProvider
from roomkit.voice.backends.local import LocalAudioBackend

logger = setup_logging("realtime_voice_local_openai")


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY to run this example.")
        print("  OPENAI_API_KEY=... uv run python examples/realtime_voice_local_openai.py")
        return

    kit = RoomKit()

    # --- OpenAI Realtime provider (speech-to-speech) ---
    provider = OpenAIRealtimeProvider(
        api_key=api_key,
        model=os.environ.get("OPENAI_MODEL", "gpt-realtime-1.5"),
    )

    # --- Audio pipeline stages ---
    sample_rate = 24000  # OpenAI Realtime uses 24 kHz for both directions
    block_ms = 20

    aec = build_aec(sample_rate, block_ms, default="webrtc")
    denoiser = build_denoiser(sample_rate, default="rnnoise")
    debug_taps = build_debug_taps()
    pipeline = build_pipeline(aec=aec, denoiser=denoiser, debug_taps=debug_taps)

    # When AEC is active it removes speaker echo from the mic signal, so we
    # can keep the mic open during playback.  Without AEC the mic is muted
    # during playback to prevent feedback loops.
    mute_env = os.environ.get("MUTE_MIC")
    mute_mic = mute_env != "0" if mute_env is not None else aec is None

    transport = LocalAudioBackend(
        input_sample_rate=sample_rate,
        output_sample_rate=sample_rate,
        block_duration_ms=block_ms,
        mute_mic_during_playback=mute_mic,
        aec=aec,
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
        voice=os.environ.get("OPENAI_VOICE", "alloy"),
        input_sample_rate=24000,
        output_sample_rate=24000,
        pipeline=pipeline,
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

    logger.info("OpenAI Realtime session started")
    logger.info("Speak into your microphone! Press Ctrl+C to stop.\n")

    # --- Keep running until Ctrl+C ---
    async def cleanup() -> None:
        await channel.end_session(session)

    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    asyncio.run(main())
