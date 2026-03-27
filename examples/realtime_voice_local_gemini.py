"""RoomKit — Speech-to-speech with Gemini Live using local mic/speakers.

Talk to Gemini using your system microphone — AI audio plays through
your speakers.  No browser, no WebSocket, no server required.

Requirements:
    pip install roomkit[realtime-gemini,local-audio]
    System (optional): libspeexdsp for Speex AEC (apt install libspeexdsp1)
    System (optional): librnnoise for noise suppression

Run with:
    GOOGLE_API_KEY=... uv run python examples/realtime_voice_local_gemini.py

Environment variables:
    GOOGLE_API_KEY      (required) Google API key
    GEMINI_MODEL        Model name (default: gemini-3.1-flash-live-preview)
    GEMINI_VOICE        Voice preset (default: Aoede)
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
    build_vad,
    run_until_stopped,
    setup_console,
    setup_logging,
)

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.providers.gemini.realtime import GeminiLiveProvider
from roomkit.skills import SkillRegistry
from roomkit.voice.backends.local import LocalAudioBackend

logger = setup_logging("realtime_voice_local_gemini")

SKILLS_DIR = Path(__file__).parent / "skills"


async def main() -> None:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Set GOOGLE_API_KEY to run this example.")
        print("  GOOGLE_API_KEY=... uv run python examples/realtime_voice_local_gemini.py")
        return

    kit = RoomKit()

    # --- Console dashboard (set CONSOLE=1 to enable) ---
    console_cleanup = setup_console(kit)

    # --- Discover skills from examples/skills/ ---
    registry = SkillRegistry()
    count = registry.discover(SKILLS_DIR)
    if count:
        logger.info("Discovered %d skill(s): %s", count, ", ".join(registry.skill_names))

    # --- Gemini Live provider (speech-to-speech) ---
    provider = GeminiLiveProvider(
        api_key=api_key,
        model=os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-live-preview"),
    )

    # --- Audio pipeline stages ---
    sample_rate = 24000  # Gemini outputs 24 kHz — mic must match for AEC
    block_ms = 20

    aec = build_aec(sample_rate, block_ms, default="webrtc")
    denoiser = build_denoiser(sample_rate, default="rnnoise")
    vad = build_vad(sample_rate, default="ten")
    debug_taps = build_debug_taps()
    pipeline = build_pipeline(aec=aec, denoiser=denoiser, vad=vad, debug_taps=debug_taps)

    # When AEC is active it removes speaker echo from the mic signal, so we
    # can keep the mic open during playback (barge-in enabled).  Without AEC
    # the mic is muted during playback to prevent feedback loops.
    mute_env = os.environ.get("MUTE_MIC")
    mute_mic = mute_env != "0" if mute_env is not None else aec is None

    if mute_mic:
        logger.warning(
            "Barge-in disabled — mic muted during playback (no AEC). "
            "Install AEC for barge-in: pip install aec-audio-processing"
        )
    else:
        logger.info("Barge-in enabled (AEC active, mic stays open during playback)")

    transport = LocalAudioBackend(
        input_sample_rate=sample_rate,
        output_sample_rate=sample_rate,
        block_duration_ms=block_ms,
        mute_mic_during_playback=mute_mic,
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
            "You are a friendly voice assistant. Be concise and helpful.",
        ),
        voice=os.environ.get("GEMINI_VOICE", "Aoede"),
        input_sample_rate=sample_rate,
        skills=registry,
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
    async def cleanup() -> None:
        if console_cleanup:
            await console_cleanup()
        await channel.end_session(session)

    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    asyncio.run(main())
