"""RoomKit — Speech-to-speech with OpenAI Realtime using local mic/speakers.

Talk to GPT Realtime using your system microphone — AI audio plays
through your speakers.  OpenAI handles turn detection server-side.

Requirements:
    pip install roomkit[realtime-openai,local-audio,webrtc-aec]

Run with:
    OPENAI_API_KEY=... uv run python examples/realtime_voice_local_openai.py

Environment variables:
    OPENAI_API_KEY      (required) OpenAI API key
    OPENAI_MODEL        Model name (default: the provider's own default)
    OPENAI_VOICE        Voice preset (default: alloy)
    OPENAI_LANGUAGE     Optional input transcription language, e.g. fr
    SYSTEM_PROMPT       Custom system prompt
    AEC                 Echo cancellation: webrtc | speex | 1 (=webrtc) | 0
                        (default: webrtc)
    DENOISE             webrtc (default) | rnnoise | sherpa | 0 to disable
    AEC_DELAY_MS        Optional measured speaker-to-mic delay (default: auto)
    BARGE_IN_GUARD_MS   Hide capture from server VAD while AEC converges
                        after playback starts (default: 2000; 0 disables)
    MUTE_MIC            Mute mic during playback: 1 | 0 (default: 1).
                        Set 0 for full-duplex/barge-in; a calibrated AEC delay
                        and headphones or a well-controlled room are advised.
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
    setup_console,
    setup_logging,
)

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.providers.openai.realtime import OpenAIRealtimeProvider
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.interruption import InterruptionConfig

logger = setup_logging("realtime_voice_local_openai")


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY to run this example.")
        print("  OPENAI_API_KEY=... uv run python examples/realtime_voice_local_openai.py")
        return

    kit = RoomKit()

    # --- Console dashboard (set CONSOLE=1 to enable) ---
    console_cleanup = setup_console(kit)

    # --- OpenAI Realtime provider (speech-to-speech) ---
    # No model id hardcoded here: OPENAI_MODEL overrides, otherwise the
    # provider's default applies, so the id lives in exactly one place.
    model = os.environ.get("OPENAI_MODEL")
    provider = OpenAIRealtimeProvider(
        api_key=api_key,
        **({"model": model} if model else {}),
    )

    # --- Audio pipeline stages ---
    sample_rate = 24000  # OpenAI Realtime uses 24 kHz for both directions
    block_ms = 20

    aec = build_aec(sample_rate, block_ms, default="webrtc")
    # Keep noise suppression separate from AEC so it remains active between
    # playback turns. OpenAI cancels a response as soon as its server VAD sees
    # speech, so passing residual echo/noise through here causes cut/reply loops.
    denoiser = build_denoiser(sample_rate, default="webrtc")
    debug_taps = build_debug_taps()
    try:
        barge_in_guard_ms = max(0, int(os.environ.get("BARGE_IN_GUARD_MS", "2000")))
    except ValueError:
        logger.warning("Invalid BARGE_IN_GUARD_MS; using 2000ms")
        barge_in_guard_ms = 2000
    interruption = (
        InterruptionConfig(allow_during_first_ms=barge_in_guard_ms) if barge_in_guard_ms else None
    )
    pipeline = build_pipeline(
        aec=aec,
        denoiser=denoiser,
        debug_taps=debug_taps,
        interruption=interruption,
    )
    if barge_in_guard_ms:
        logger.info(
            "Barge-in guard enabled (%dms after physical playback starts)",
            barge_in_guard_ms,
        )

    # OpenAI's WebSocket VAD creates a turn from any detected speech even after
    # response.done, while generated audio may still be draining from the local
    # speaker buffer. Default to half-duplex so speaker leakage cannot create
    # recursive turns. Full-duplex remains an explicit opt-in for calibrated
    # AEC setups (or headphones).
    mute_env = os.environ.get("MUTE_MIC")
    mute_mic = mute_env != "0" if mute_env is not None else True
    if mute_mic:
        logger.info("OpenAI local playback guard: microphone muted while assistant speaks")
    else:
        logger.warning(
            "OpenAI full-duplex enabled by MUTE_MIC=%r: server VAD may react to residual "
            "speaker echo; unset MUTE_MIC for stable half-duplex",
            mute_env,
        )

    provider_config: dict[str, str] = {}
    transcription_language = os.environ.get("OPENAI_LANGUAGE")
    if transcription_language:
        provider_config["language"] = transcription_language
        logger.info("OpenAI input transcription language pinned to %s", transcription_language)

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
        metadata={"provider_config": provider_config} if provider_config else None,
    )

    logger.info("OpenAI Realtime session started")
    logger.info("Speak into your microphone! Press Ctrl+C to stop.\n")

    # --- Keep running until Ctrl+C ---
    async def cleanup() -> None:
        if console_cleanup:
            await console_cleanup()
        await channel.end_session(session)

    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    asyncio.run(main())
