"""RoomKit — Speech-to-speech with NVIDIA PersonaPlex using local mic/speakers.

Talk to a PersonaPlex model using your system microphone — AI audio plays
through your speakers.  PersonaPlex handles turn detection, interruptions,
and backchannels natively (Moshi architecture, 7B parameters).

Requirements:
    pip install roomkit[local-audio] websockets 'sphn>=0.1.4,<0.2' numpy
    A running PersonaPlex server (GPU: A100/H100 recommended):
        git clone https://github.com/NVIDIA/personaplex
        cd personaplex && pip install -r requirements.txt
        python -m moshi.server --host 0.0.0.0 --port 8998

Run with:
    PERSONAPLEX_URL=wss://gpu-host:8998/api/chat \
        uv run python examples/realtime_voice_local_personaplex.py

Environment variables:
    PERSONAPLEX_URL     Server WebSocket URL (default: wss://localhost:8998/api/chat)
    VOICE_PROMPT        Voice preset file (default: NATF2.pt)
                        Natural: NATF0-3 (female), NATM0-3 (male)
                        Varied:  VARF0-4 (female), VARM0-4 (male)
    TEXT_PROMPT          Persona description (default: friendly assistant)
    SEED                Random seed (-1 = disabled, default: -1)
    AEC                 Echo cancellation: webrtc | speex | 0 (default: webrtc)
    MUTE_MIC            Mute mic during playback: 1 | 0 (default: auto)

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import build_aec, build_pipeline, run_until_stopped, setup_console, setup_logging

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.providers.personaplex.realtime import PersonaPlexRealtimeProvider
from roomkit.voice.backends.local import LocalAudioBackend

logger = setup_logging("realtime_voice_local_personaplex")


async def main() -> None:
    server_url = os.environ.get("PERSONAPLEX_URL", "wss://localhost:8998/api/chat")
    voice_prompt = os.environ.get("VOICE_PROMPT", "NATF2.pt")
    text_prompt = os.environ.get(
        "TEXT_PROMPT",
        "You are a friendly voice assistant. Be concise and helpful.",
    )
    seed = int(os.environ.get("SEED", "-1"))

    kit = RoomKit()

    # --- Console dashboard (set CONSOLE=1 to enable) ---
    console_cleanup = setup_console(kit)

    # --- PersonaPlex provider (speech-to-speech) ---
    provider = PersonaPlexRealtimeProvider(server_url=server_url, seed=seed)

    # --- Audio pipeline (AEC) ---
    # PersonaPlex uses 24kHz audio natively
    sample_rate = 24000
    block_ms = 20

    aec = build_aec(sample_rate, block_ms, default="webrtc")
    pipeline = build_pipeline(aec=aec)

    mute_env = os.environ.get("MUTE_MIC")
    mute_mic = mute_env != "0" if mute_env is not None else aec is None

    transport = LocalAudioBackend(
        input_sample_rate=sample_rate,
        output_sample_rate=sample_rate,
        block_duration_ms=block_ms,
        mute_mic_during_playback=mute_mic,
    )

    # --- Realtime voice channel ---
    channel = RealtimeVoiceChannel(
        "voice",
        provider=provider,
        transport=transport,
        system_prompt=text_prompt,
        voice=voice_prompt,
        input_sample_rate=sample_rate,
        output_sample_rate=sample_rate,
        pipeline=pipeline,
    )
    kit.register_channel(channel)

    # --- Room ---
    await kit.create_room(room_id="personaplex-demo")
    await kit.attach_channel("personaplex-demo", "voice")

    # --- Start session ---
    session = await channel.start_session(
        "personaplex-demo",
        "local-user",
        connection=None,
    )

    logger.info("PersonaPlex session started (voice=%s)", voice_prompt)
    logger.info("Speak into your microphone! Press Ctrl+C to stop.\n")

    # --- Keep running until Ctrl+C ---
    async def cleanup() -> None:
        if console_cleanup:
            await console_cleanup()
        await channel.end_session(session)

    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    asyncio.run(main())
