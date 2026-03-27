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
import logging
import os
import signal

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.providers.personaplex.realtime import PersonaPlexRealtimeProvider
from roomkit.voice.backends.local import LocalAudioBackend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("realtime_voice_local_personaplex")


async def main() -> None:
    server_url = os.environ.get("PERSONAPLEX_URL", "wss://localhost:8998/api/chat")
    voice_prompt = os.environ.get("VOICE_PROMPT", "NATF2.pt")
    text_prompt = os.environ.get(
        "TEXT_PROMPT",
        "You are a friendly voice assistant. Be concise and helpful.",
    )
    seed = int(os.environ.get("SEED", "-1"))

    kit = RoomKit()

    # --- PersonaPlex provider (speech-to-speech) ---
    provider = PersonaPlexRealtimeProvider(server_url=server_url, seed=seed)

    # --- AEC (Acoustic Echo Cancellation) ---
    # PersonaPlex uses 24kHz audio natively
    sample_rate = 24000
    block_ms = 20
    frame_size = sample_rate * block_ms // 1000  # 480 samples

    aec = None
    aec_mode = os.environ.get("AEC", "webrtc").lower()
    if aec_mode in ("1", "webrtc"):
        from roomkit.voice.pipeline.aec.webrtc import WebRTCAECProvider

        aec = WebRTCAECProvider(sample_rate=sample_rate, enable_ns=True)
        logger.info("AEC enabled (WebRTC AEC3 + noise suppression)")
    elif aec_mode == "speex":
        from roomkit.voice.pipeline.aec.speex import SpeexAECProvider

        aec = SpeexAECProvider(
            frame_size=frame_size,
            filter_length=frame_size * 25,
            sample_rate=sample_rate,
        )
        logger.info("AEC enabled (Speex)")

    # --- Audio pipeline ---
    from roomkit.voice.pipeline.config import AudioPipelineConfig

    pipeline = AudioPipelineConfig(aec=aec) if aec else None

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
