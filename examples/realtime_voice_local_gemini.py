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
    GEMINI_MODEL        Model name (default: gemini-2.5-flash-native-audio-preview-12-2025)
    GEMINI_VOICE        Voice preset (default: Aoede)
    SYSTEM_PROMPT       Custom system prompt
    AEC                 Echo cancellation: webrtc | speex | 1 (=webrtc) | 0
                        (default: webrtc)
    DENOISE             Enable RNNoise noise suppression: 1 | 0 (default: 0)
    MUTE_MIC            Mute mic during playback: 1 | 0 (default: auto,
                        off with AEC)

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

    # --- AEC (Acoustic Echo Cancellation) ---
    #
    # Strips speaker echo from the mic signal in real time, enabling
    # full-duplex conversation.  Disable with AEC=0 and use MUTE_MIC=1
    # as a fallback.
    #
    # AEC=webrtc (default) — WebRTC AEC3 (pip install aec-audio-processing)
    # AEC=speex            — SpeexDSP    (apt install libspeexdsp1)
    use_denoise = os.environ.get("DENOISE", "0") == "1"

    sample_rate = 24000  # Gemini outputs 24 kHz — mic must match for AEC
    block_ms = 20
    frame_size = sample_rate * block_ms // 1000  # 480 samples

    aec = None
    aec_mode = os.environ.get("AEC", "webrtc").lower()
    if aec_mode in ("1", "webrtc"):
        from roomkit.voice.pipeline.aec.webrtc import WebRTCAECProvider

        aec = WebRTCAECProvider(sample_rate=sample_rate)
        logger.info("AEC enabled (WebRTC AEC3)")
    elif aec_mode == "speex":
        from roomkit.voice.pipeline.aec.speex import SpeexAECProvider

        aec = SpeexAECProvider(
            frame_size=frame_size,
            filter_length=frame_size * 10,  # 200ms echo tail
            sample_rate=sample_rate,
        )
        logger.info("AEC enabled (Speex)")

    # --- Denoiser (RNNoise noise suppression) ---
    denoiser = None
    if use_denoise:
        from roomkit.voice.pipeline.denoiser.rnnoise import RNNoiseDenoiserProvider

        denoiser = RNNoiseDenoiserProvider(sample_rate=sample_rate)

    # When AEC is active it removes speaker echo from the mic signal, so we
    # can keep the mic open during playback.  Without AEC the mic is muted
    # during playback to prevent feedback loops.
    mute_env = os.environ.get("MUTE_MIC")
    mute_mic = mute_env != "0" if mute_env is not None else aec is None

    transport = LocalAudioTransport(
        input_sample_rate=sample_rate,
        output_sample_rate=sample_rate,
        block_duration_ms=block_ms,
        mute_mic_during_playback=mute_mic,
        aec=aec,
        denoiser=denoiser,
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
