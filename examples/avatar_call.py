"""RoomKit -- Avatar video call: AI agent with a talking face.

Demonstrates the avatar system: when the AI speaks (TTS), a
lip-synced video of the agent's face is generated and sent to
the caller alongside the audio.

Audio flow:  SIP INVITE → RTP audio → Deepgram STT → Claude AI → ElevenLabs TTS → speaker
Video flow:  TTS audio → Avatar → H.264 encode → RTP video → caller sees avatar

Modes:
  - **Mock avatar** (default): displays reference image as static frame.
  - **MuseTalk**: real lip-sync from portrait photo (GPU required).

Prerequisites:
    pip install roomkit[sip,video,deepgram,elevenlabs,anthropic]

    Environment variables:
        DEEPGRAM_API_KEY=...
        ELEVENLABS_API_KEY=...
        ANTHROPIC_API_KEY=...

    # For MuseTalk mode:
    git clone https://github.com/TMElyralab/MuseTalk.git
    cd MuseTalk && pip install -r requirements.txt

Run with:
    uv run python examples/avatar_call.py --image avatar.png
    uv run python examples/avatar_call.py --image avatar.png --musetalk /path/to/MuseTalk

Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
from pathlib import Path

from roomkit import (
    AudioVideoChannel,
    ChannelCategory,
    RoomKit,
)
from roomkit.channels.ai import AIChannel
from roomkit.recorder.base import (
    ChannelRecordingConfig,
    MediaRecordingConfig,
    RoomRecorderBinding,
)
from roomkit.recorder.pyav import PyAVMediaRecorder
from roomkit.video.avatar.base import AvatarProvider
from roomkit.video.avatar.mock import MockAvatarProvider
from roomkit.video.backends.sip import SIPVideoBackend
from roomkit.video.pipeline.filter.watermark import WatermarkFilter
from roomkit.voice.base import VoiceSession
from roomkit.voice.pipeline import AudioPipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("avatar_call")

if os.environ.get("DEBUG") == "1":
    logging.getLogger("roomkit").setLevel(logging.DEBUG)


def _require_env(key: str) -> str:
    val = os.environ.get(key, "")
    if not val:
        raise SystemExit(f"Missing environment variable: {key}")
    return val


def _build_avatar(args: argparse.Namespace) -> AvatarProvider:
    if args.musetalk:
        from roomkit.video.avatar.musetalk import MuseTalkAvatarProvider

        return MuseTalkAvatarProvider(musetalk_dir=args.musetalk, fps=30)
    return MockAvatarProvider(fps=30, color=(0, 200, 0), idle_color=(80, 80, 80))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Avatar Video Call Demo")
    parser.add_argument("--musetalk", default=None, help="Path to MuseTalk directory")
    parser.add_argument("--image", default=None, help="Reference portrait image")
    parser.add_argument("--sip-port", type=int, default=5060, help="SIP port")
    parser.add_argument("--rtp-ip", default="0.0.0.0", help="RTP IP")
    args = parser.parse_args()

    # --- API keys ---------------------------------------------------------------
    deepgram_key = _require_env("DEEPGRAM_API_KEY")
    elevenlabs_key = _require_env("ELEVENLABS_API_KEY")
    anthropic_key = _require_env("ANTHROPIC_API_KEY")

    kit = RoomKit()

    # --- STT: Deepgram ----------------------------------------------------------
    from roomkit.voice.stt.deepgram import DeepgramConfig, DeepgramSTTProvider

    stt = DeepgramSTTProvider(
        config=DeepgramConfig(
            api_key=deepgram_key,
            model=os.environ.get("DEEPGRAM_MODEL", "nova-2"),
            language=os.environ.get("LANGUAGE", "en"),
            punctuate=True,
            smart_format=True,
            endpointing=300,
        )
    )

    # --- TTS: ElevenLabs --------------------------------------------------------
    from roomkit.voice.tts.elevenlabs import ElevenLabsConfig, ElevenLabsTTSProvider

    tts = ElevenLabsTTSProvider(
        config=ElevenLabsConfig(
            api_key=elevenlabs_key,
            voice_id=os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
            model_id="eleven_multilingual_v2",
            output_format="pcm_16000",
            optimize_streaming_latency=3,
        )
    )

    # --- AI: Claude -------------------------------------------------------------
    from roomkit.providers.anthropic.ai import AnthropicAIProvider, AnthropicConfig

    ai_provider = AnthropicAIProvider(
        AnthropicConfig(
            api_key=anthropic_key,
            model=os.environ.get("AI_MODEL", "claude-sonnet-4-20250514"),
            max_tokens=256,
            temperature=0.7,
        )
    )

    system_prompt = os.environ.get(
        "SYSTEM_PROMPT",
        "You are a friendly AI assistant with a video avatar. "
        "Keep your responses short and conversational — one or two sentences. "
        "The caller can see your face on their screen.",
    )

    # --- Avatar -----------------------------------------------------------------
    avatar = _build_avatar(args)
    image_bytes = Path(args.image).read_bytes() if args.image else b"\x00\x00\x80" * (512 * 512)
    await avatar.start(image_bytes, width=512, height=512)

    # --- SIP A/V backend --------------------------------------------------------
    backend = SIPVideoBackend(
        local_sip_addr=("0.0.0.0", args.sip_port),  # nosec B104
        local_rtp_ip=args.rtp_ip,
        rtp_port_start=10000,
        supported_video_codecs=["H264"],
    )

    # --- AEC (echo cancellation) ------------------------------------------------
    # Prevents TTS audio reflecting back through the mic from triggering
    # false barge-in interruptions.
    from roomkit.voice.pipeline.aec.webrtc import WebRTCAECProvider

    aec = WebRTCAECProvider(sample_rate=16000)

    # --- H.264 encoder for avatar → RTP ----------------------------------------
    from roomkit.video.pipeline.config import VideoPipelineConfig
    from roomkit.video.pipeline.encoder.pyav import PyAVVideoEncoder

    avatar_encoder = PyAVVideoEncoder(width=512, height=512, fps=avatar.fps)

    # --- A/V channel with avatar ------------------------------------------------
    av = AudioVideoChannel(
        "voice",
        stt=stt,
        tts=tts,
        backend=backend,
        pipeline=AudioPipelineConfig(aec=aec),
        avatar=avatar,
        avatar_encoder=avatar_encoder,
        video_pipeline=VideoPipelineConfig(
            filters=[WatermarkFilter(text="AI Avatar {timestamp}", position="bottom-left")],
        ),
        recording=ChannelRecordingConfig(audio=True, video=True),
    )
    kit.register_channel(av)

    # --- AI channel -------------------------------------------------------------
    ai = AIChannel("ai", provider=ai_provider, system_prompt=system_prompt)
    kit.register_channel(ai)

    # --- Route incoming calls ---------------------------------------------------
    async def on_call(session: VoiceSession) -> None:
        room_id = session.id
        caller = session.metadata.get("caller", "unknown")
        has_video = session.metadata.get("has_video", False)
        logger.info("Incoming call: room=%s, caller=%s, video=%s", room_id[:8], caller, has_video)

        await kit.create_room(
            room_id=room_id,
            recorders=[
                RoomRecorderBinding(
                    recorder=PyAVMediaRecorder(),
                    config=MediaRecordingConfig(storage="recordings"),
                ),
            ],
        )
        await kit.attach_channel(room_id, "voice")
        await kit.attach_channel(room_id, "ai", category=ChannelCategory.INTELLIGENCE)
        await kit.bind_voice_session(session, room_id, "voice")

    backend.on_call(on_call)

    # --- Disconnect handler -----------------------------------------------------
    def on_call_ended(session: object) -> None:
        session_id = getattr(session, "id", None)
        if not session_id:
            return
        logger.info("Call ended: session=%s", session_id[:8])
        asyncio.create_task(kit.close_room(session_id))

    backend.on_client_disconnected(on_call_ended)

    # --- Start ------------------------------------------------------------------
    await backend.start()

    mode = "MuseTalk" if args.musetalk else "Mock"
    print("Avatar Video Call Demo")
    print("=" * 60)
    print(f"Avatar  : {mode} ({avatar.name}, {avatar.fps}fps)")
    print("STT     : Deepgram nova-3")
    print("TTS     : ElevenLabs")
    print("AI      : Claude")
    print(f"SIP     : 0.0.0.0:{args.sip_port}")
    print(f"Image   : {args.image or 'placeholder (blue)'}")
    print("Record  : ./recordings/")
    print("Press Ctrl+C to stop.\n")

    # --- Wait -------------------------------------------------------------------
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()

    # --- Cleanup ----------------------------------------------------------------
    logger.info("Stopping...")
    await avatar.stop()
    await backend.close()
    await kit.close()
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
