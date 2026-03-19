"""RoomKit -- Avatar video call: AI agent with a talking face.

Demonstrates the avatar system: when the AI speaks (TTS), a
lip-synced video of the agent's face is generated and sent to
the caller alongside the audio.

Audio flow:  SIP INVITE → RTP audio → Deepgram STT → Claude AI → ElevenLabs TTS → speaker
Video flow:  TTS audio → Avatar → H.264 encode → RTP video → caller sees avatar

Modes:
  - **Mock avatar** (default): displays reference image as static frame.
  - **WebSocket avatar**: connects to a remote animation server (any model).

Prerequisites:
    pip install roomkit[sip,video,deepgram,elevenlabs,anthropic]

    Environment variables:
        DEEPGRAM_API_KEY=...
        ELEVENLABS_API_KEY=...
        ANTHROPIC_API_KEY=...

Run with:
    uv run python examples/avatar_call.py --image avatar.png
    uv run python examples/avatar_call.py --image avatar.png --avatar-url http://gpu-server:8765

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
    if args.avatar_url:
        from roomkit.video.avatar.websocket import WebSocketAvatarProvider

        return WebSocketAvatarProvider(base_url=args.avatar_url, fps=30)
    return MockAvatarProvider(fps=30, color=(0, 200, 0), idle_color=(80, 80, 80))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Avatar Video Call Demo")
    parser.add_argument("--avatar-url", default=None, help="Avatar service URL")
    parser.add_argument("--image", default=None, help="Reference portrait image")
    parser.add_argument(
        "--size",
        default="512x512",
        help="Avatar video size WxH (default 512x512)",
    )
    parser.add_argument("--sip-port", type=int, default=5060, help="SIP port")
    parser.add_argument("--rtp-ip", default="0.0.0.0", help="RTP IP")
    args = parser.parse_args()

    # Parse size
    avatar_width, avatar_height = (int(x) for x in args.size.split("x"))

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
    if args.image:
        image_bytes = Path(args.image).read_bytes()
    else:
        image_bytes = b"\x00\x00\x80" * (avatar_width * avatar_height)
    await avatar.start(image_bytes, width=avatar_width, height=avatar_height)

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

    avatar_encoder = PyAVVideoEncoder(width=avatar_width, height=avatar_height, fps=avatar.fps)

    # --- A/V channel with avatar ------------------------------------------------
    # Disable interruption — SIP echo cancellation can't handle the
    # variable network delay, causing false barge-in triggers.
    from roomkit.voice.interruption import InterruptionConfig, InterruptionStrategy

    av = AudioVideoChannel(
        "voice",
        stt=stt,
        tts=tts,
        backend=backend,
        pipeline=AudioPipelineConfig(aec=aec),
        interruption=InterruptionConfig(strategy=InterruptionStrategy.DISABLED),
        avatar=avatar,
        avatar_encoder=avatar_encoder,
        video_pipeline=VideoPipelineConfig(
            filters=[WatermarkFilter(text="AI Avatar {timestamp}", position="bottom-left")],
        ),
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

    mode = f"WebSocket ({args.avatar_url})" if args.avatar_url else "Mock"
    print("Avatar Video Call Demo")
    print("=" * 60)
    print(f"Avatar  : {mode} ({avatar.name}, {avatar.fps}fps, {avatar_width}x{avatar_height})")
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
