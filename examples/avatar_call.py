"""RoomKit -- Avatar video call: AI agent with a talking face.

Demonstrates the avatar system: when the AI speaks (TTS), a
lip-synced video of the agent's face is generated and sent to
the caller alongside the audio.

Audio flow:  SIP INVITE → RTP audio → Deepgram STT → Claude AI → ElevenLabs TTS → speaker
Video flow:  TTS audio → Avatar → H.264 encode → RTP video → caller sees avatar

Modes:
  - **Mock avatar** (default): shows the reference image as a static frame.
  - **WebSocket avatar** (``--avatar-url``): connects to a remote animation
    server (any model speaking the protocol of
    ``roomkit.video.avatar.websocket``).

Reference image: no portrait ships with the examples. Put one (PNG/JPEG) at
``examples/avatar.png`` (next to this file) or pass ``--image PATH``; with
neither, a solid blue placeholder is used.

Prerequisites:
    pip install roomkit[sip,video,video-overlay,local-video,webrtc-aec,deepgram,elevenlabs,anthropic]
    # --avatar-url also needs: pip install roomkit[httpx,websocket]

Environment variables:
    DEEPGRAM_API_KEY     (required) Deepgram API key
    ELEVENLABS_API_KEY   (required) ElevenLabs API key
    ANTHROPIC_API_KEY    (required) Anthropic API key
    DEEPGRAM_MODEL       Deepgram model (default: nova-2)
    VOICE_LANGUAGE       Speech language (default: en)
    ELEVENLABS_VOICE_ID  ElevenLabs voice (default: Rachel)
    AI_MODEL             Claude model (default: claude-haiku-5-5)
    SYSTEM_PROMPT        System prompt override
    RECORDING_DIR        Record each call (MP4) into this directory;
                         unset = no recording
    RECORDING_ENCRYPTED_AT_REST
                         Required with RECORDING_DIR: set it to 1 to state
                         that RECORDING_DIR is on encrypted storage (RFC 17.6:
                         the recorder refuses plaintext recordings)
    DEBUG                Set to 1 for verbose logging

Run with:
    uv run python examples/avatar_call.py
    uv run python examples/avatar_call.py --image path/to/portrait.png
    uv run python examples/avatar_call.py --avatar-url http://gpu-server:8765

Press Ctrl+C to stop.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import asyncio
import io
import logging
import os

from PIL import Image
from shared import env_bool, require_env, run_until_stopped, setup_logging, voice_language

from roomkit import (
    AudioVideoChannel,
    ChannelCategory,
    RoomKit,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.anthropic.ai import AnthropicAIProvider, AnthropicConfig
from roomkit.recorder.base import (
    MediaRecordingConfig,
    RoomRecorderBinding,
)
from roomkit.recorder.pyav import PyAVMediaRecorder
from roomkit.video.avatar.base import AvatarProvider
from roomkit.video.avatar.mock import MockAvatarProvider
from roomkit.video.avatar.websocket import WebSocketAvatarProvider
from roomkit.video.backends.sip import SIPVideoBackend
from roomkit.video.pipeline.config import VideoPipelineConfig
from roomkit.video.pipeline.encoder.pyav import PyAVVideoEncoder
from roomkit.video.pipeline.filter.watermark import WatermarkFilter
from roomkit.voice.base import VoiceSession
from roomkit.voice.interruption import InterruptionConfig, InterruptionStrategy
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.pipeline.aec.webrtc import WebRTCAECProvider
from roomkit.voice.stt.deepgram import DeepgramConfig, DeepgramSTTProvider
from roomkit.voice.tts.elevenlabs import ElevenLabsConfig, ElevenLabsTTSProvider

logger = setup_logging("avatar_call")

if os.environ.get("DEBUG") == "1":
    logging.getLogger("roomkit").setLevel(logging.DEBUG)


DEFAULT_IMAGE = Path(__file__).resolve().parent / "avatar.png"
PLACEHOLDER_RGB = (0, 0, 128)  # navy blue


def _build_avatar(args: argparse.Namespace) -> AvatarProvider:
    if args.avatar_url:
        return WebSocketAvatarProvider(base_url=args.avatar_url, fps=30)
    return MockAvatarProvider(fps=30, color=(0, 200, 0), idle_color=(80, 80, 80))


def _reference_image(image: str | None, width: int, height: int) -> tuple[bytes, str]:
    """The portrait to animate and a label for the banner.

    ``--image`` wins, then ``avatar.png`` next to this file; otherwise a solid
    blue PNG. It must be an encoded image: the avatar decodes it, and raw
    pixel bytes would be rejected (the mock avatar then silently falls back
    to its own colours).
    """
    if image:
        path = Path(image)
        if not path.is_file():
            sys.exit(f"Error: --image {image} not found")
        return path.read_bytes(), str(path)
    if DEFAULT_IMAGE.is_file():
        return DEFAULT_IMAGE.read_bytes(), str(DEFAULT_IMAGE)
    buf = io.BytesIO()
    Image.new("RGB", (width, height), PLACEHOLDER_RGB).save(buf, format="PNG")
    return buf.getvalue(), "placeholder (blue)"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Avatar Video Call Demo")
    parser.add_argument("--avatar-url", default=None, help="Avatar service URL")
    parser.add_argument(
        "--image",
        default=None,
        help="Reference portrait (PNG/JPEG); default: avatar.png next to this file, else blue",
    )
    parser.add_argument(
        "--size",
        default="512x512",
        help="Avatar video size WxH (default 512x512)",
    )
    parser.add_argument("--sip-port", type=int, default=5060, help="SIP port")
    parser.add_argument("--rtp-ip", default="0.0.0.0", help="RTP IP")
    parser.add_argument(
        "--rtp-port-start", type=int, default=10000, help="First RTP port (below 20000)"
    )
    args = parser.parse_args()

    # Parse size
    avatar_width, avatar_height = (int(x) for x in args.size.split("x"))

    # --- API keys ---------------------------------------------------------------
    env = require_env("DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY", "ANTHROPIC_API_KEY")
    deepgram_key = env["DEEPGRAM_API_KEY"]
    elevenlabs_key = env["ELEVENLABS_API_KEY"]
    anthropic_key = env["ANTHROPIC_API_KEY"]

    kit = RoomKit()

    # --- STT: Deepgram ----------------------------------------------------------
    deepgram_model = os.environ.get("DEEPGRAM_MODEL", "nova-2")
    stt = DeepgramSTTProvider(
        config=DeepgramConfig(
            api_key=deepgram_key,
            model=deepgram_model,
            language=voice_language("en") or "en",
            punctuate=True,
            smart_format=True,
            endpointing=300,
        )
    )

    # --- TTS: ElevenLabs --------------------------------------------------------
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
    # A fast model: replies are one or two spoken sentences (256 tokens).
    ai_model = os.environ.get("AI_MODEL", "claude-haiku-5-5")
    ai_provider = AnthropicAIProvider(
        AnthropicConfig(
            api_key=anthropic_key,
            model=ai_model,
            max_tokens=256,
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
    image_bytes, image_label = _reference_image(args.image, avatar_width, avatar_height)
    await avatar.start(image_bytes, width=avatar_width, height=avatar_height)

    # --- SIP A/V backend --------------------------------------------------------
    backend = SIPVideoBackend(
        local_sip_addr=("0.0.0.0", args.sip_port),  # nosec B104
        local_rtp_ip=args.rtp_ip,
        rtp_port_start=args.rtp_port_start,
        supported_video_codecs=["H264"],
    )

    # --- AEC (echo cancellation) ------------------------------------------------
    # Prevents TTS audio reflecting back through the mic from triggering
    # false barge-in interruptions.
    aec = WebRTCAECProvider(sample_rate=16000)

    # --- H.264 encoder for avatar → RTP ----------------------------------------
    avatar_encoder = PyAVVideoEncoder(width=avatar_width, height=avatar_height, fps=avatar.fps)

    # --- A/V channel with avatar ------------------------------------------------
    # Disable interruption — SIP echo cancellation can't handle the
    # variable network delay, causing false barge-in triggers.
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

    # --- Recording (opt-in) ----------------------------------------------------
    recording_dir = os.environ.get("RECORDING_DIR", "")
    if recording_dir and not env_bool("RECORDING_ENCRYPTED_AT_REST", default=False):
        sys.exit(
            "Error: RECORDING_DIR needs RECORDING_ENCRYPTED_AT_REST=1, stating that "
            f"{recording_dir} is on encrypted storage (RFC 17.6 refuses plaintext recordings)"
        )

    def recorders() -> list[RoomRecorderBinding]:
        if not recording_dir:
            return []
        return [
            RoomRecorderBinding(
                recorder=PyAVMediaRecorder(),
                config=MediaRecordingConfig(
                    storage=recording_dir,
                    storage_encrypted_at_rest=True,  # stated by RECORDING_ENCRYPTED_AT_REST=1
                ),
            )
        ]

    # --- Route incoming calls ---------------------------------------------------
    async def on_call(session: VoiceSession) -> None:
        room_id = session.id
        caller = session.metadata.get("caller", "unknown")
        has_video = session.metadata.get("has_video", False)
        logger.info("Incoming call: room=%s, caller=%s, video=%s", room_id[:8], caller, has_video)

        await kit.create_room(room_id=room_id, recorders=recorders())
        await kit.attach_channel(room_id, "voice")
        await kit.attach_channel(room_id, "ai", category=ChannelCategory.INTELLIGENCE)
        await kit.join(room_id, "voice", session=session)

    backend.on_call(on_call)

    # --- Disconnect handler -----------------------------------------------------
    # on_call_disconnected fires on every BYE; SIPVideoBackend.on_client_disconnected
    # only reports the end of a call's video session.
    def on_call_ended(session: VoiceSession) -> None:
        logger.info("Call ended: session=%s", session.id[:8])
        asyncio.create_task(kit.close_room(session.room_id))

    backend.on_call_disconnected(on_call_ended)

    # --- Start ------------------------------------------------------------------
    await backend.start()

    mode = f"WebSocket ({args.avatar_url})" if args.avatar_url else "Mock"
    print("Avatar Video Call Demo")
    print("=" * 60)
    print(f"Avatar  : {mode} ({avatar.name}, {avatar.fps}fps, {avatar_width}x{avatar_height})")
    print(f"STT     : Deepgram {deepgram_model}")
    print("TTS     : ElevenLabs")
    print(f"AI      : {ai_model}")
    print(f"SIP     : 0.0.0.0:{args.sip_port}")
    print(f"Image   : {image_label}")
    print(
        f"Record  : {Path(recording_dir).resolve() if recording_dir else 'off (set RECORDING_DIR)'}"
    )
    print("Press Ctrl+C to stop.\n")

    # --- Wait -------------------------------------------------------------------
    async def cleanup() -> None:
        await avatar.stop()
        await backend.close()

    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    asyncio.run(main())
