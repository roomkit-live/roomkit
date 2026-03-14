"""RoomKit -- Avatar video call: AI agent with a talking face.

Demonstrates the avatar system: when the AI speaks (TTS), a
lip-synced video of the agent's face is generated and sent to
the caller alongside the audio.

Modes:
  - **Mock mode** (default): MockAvatarProvider generates colored
    frames (green when speaking, gray when idle).  No GPU needed.
  - **MuseTalk mode**: Real lip-sync from a portrait photo.
    Requires GPU + MuseTalk installation.

Audio flow:  SIP INVITE → RTP audio → STT → AI → TTS → speaker
Video flow:  TTS audio → Avatar → video frames → RTP video

Prerequisites:
    pip install roomkit[sip,video]

    # For MuseTalk mode:
    git clone https://github.com/TMElyralab/MuseTalk.git
    cd MuseTalk && pip install -r requirements.txt

Run with:
    uv run python examples/avatar_call.py                    # mock avatar
    uv run python examples/avatar_call.py --musetalk /path/to/MuseTalk --image agent.png

Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
from collections.abc import AsyncIterator
from pathlib import Path

from roomkit import (
    AudioVideoChannel,
    HookResult,
    HookTrigger,
    RoomKit,
)
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
from roomkit.voice.base import AudioChunk, VoiceSession
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.base import TTSProvider


class SilenceTTSProvider(TTSProvider):
    """TTS that produces real PCM silence — triggers avatar frame generation."""

    @property
    def default_voice(self) -> str:
        return "silence"

    async def synthesize_stream(
        self,
        text: str,
        *,
        voice: str | None = None,
    ) -> AsyncIterator[AudioChunk]:
        # Generate ~2 seconds of 16kHz PCM silence (100 chunks × 20ms)
        chunk_samples = 320  # 20ms at 16kHz
        chunk_bytes = b"\x00" * (chunk_samples * 2)  # 16-bit PCM
        for i in range(100):
            yield AudioChunk(
                data=chunk_bytes,
                sample_rate=16000,
                is_final=(i == 99),
            )
            await asyncio.sleep(0.02)  # simulate real-time pacing


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("avatar_call")

if os.environ.get("DEBUG") == "1":
    logging.getLogger("roomkit").setLevel(logging.DEBUG)


def _build_avatar(args: argparse.Namespace) -> AvatarProvider:
    """Build avatar provider based on CLI args."""
    if args.musetalk:
        from roomkit.video.avatar.musetalk import MuseTalkAvatarProvider

        return MuseTalkAvatarProvider(
            musetalk_dir=args.musetalk,
            fps=30,
        )
    return MockAvatarProvider(
        fps=30,
        color=(0, 200, 0),  # green when speaking
        idle_color=(80, 80, 80),  # gray when idle
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Avatar Video Call Demo")
    parser.add_argument("--musetalk", default=None, help="Path to MuseTalk directory")
    parser.add_argument("--image", default=None, help="Reference portrait image (PNG/JPEG)")
    parser.add_argument("--sip-port", type=int, default=5060, help="SIP port")
    parser.add_argument("--rtp-ip", default="0.0.0.0", help="RTP IP")
    args = parser.parse_args()

    kit = RoomKit()

    # --- Avatar provider -------------------------------------------------------
    avatar = _build_avatar(args)

    # Start avatar with reference image (or blue placeholder)
    image_bytes = Path(args.image).read_bytes() if args.image else b"\x00\x00\x80" * (512 * 512)
    await avatar.start(image_bytes, width=512, height=512)

    # --- SIP A/V backend -------------------------------------------------------
    backend = SIPVideoBackend(
        local_sip_addr=("0.0.0.0", args.sip_port),  # nosec B104
        local_rtp_ip=args.rtp_ip,
        rtp_port_start=10000,
        supported_video_codecs=["H264", "VP9"],
    )

    # --- A/V channel with avatar -----------------------------------------------
    from roomkit.video.pipeline.config import VideoPipelineConfig

    av = AudioVideoChannel(
        "voice",
        stt=MockSTTProvider(),
        tts=SilenceTTSProvider(),
        backend=backend,
        pipeline=AudioPipelineConfig(),
        avatar=avatar,
        video_pipeline=VideoPipelineConfig(
            filters=[WatermarkFilter(text="AI Avatar {timestamp}", position="bottom-left")],
        ),
        recording=ChannelRecordingConfig(audio=True, video=True),
    )
    kit.register_channel(av)

    # --- Hooks -----------------------------------------------------------------
    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event, ctx):
        logger.info("User said: %s", event.text)
        # Trigger TTS response so avatar generates video frames
        return HookResult.allow(event=event)

    # --- Route incoming calls ---------------------------------------------------
    recorder = PyAVMediaRecorder()

    async def on_call(session: VoiceSession) -> None:
        room_id = session.metadata.get("room_id", session.id)
        caller = session.metadata.get("caller", "unknown")
        has_video = session.metadata.get("has_video", False)

        logger.info("Incoming call: room=%s, caller=%s, video=%s", room_id, caller, has_video)

        await kit.create_room(
            room_id=room_id,
            recorders=[
                RoomRecorderBinding(
                    recorder=recorder,
                    config=MediaRecordingConfig(storage="recordings"),
                ),
            ],
        )
        await kit.attach_channel(room_id, "voice")
        await kit.bind_voice_session(session, room_id, "voice")

        # Send a greeting via TTS — this triggers the avatar to generate
        # lip-synced video frames from the TTS audio output.
        logger.info("Sending greeting TTS (triggers avatar)...")
        await av._send_tts(session, "Hello, welcome to our service.")

    backend.on_call(on_call)

    # --- Disconnect handler (finalize recording) --------------------------------
    def on_call_ended(session: object) -> None:
        session_id = getattr(session, "id", "unknown")
        room_id = getattr(session, "room_id", None)
        logger.info("Call ended: session=%s", session_id)
        if room_id:
            asyncio.ensure_future(kit.close_room(room_id))

    backend.on_client_disconnected(on_call_ended)

    # --- Start -----------------------------------------------------------------
    await backend.start()

    mode = "MuseTalk" if args.musetalk else "Mock"
    print("Avatar Video Call Demo")
    print("=" * 60)
    print(f"Avatar  : {mode} ({avatar.name}, {avatar.fps}fps)")
    print(f"SIP     : 0.0.0.0:{args.sip_port}")
    print(f"Image   : {args.image or 'placeholder (blue)'}")
    print("Record  : ./recordings/")
    print("Press Ctrl+C to stop.\n")

    # --- Wait ------------------------------------------------------------------
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()

    # --- Cleanup ---------------------------------------------------------------
    logger.info("Stopping...")
    await avatar.stop()
    await backend.close()
    await kit.close()
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
