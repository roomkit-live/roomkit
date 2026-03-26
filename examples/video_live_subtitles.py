"""Live translated subtitles on webcam video.

Captures webcam video and microphone audio.  STT transcribes your
speech, Claude translates it to English, and subtitles are rendered
directly on the video frames in real time.

Requires:
    pip install roomkit[local-video,local-audio,deepgram,sherpa-onnx,anthropic]

    A Deepgram API key for STT and an Anthropic API key for translation.

Run with:
    DEEPGRAM_API_KEY=... ANTHROPIC_API_KEY=... uv run python examples/video_live_subtitles.py
"""

from __future__ import annotations

import asyncio
import logging

from shared.env import require_env

from roomkit import (
    HookExecution,
    HookTrigger,
    RoomKit,
    VoiceChannel,
)
from roomkit.channels.video import VideoChannel
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig
from roomkit.video.pipeline.config import VideoPipelineConfig
from roomkit.video.pipeline.overlay import (
    Overlay,
    OverlayPosition,
    SubtitleManager,
)

logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("roomkit").setLevel(logging.WARNING)
logger = logging.getLogger("subtitles")
logger.setLevel(logging.INFO)


async def main() -> None:
    env = require_env("DEEPGRAM_API_KEY", "ANTHROPIC_API_KEY")

    # --- Providers -------------------------------------------------------

    # STT: Deepgram (supports French, English, etc.)
    from roomkit.voice import get_deepgram_config, get_deepgram_provider

    DeepgramSTTProvider = get_deepgram_provider()
    DeepgramConfig = get_deepgram_config()

    stt = DeepgramSTTProvider(
        DeepgramConfig(api_key=env["DEEPGRAM_API_KEY"], language="fr"),
    )

    # Translation via Claude Haiku (fast + cheap)
    translator = AnthropicAIProvider(
        AnthropicConfig(
            api_key=env["ANTHROPIC_API_KEY"],
            model="claude-haiku-4-5-20251001",
        )
    )

    async def translate_to_english(text: str) -> str:
        """Translate French text to English using Claude."""
        response = await translator.generate(
            messages=[],
            system=f"Translate the following French text to English. Return ONLY the translation, nothing else.\n\n{text}",
        )
        return response.text.strip() if response.text else text

    # --- Backends --------------------------------------------------------

    from roomkit.voice import get_local_audio_backend

    LocalAudioBackend = get_local_audio_backend()
    audio_backend = LocalAudioBackend(input_sample_rate=16000, output_sample_rate=24000)

    from roomkit.video import get_local_video_backend

    LocalVideoBackend = get_local_video_backend()
    video_backend = LocalVideoBackend(device=0, fps=15, width=640, height=480)

    # --- Overlay setup ---------------------------------------------------

    kit = RoomKit()

    # Subtitle overlay with French → English translation
    subtitle_mgr = SubtitleManager(
        kit,
        translate_fn=translate_to_english,
        max_lines=2,
        style={
            "font_scale": 0.8,
            "color": (255, 255, 0),
            "bg_color": (0, 0, 0),
            "padding": 10,
        },
    )
    overlay_filter = subtitle_mgr.overlay_filter

    # Static title
    overlay_filter.add_overlay(
        Overlay(
            id="title",
            content="FR -> EN Live Subtitles",
            position=OverlayPosition.TOP_LEFT,
            z_order=50,
            style={"font_scale": 0.5, "color": (180, 180, 180)},
        )
    )

    # --- Channels --------------------------------------------------------

    from roomkit.voice import get_sherpa_onnx_vad_config, get_sherpa_onnx_vad_provider
    from roomkit.voice.pipeline import AudioPipelineConfig

    SherpaOnnxVADProvider = get_sherpa_onnx_vad_provider()
    SherpaOnnxVADConfig = get_sherpa_onnx_vad_config()
    vad = SherpaOnnxVADProvider(SherpaOnnxVADConfig())

    voice = VoiceChannel(
        "voice",
        stt=stt,
        backend=audio_backend,
        pipeline=AudioPipelineConfig(vad=vad),
    )

    video = VideoChannel(
        "video",
        backend=video_backend,
        pipeline=VideoPipelineConfig(filters=[overlay_filter]),
    )

    kit.register_channel(voice)
    kit.register_channel(video)

    # --- Hooks -----------------------------------------------------------

    @kit.hook(HookTrigger.ON_TRANSCRIPTION, execution=HookExecution.ASYNC)
    async def log_transcription(event, ctx):
        logger.info("FR: %s", event.text)

    # --- Room + session --------------------------------------------------

    await kit.create_room(room_id="subtitle-demo")
    await kit.attach_channel("subtitle-demo", "voice")
    await kit.attach_channel("subtitle-demo", "video")

    session = await kit.join("subtitle-demo", "voice", participant_id="user")
    await kit.join("subtitle-demo", "video", participant_id="user")

    logger.info("Listening on mic + webcam. Speak French to see English subtitles.")
    logger.info("Press Ctrl+C to stop.")

    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        await kit.leave(session)
        await kit.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
