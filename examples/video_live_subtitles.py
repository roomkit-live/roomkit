"""Live translated subtitles on webcam video.

Captures webcam video and microphone audio.  STT transcribes your
speech in French, Claude translates it to English, and subtitles
are rendered on the video frames in real time.

Open http://localhost:8089 in a browser to see the live video feed.

Requires:
    pip install roomkit[local-video,local-audio,deepgram,sherpa-onnx,anthropic]

Run with:
    VAD_MODEL=ten-vad.onnx DEEPGRAM_API_KEY=... ANTHROPIC_API_KEY=... \
        uv run python examples/video_live_subtitles.py
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from shared import run_until_stopped, setup_logging
from shared.env import require_env

from roomkit import HookExecution, HookTrigger, RoomKit, VoiceChannel
from roomkit.channels.video import VideoChannel
from roomkit.providers.ai.base import AIContext, AIMessage
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig
from roomkit.video.pipeline.config import VideoPipelineConfig
from roomkit.video.pipeline.filter.base import FilterContext, VideoFilterProvider
from roomkit.video.pipeline.overlay import Overlay, OverlayPosition, SubtitleManager
from roomkit.video.video_frame import VideoFrame

logger = setup_logging("subtitles")


# -- MJPEG HTTP server: view at http://localhost:8089 ----------------------

_latest_jpeg: bytes = b""
_jpeg_lock = threading.Lock()


class MJPEGHandler(BaseHTTPRequestHandler):
    """Serve latest frame as MJPEG stream."""

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        while True:
            with _jpeg_lock:
                jpeg = _latest_jpeg
            if not jpeg:
                import time

                time.sleep(0.05)
                continue
            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except BrokenPipeError:
                break

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass  # Suppress HTTP access logs


def _start_mjpeg_server(port: int = 8089) -> None:
    server = HTTPServer(("0.0.0.0", port), MJPEGHandler)  # noqa: S104
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("MJPEG server at http://localhost:%d", port)


class MJPEGFilter(VideoFilterProvider):
    """Encode each frame to JPEG and push to the MJPEG server."""

    def __init__(self) -> None:
        self._cv2: object | None = None

    @property
    def name(self) -> str:
        return "mjpeg"

    def filter(self, frame: VideoFrame, context: FilterContext) -> VideoFrame:
        global _latest_jpeg  # noqa: PLW0603
        if not frame.is_raw or frame.codec != "raw_rgb24":
            return frame
        if self._cv2 is None:
            import cv2

            self._cv2 = cv2
        arr = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 3)
        bgr = self._cv2.cvtColor(arr, self._cv2.COLOR_RGB2BGR)
        _, buf = self._cv2.imencode(".jpg", bgr, [self._cv2.IMWRITE_JPEG_QUALITY, 80])
        with _jpeg_lock:
            _latest_jpeg = buf.tobytes()
        return frame


# -- Main ------------------------------------------------------------------


async def main() -> None:
    env = require_env("DEEPGRAM_API_KEY", "ANTHROPIC_API_KEY", "VAD_MODEL")

    # --- STT (Deepgram, French) ------------------------------------------

    from roomkit.voice import get_deepgram_config, get_deepgram_provider

    stt = get_deepgram_provider()(
        get_deepgram_config()(api_key=env["DEEPGRAM_API_KEY"], language="fr"),
    )

    # --- Translation (Claude Haiku) --------------------------------------

    translator = AnthropicAIProvider(
        AnthropicConfig(
            api_key=env["ANTHROPIC_API_KEY"],
            model="claude-haiku-4-5-20251001",
        )
    )

    async def translate(text: str) -> str:
        ctx = AIContext(
            messages=[AIMessage(role="user", content=text)],
            system_prompt=(
                "Translate the following French text to English. "
                "Return ONLY the translation, nothing else."
            ),
            max_tokens=256,
        )
        resp = await translator.generate(ctx)
        result = resp.content.strip() if resp.content else text
        logger.info("EN: %s", result)
        return result

    # --- Backends --------------------------------------------------------

    from roomkit.video import get_local_video_backend
    from roomkit.voice import (
        get_local_audio_backend,
        get_sherpa_onnx_vad_config,
        get_sherpa_onnx_vad_provider,
    )
    from roomkit.voice.pipeline import AudioPipelineConfig

    audio_backend = get_local_audio_backend()(
        input_sample_rate=16000,
        output_sample_rate=24000,
    )
    video_backend = get_local_video_backend()(
        device=0,
        fps=15,
        width=640,
        height=480,
    )
    vad = get_sherpa_onnx_vad_provider()(
        get_sherpa_onnx_vad_config()(model=env["VAD_MODEL"]),
    )

    # --- Kit + overlays --------------------------------------------------

    kit = RoomKit()

    subtitle_mgr = SubtitleManager(
        kit,
        translate_fn=translate,
        max_lines=2,
        style={
            "font_scale": 0.8,
            "color": (255, 255, 0),
            "bg_color": (0, 0, 0),
            "padding": 10,
        },
    )

    subtitle_mgr.overlay_filter.add_overlay(
        Overlay(
            id="title",
            content="FR -> EN Live Subtitles",
            position=OverlayPosition.TOP_LEFT,
            z_order=50,
            style={"font_scale": 0.5, "color": (180, 180, 180)},
        )
    )

    # --- Channels --------------------------------------------------------

    _start_mjpeg_server()

    filters = [subtitle_mgr.overlay_filter, MJPEGFilter()]

    voice = VoiceChannel(
        "voice",
        stt=stt,
        backend=audio_backend,
        pipeline=AudioPipelineConfig(vad=vad),
    )
    video = VideoChannel(
        "video",
        backend=video_backend,
        pipeline=VideoPipelineConfig(filters=filters),
    )

    kit.register_channel(voice)
    kit.register_channel(video)

    @kit.hook(HookTrigger.ON_TRANSCRIPTION, execution=HookExecution.ASYNC)
    async def on_transcription(event, ctx):
        logger.info("FR: %s", event.text)

    # --- Run -------------------------------------------------------------

    await kit.create_room(room_id="subtitle-demo")
    await kit.attach_channel("subtitle-demo", "voice")
    await kit.attach_channel("subtitle-demo", "video")

    session = await kit.join("subtitle-demo", "voice", participant_id="user")
    await kit.join("subtitle-demo", "video", participant_id="user")

    logger.info("Open http://localhost:8089 in a browser to see the video.")
    logger.info("Speak French — English subtitles appear on the video.")
    logger.info("Ctrl+C to stop.")

    async def cleanup() -> None:
        await kit.leave(session)

    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    import asyncio

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
