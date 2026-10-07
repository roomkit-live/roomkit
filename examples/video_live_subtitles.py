"""Live translated subtitles on webcam video.

Captures webcam video and microphone audio.  STT transcribes your
speech in French, Claude translates it to English, and subtitles
are rendered on the video frames in real time.

Open http://127.0.0.1:8089 in a browser to see the live video feed
(an MJPEG stream: each new frame is sent once, at most 15 per second).
The server listens on 127.0.0.1 only: it has no authentication, so
binding it to another interface (MJPEG_HOST=0.0.0.0) shows your webcam
to anyone who can reach that address.

Requires:
    pip install roomkit[local-video,local-audio,deepgram,sherpa-onnx,anthropic]

    # VAD model (sherpa-onnx TEN-VAD):
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.onnx

Environment variables:
    DEEPGRAM_API_KEY   Deepgram key (French STT)
    ANTHROPIC_API_KEY  Anthropic key (translation)
    VAD_MODEL          path to the VAD model file (ten-vad.onnx above)
    MJPEG_HOST         (optional) interface of the video server, default 127.0.0.1
    MJPEG_PORT         (optional) port of the video server, default 8089

Run with:
    VAD_MODEL=ten-vad.onnx DEEPGRAM_API_KEY=... ANTHROPIC_API_KEY=... \\
        uv run python examples/video_live_subtitles.py
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from shared import run_until_stopped, setup_logging
from shared.env import require_env

from roomkit import HookExecution, HookTrigger, RoomKit, VoiceChannel
from roomkit.channels.video import VideoChannel
from roomkit.providers.ai.base import AIContext, AIMessage
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig
from roomkit.video import get_local_video_backend
from roomkit.video.pipeline.config import VideoPipelineConfig
from roomkit.video.pipeline.filter.base import FilterContext, VideoFilterProvider
from roomkit.video.pipeline.overlay import Overlay, OverlayPosition, SubtitleManager
from roomkit.video.video_frame import VideoFrame
from roomkit.voice import (
    get_deepgram_config,
    get_deepgram_provider,
    get_local_audio_backend,
    get_sherpa_onnx_vad_config,
    get_sherpa_onnx_vad_provider,
)
from roomkit.voice.pipeline import AudioPipelineConfig

logger = setup_logging("subtitles")

CAPTURE_FPS = 15


# -- MJPEG HTTP server: view at http://127.0.0.1:8089 -----------------------


class LatestFrame:
    """The latest JPEG and its number, shared by the pipeline and HTTP threads."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._jpeg = b""
        self._seq = 0
        self._closed = False

    def publish(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._cond.notify_all()

    def wait_newer(self, seq: int, timeout: float) -> tuple[int, bytes] | None:
        """Block until a frame newer than *seq* exists; ``None`` once closed."""
        with self._cond:
            self._cond.wait_for(lambda: self._closed or self._seq != seq, timeout)
            if self._closed:
                return None
            return self._seq, self._jpeg

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()


class MJPEGHandler(BaseHTTPRequestHandler):
    """Serve each new frame once, at most ``max_fps`` frames per second."""

    frames: LatestFrame
    max_fps: float = CAPTURE_FPS

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        min_gap = 1.0 / self.max_fps
        sent_seq = 0
        last_sent = 0.0
        while True:
            latest = self.frames.wait_newer(sent_seq, timeout=1.0)
            if latest is None:
                break  # server stopping
            seq, jpeg = latest
            if seq == sent_seq:
                continue  # no new frame within the timeout
            delay = last_sent + min_gap - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break
            sent_seq = seq
            last_sent = time.monotonic()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass  # Suppress HTTP access logs


def _start_mjpeg_server(frames: LatestFrame, host: str, port: int) -> ThreadingHTTPServer:
    handler = type("BoundMJPEGHandler", (MJPEGHandler,), {"frames": frames})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("MJPEG server at http://%s:%d", host, port)
    return server


class MJPEGFilter(VideoFilterProvider):
    """Encode each frame to JPEG and publish it to the MJPEG server."""

    def __init__(self, frames: LatestFrame) -> None:
        self._frames = frames

    @property
    def name(self) -> str:
        return "mjpeg"

    def filter(self, frame: VideoFrame, context: FilterContext) -> VideoFrame:
        if not frame.is_raw or frame.codec != "raw_rgb24":
            return frame
        arr = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 3)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        self._frames.publish(buf.tobytes())
        return frame


# -- Main ------------------------------------------------------------------


async def main() -> None:
    env = require_env("DEEPGRAM_API_KEY", "ANTHROPIC_API_KEY", "VAD_MODEL")
    host = os.environ.get("MJPEG_HOST", "127.0.0.1")
    port = int(os.environ.get("MJPEG_PORT", "8089"))

    # --- STT (Deepgram, French) ------------------------------------------

    stt = get_deepgram_provider()(
        get_deepgram_config()(api_key=env["DEEPGRAM_API_KEY"], language="fr"),
    )

    # --- Translation (Claude Haiku) --------------------------------------

    translator = AnthropicAIProvider(
        AnthropicConfig(
            api_key=env["ANTHROPIC_API_KEY"],
            model="claude-haiku-5-5",
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

    audio_backend = get_local_audio_backend()(
        input_sample_rate=16000,
        output_sample_rate=24000,
    )
    video_backend = get_local_video_backend()(
        device=0,
        fps=CAPTURE_FPS,
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
            "font_scale": 0.45,
            "color": (255, 255, 255),
            "bg_color": (0, 0, 0),
            "padding": 6,
            "thickness": 1,
        },
    )

    subtitle_mgr.overlay_filter.add_overlay(
        Overlay(
            id="title",
            content="FR -> EN Live Subtitles",
            position=OverlayPosition.TOP_LEFT,
            z_order=50,
            style={"font_scale": 0.35, "color": (150, 150, 150)},
        )
    )

    # --- Channels --------------------------------------------------------

    frames = LatestFrame()
    server = _start_mjpeg_server(frames, host, port)

    filters: list[VideoFilterProvider] = [subtitle_mgr.overlay_filter, MJPEGFilter(frames)]

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
    video_session = await kit.join("subtitle-demo", "video", participant_id="user")

    # Start webcam capture (connect() creates the session, start_capture fires the frames)
    await video_backend.start_capture(video_session)

    logger.info("Open http://%s:%d in a browser to see the video.", host, port)
    logger.info("Speak French — English subtitles appear on the video.")
    logger.info("Ctrl+C to stop.")

    async def cleanup() -> None:
        await video_backend.stop_capture(video_session)
        await kit.leave(video_session)
        await kit.leave(session)
        frames.close()  # releases the HTTP client threads
        await asyncio.to_thread(server.shutdown)
        server.server_close()

    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
