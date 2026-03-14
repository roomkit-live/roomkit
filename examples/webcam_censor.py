"""RoomKit -- Webcam censor: black out video when a person is detected.

Demonstrates the video pipeline filter stage.  Vision periodically
analyzes frames and when it detects a "person" label, the censor
filter replaces all subsequent frames with a black image until
the person leaves the frame.

Pipeline:  Camera → [CensorFilter] → Vision (periodic) → display/taps

Prerequisites:
    pip install roomkit[local-video]

    # For Gemini vision:
    pip install roomkit[gemini]
    export GEMINI_API_KEY=AIza...

Run with:
    uv run python examples/webcam_censor.py                  # mock vision
    uv run python examples/webcam_censor.py --gemini         # real vision

Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal

from roomkit import (
    GeminiVisionConfig,
    GeminiVisionProvider,
    HookTrigger,
    MockVisionProvider,
    RoomKit,
    VideoChannel,
    VideoPipelineConfig,
)
from roomkit.models.session_event import SessionStartedEvent
from roomkit.video.backends.local import LocalVideoBackend
from roomkit.video.pipeline.filter.censor import CensorVideoFilter
from roomkit.video.vision.base import VisionProvider

logging.basicConfig(level=logging.INFO, format="%(name)s  %(message)s")


def _build_vision(args: argparse.Namespace) -> VisionProvider:
    if args.gemini:
        import os

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise SystemExit("Set GEMINI_API_KEY environment variable.")
        return GeminiVisionProvider(
            GeminiVisionConfig(
                api_key=api_key,
                model="gemini-3.1-flash-lite-preview",
                prompt=(
                    "List the objects you see. Include 'person' if any "
                    "human is visible. Respond with comma-separated labels only."
                ),
            )
        )
    return MockVisionProvider(
        descriptions=[
            "Empty room with a desk",
            "A person sitting at the desk",
            "A person waving at the camera",
            "Empty room, person has left",
            "Still empty",
            "A person walking into the room",
        ],
        labels=[
            ["desk", "room"],
            ["person", "desk"],
            ["person", "gesture"],
            ["desk", "room"],
            ["room"],
            ["person", "room"],
        ],
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Webcam Censor Demo")
    parser.add_argument("--gemini", action="store_true", help="Use Gemini vision")
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument("--fps", type=int, default=15, help="Capture FPS")
    parser.add_argument("--interval", type=int, default=3000, help="Vision interval ms")
    args = parser.parse_args()

    kit = RoomKit()

    backend = LocalVideoBackend(device=args.device, fps=args.fps, width=640, height=480)

    vision = _build_vision(args)
    censor = CensorVideoFilter(blocked_labels={"person"}, grace_frames=30)

    video = VideoChannel(
        "video-main",
        backend=backend,
        pipeline=VideoPipelineConfig(
            filters=[censor],
            vision=vision,
        ),
        vision_interval_ms=args.interval,
    )
    kit.register_channel(video)

    await kit.create_room(room_id="censor-demo")
    await kit.attach_channel("censor-demo", "video-main")

    frame_count = 0
    censored_count = 0

    @kit.hook(HookTrigger.ON_VIDEO_SESSION_STARTED)
    async def on_started(event: SessionStartedEvent, ctx: object) -> None:
        print(f"  Session started: {event.session.id[:8]}...")  # type: ignore[union-attr]

    @kit.on("video_vision_result")
    async def on_vision(event: object) -> None:
        nonlocal frame_count, censored_count
        data = event.data  # type: ignore[attr-defined]
        labels = data.get("labels", [])
        desc = data.get("description", "")
        status = "CENSORED" if censor._censoring else "LIVE"
        print(f"  [{status}] Vision: {desc[:80]} | labels={labels}")
        print(f"    Frames: {frame_count} total, {censored_count} censored")

    def on_frame(session: object, frame: object) -> None:
        nonlocal frame_count, censored_count
        frame_count += 1
        if censor._censoring:
            censored_count += 1

    # Tap on the channel (post-pipeline) to see filtered frames
    video.add_media_tap(on_frame)

    session = await kit.connect_video("censor-demo", "local-user", "video-main")

    print("Webcam Censor Demo")
    print("=" * 60)
    print(f"Mode: {'Gemini' if args.gemini else 'Mock'} vision")
    print("Blocked labels: {'person'}")
    print(f"Camera: device {args.device} at 640x480 @ {args.fps}fps")
    print(f"Vision every {args.interval}ms")
    print("When a person is detected, frames are replaced with black.")
    print("Press Ctrl+C to stop.\n")

    await backend.start_capture(session)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig_name in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig_name, stop.set)

    await stop.wait()

    print(f"\nDone. {frame_count} frames, {censored_count} censored.")
    await backend.stop_capture(session)
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
