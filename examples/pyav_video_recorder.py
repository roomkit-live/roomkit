"""RoomKit -- PyAV video recorder: H.264 webcam capture to compressed MP4.

Records webcam frames to a properly compressed H.264 MP4 file using
PyAVVideoRecorder (FFmpeg via PyAV). Produces 10-50x smaller files
than the OpenCV recorder. Supports NVIDIA hardware encoding (NVENC).

Prerequisites:
    pip install roomkit[local-video,video]

Run with:
    uv run python examples/pyav_video_recorder.py
    uv run python examples/pyav_video_recorder.py --duration 30
    uv run python examples/pyav_video_recorder.py --codec h264_nvenc
    uv run python examples/pyav_video_recorder.py --output ./my_recordings --fps 30

Press Ctrl+C to stop early.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import asyncio
import contextlib
import logging
import os
import signal

from shared import setup_logging

from roomkit import HookExecution, HookTrigger, RoomKit, VideoChannel
from roomkit.models.session_event import SessionStartedEvent
from roomkit.video import VideoFrame
from roomkit.video.backends.local import LocalVideoBackend
from roomkit.video.pipeline import VideoPipelineConfig
from roomkit.video.recorder import VideoRecordingConfig
from roomkit.video.recorder.pyav import PyAVVideoRecorder

setup_logging("pyav_video_recorder", level=logging.WARNING)


async def main() -> None:
    parser = argparse.ArgumentParser(description="PyAV Video Recorder Demo")
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument("--fps", type=int, default=15, help="Capture FPS")
    parser.add_argument("--duration", type=int, default=0, help="Record N seconds (0=Ctrl+C)")
    parser.add_argument("--output", default="./recordings", help="Output directory")
    parser.add_argument(
        "--codec",
        default="auto",
        choices=["auto", "libx264", "h264_nvenc", "libx265"],
        help="Video codec (auto = NVENC if available, else libx264)",
    )
    args = parser.parse_args()

    kit = RoomKit()

    # --- Video backend: local webcam -----------------------------------------
    backend = LocalVideoBackend(device=args.device, fps=args.fps, width=640, height=480)

    # --- Recorder: PyAV (FFmpeg H.264) ---------------------------------------
    recorder = PyAVVideoRecorder()
    recording_config = VideoRecordingConfig(
        storage=args.output,
        format="mp4",
        codec=args.codec,
        fps=float(args.fps),
    )

    # --- Video channel with pipeline recorder --------------------------------
    video = VideoChannel(
        "video-rec",
        backend=backend,
        pipeline=VideoPipelineConfig(
            recorder=recorder,
            recording_config=recording_config,
        ),
    )
    kit.register_channel(video)

    # --- Room setup ----------------------------------------------------------
    await kit.create_room(room_id="recording-demo")
    await kit.attach_channel("recording-demo", "video-rec")

    # --- Hooks: log events ---------------------------------------------------
    frame_count = 0

    @kit.hook(HookTrigger.ON_VIDEO_SESSION_STARTED, execution=HookExecution.ASYNC)
    async def on_started(event: SessionStartedEvent, ctx: object) -> None:
        print(f"  Session started: {event.session.id[:8]}...")  # type: ignore[union-attr]

    def on_frame(session: object, frame: VideoFrame) -> None:
        nonlocal frame_count
        frame_count += 1
        if frame_count % args.fps == 0:
            secs = frame_count // args.fps
            print(f"\r  Recording... {secs}s ({frame_count} frames)", end="", flush=True)

    backend.on_video_received(on_frame)

    # --- Connect and start ---------------------------------------------------
    session = await kit.connect_video("recording-demo", "local-user", "video-rec")

    print("PyAV Video Recorder Demo")
    print("=" * 60)
    print(f"Codec: {args.codec}")
    print(f"Camera: device {args.device} at 640x480 @ {args.fps}fps")
    print(f"Output: {args.output}/")
    if args.duration:
        print(f"Duration: {args.duration}s")
    else:
        print("Duration: until Ctrl+C")
    print()

    await backend.start_capture(session)

    # --- Wait for duration or Ctrl+C ----------------------------------------
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig_name in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig_name, stop.set)

    if args.duration:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=args.duration)
    else:
        await stop.wait()

    # --- Cleanup -------------------------------------------------------------
    print("\n\n  Stopping...")
    await backend.stop_capture(session)
    await kit.disconnect_video(session)
    await kit.close()

    # Show output info
    output_dir = args.output
    if os.path.isdir(output_dir):
        files = sorted(f for f in os.listdir(output_dir) if f.endswith(".mp4"))
        if files:
            latest = os.path.join(output_dir, files[-1])
            size_kb = os.path.getsize(latest) / 1024
            print(f"  Recorded {frame_count} frames → {latest} ({size_kb:.1f} KB)")
    print("  Done.")


if __name__ == "__main__":
    asyncio.run(main())
