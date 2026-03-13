"""RoomKit -- Webcam recording: capture camera to MP4 file.

Records webcam frames to an MP4 file using OpenCVVideoRecorder.
Pure recording — no AI, no vision, just camera → MP4.

Prerequisites:
    pip install roomkit[local-video]

Run with:
    uv run python examples/webcam_recording.py
    uv run python examples/webcam_recording.py --duration 30
    uv run python examples/webcam_recording.py --output ./my_recordings
    uv run python examples/webcam_recording.py --fps 30 --device 0

Press Ctrl+C to stop early.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal

from roomkit import (
    HookExecution,
    HookTrigger,
    RoomKit,
    VideoChannel,
    VideoFrame,
    VideoPipelineConfig,
)
from roomkit.models.session_event import SessionStartedEvent
from roomkit.video.backends.local import LocalVideoBackend
from roomkit.video.recorder import MockVideoRecorder, VideoRecordingConfig

logging.basicConfig(level=logging.WARNING)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Webcam Recording Demo")
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument("--fps", type=int, default=15, help="Capture FPS")
    parser.add_argument("--duration", type=int, default=0, help="Record N seconds (0=Ctrl+C)")
    parser.add_argument("--output", default="./recordings", help="Output directory")
    args = parser.parse_args()

    kit = RoomKit()

    # --- Video backend: local webcam -----------------------------------------
    backend = LocalVideoBackend(device=args.device, fps=args.fps, width=640, height=480)

    # --- Recorder: OpenCV if available, mock fallback ------------------------
    try:
        from roomkit.video.recorder.opencv import OpenCVVideoRecorder

        recorder = OpenCVVideoRecorder()
        recorder_name = "OpenCV → MP4"
    except ImportError:
        recorder = MockVideoRecorder()
        recorder_name = "Mock (no opencv)"

    recording_config = VideoRecordingConfig(
        storage=args.output,
        format="mp4",
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

    print("Webcam Recording Demo")
    print("=" * 60)
    print(f"Recorder: {recorder_name}")
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

    print(f"  Recorded {frame_count} frames")
    print("  Done.")


if __name__ == "__main__":
    asyncio.run(main())
