"""RoomKit -- Room-level media recording: mic + webcam → single MP4.

Records audio from the microphone and video from the webcam into a
single muxed MP4 file using the room-level MediaRecorder.

This demonstrates the production recording path — audio and video from
different channels are combined into one output file per room.

Prerequisites:
    pip install roomkit[local-audio,local-video,video]

Run with:
    uv run python examples/room_media_recorder.py
    uv run python examples/room_media_recorder.py --duration 10
    uv run python examples/room_media_recorder.py --output ./my_recordings
    uv run python examples/room_media_recorder.py --fps 15 --device 0

Press Ctrl+C to stop early.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal

from roomkit import (
    ChannelRecordingConfig,
    MediaRecordingConfig,
    RoomKit,
    VideoChannel,
    VideoFrame,
    VoiceChannel,
)
from roomkit.recorder import MockMediaRecorder, RoomRecorderBinding
from roomkit.video.backends.local import LocalVideoBackend
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.pipeline.config import AudioPipelineConfig

logging.basicConfig(level=logging.WARNING, format="%(name)s  %(message)s")
logging.getLogger("roomkit.recorder").setLevel(logging.INFO)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Room Media Recording Demo")
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument("--fps", type=int, default=15, help="Video capture FPS")
    parser.add_argument("--duration", type=int, default=0, help="Record N seconds (0=Ctrl+C)")
    parser.add_argument("--output", default="./recordings", help="Output directory")
    args = parser.parse_args()

    # --- Recorder: PyAV if available, mock fallback ----------------------
    try:
        from roomkit.recorder.pyav import PyAVMediaRecorder

        recorder = PyAVMediaRecorder()
        recorder_name = "PyAV → MP4"
    except ImportError:
        recorder = MockMediaRecorder()
        recorder_name = "Mock (install roomkit[video] for real MP4)"

    recording_config = MediaRecordingConfig(
        storage=args.output,
        video_codec="auto",
        audio_codec="aac",
        audio_sample_rate=16000,
        format="mp4",
    )

    # --- Backends: local mic + webcam ------------------------------------
    audio_backend = LocalAudioBackend(input_sample_rate=16000)
    video_backend = LocalVideoBackend(device=args.device, fps=args.fps, width=640, height=480)

    # --- Channels with recording configs ---------------------------------
    voice_ch = VoiceChannel(
        "voice-rec",
        backend=audio_backend,
        pipeline=AudioPipelineConfig(),
        recording=ChannelRecordingConfig(audio=True),
    )
    video_ch = VideoChannel(
        "video-rec",
        backend=video_backend,
        recording=ChannelRecordingConfig(video=True),
    )

    # --- Build the room --------------------------------------------------
    kit = RoomKit(voice=audio_backend)
    kit.register_channel(voice_ch)
    kit.register_channel(video_ch)

    room = await kit.create_room(
        room_id="recording-demo",
        recorders=[
            RoomRecorderBinding(
                recorder=recorder,
                config=recording_config,
                name="main",
            ),
        ],
    )
    await kit.attach_channel(room.id, "voice-rec")
    await kit.attach_channel(room.id, "video-rec")

    # --- Connect participant ---------------------------------------------
    voice_session = await kit.connect_voice(room.id, "local-user", "voice-rec")
    video_session = await kit.connect_video(room.id, "local-user", "video-rec")

    # --- Progress counter ------------------------------------------------
    frame_count = 0

    def on_frame(_session: object, _frame: VideoFrame) -> None:
        nonlocal frame_count
        frame_count += 1
        if frame_count % args.fps == 0:
            secs = frame_count // args.fps
            print(
                f"\r  Recording... {secs}s ({frame_count} video frames)",
                end="",
                flush=True,
            )

    video_backend.on_video_received(on_frame)

    # --- Start capture ---------------------------------------------------
    print("Room Media Recording Demo")
    print("=" * 60)
    print(f"Recorder : {recorder_name}")
    print(f"Camera   : device {args.device} at 640x480 @ {args.fps}fps")
    print("Mic      : 16kHz mono")
    print(f"Output   : {args.output}/")
    if args.duration:
        print(f"Duration : {args.duration}s")
    else:
        print("Duration : until Ctrl+C")
    print()

    await audio_backend.start_listening(voice_session)
    await video_backend.start_capture(video_session)

    # --- Wait for duration or Ctrl+C ------------------------------------
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig_name in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig_name, stop.set)

    if args.duration:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=args.duration)
    else:
        await stop.wait()

    # --- Cleanup ---------------------------------------------------------
    print("\n\n  Stopping...")
    await video_backend.stop_capture(video_session)
    await audio_backend.stop_listening(voice_session)
    await kit.disconnect_video(video_session)
    await kit.disconnect_voice(voice_session)
    await kit.close_room(room.id)
    await kit.close()

    print(f"  Recorded {frame_count} video frames")
    print("  Done.")


if __name__ == "__main__":
    asyncio.run(main())
