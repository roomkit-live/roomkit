# Video

Real-time video channels with pluggable processing pipelines, vision AI, bridging, recording, and avatar output.

## Video Channels

| Channel | ChannelType | Use Case |
|---------|-------------|----------|
| `VideoChannel` | `VIDEO` | Video-only transport (webcam, screen, WebSocket, WebRTC) |
| `AudioVideoChannel` | `AUDIO_VIDEO` | Combined A/V backends (SIP, RTP, FastRTC); extends `VoiceChannel` |
| `RealtimeAudioVideoChannel` | `REALTIME_AUDIO_VIDEO` | Speech-to-speech AI with video (e.g. Anam avatar); extends `RealtimeVoiceChannel` |

```python
from roomkit import RoomKit, VideoChannel
from roomkit.video import VideoPipelineConfig
from roomkit.video.backends.local import LocalVideoBackend
from roomkit.video.vision.mock import MockVisionProvider

kit = RoomKit()
video = VideoChannel(
    "video-main",
    backend=LocalVideoBackend(device=0, fps=30, width=640, height=480),
    pipeline=VideoPipelineConfig(vision=MockVisionProvider()),
    vision_interval_ms=2000,    # vision analysis throttle
    bridge=True,                # optional: or VideoBridgeConfig
)
kit.register_channel(video)

@kit.on("video_vision_result")
async def on_vision(event):
    print(event.data["description"], event.data.get("labels"))

session = await kit.join("webcam-demo", "video-main", participant_id="local-user")
await video.backend.start_capture(session)   # LocalVideoBackend-specific
# ... later
await kit.leave(session)
```

`AudioVideoChannel` adds `video_pipeline=`, `vision=`, `vision_interval_ms=`, `avatar=`, `avatar_encoder=`, `video_bridge=` to the `VoiceChannel` kwargs; its `backend` must implement both `VoiceBackend` and `VideoBackend`. `RealtimeAudioVideoChannel` adds `video_pipeline=`, `vision=`, `vision_interval_ms=`; its provider must implement `RealtimeAudioVideoProvider.on_video()`.

## VideoFrame

Inbound frames are `VideoFrame` dataclasses: `data: bytes`, `codec="h264"`, `width=640`, `height=480`, `timestamp_ms`, `keyframe=False`, `sequence`, `metadata`. Encoded codecs: `h264`, `vp8`, `vp9`, `av1`; raw: `raw_rgb24`, `raw_bgr24`, `raw_yuv420p`, `raw_nv12` (`is_encoded`/`is_raw` properties). Outbound streaming uses `VideoChunk` (encoded only).

## Video Pipeline

Inbound order: `[Decoder] -> [Resizer] -> [Transforms...] -> [Filters...] -> taps/bridge/vision`. All stages optional; decoder, resizer, and vision are singular, transforms and filters are chained lists. Vision runs async on its own throttled schedule; the rest is synchronous per-frame.

```python
from roomkit.video.pipeline.decoder.pyav import PyAVVideoDecoder
from roomkit.video.pipeline.resizer.pyav import PyAVVideoResizer

pipeline = VideoPipelineConfig(
    decoder=PyAVVideoDecoder(output_format="rgb24"),
    resizer=PyAVVideoResizer(width=640, height=480, keep_aspect=True),
    transforms=[], filters=[], vision=vision,
    recorder=recorder,
    recording_config=VideoRecordingConfig(storage="./recordings", format="mp4",
                                          codec="auto", fps=15.0),
)
```

| Stage | Role | Implementations |
|-------|------|-----------------|
| Decoder (`VideoDecoderProvider`) | Encoded → raw pixels; returns `None` until keyframe | `PyAVVideoDecoder`, mock |
| Resizer (`VideoResizerProvider`) | Downscale raw frames to fit target box | `PyAVVideoResizer`, mock |
| Transform (`VideoTransformProvider`) | Per-frame pixel modification (fast, sync) | `VideoEffectTransform(effect=...)`: grayscale, sepia, invert, blur, cartoon, edges, sketch, pixelate; mock |
| Filter (`VideoFilterProvider`) | Inspect/replace frames via `FilterContext` (latest vision result); emit `FilterEvent`s | `YOLODetectorFilter`, `WatermarkFilter`, `CensorVideoFilter`, `FaceTouchFilter` (MediaPipe), `OverlayFilter`, mocks |
| Vision (`VisionProvider`) | Async frame analysis → `VisionResult` | `OpenAIVisionProvider`, `GeminiVisionProvider`, mock |
| Recorder (`VideoRecorder`) | Tap every frame to file | `PyAVVideoRecorder`, `OpenCVVideoRecorder`, mock |
| Encoder (`VideoEncoderProvider`) | Outbound: raw → H.264 NALs (avatar path) | `PyAVVideoEncoder(width=512, height=512, fps=30, codec="libx264")` |

Key filters: `YOLODetectorFilter(model="yolo26n.pt", confidence=0.5, classes=None, every_n_frames=1, draw_boxes=False)`; `CensorVideoFilter(blocked_labels, replacement="black", grace_frames=0)` blanks frames while vision reports a blocked label; `FaceTouchFilter` emits `face_touch` detections. Overlays (`roomkit.video.pipeline.overlay`): `TextOverlayRenderer`, `ImageOverlayRenderer`, `SubtitleManager`/`subtitle_overlay` for live subtitles.

## Video Backends

`VideoBackend` ABC: `connect()`, `accept()`, `disconnect()`, `send_video()`, `send_video_sync()`, `request_keyframe()`, `set_video_passthrough()`; callbacks `on_video_received`, `on_session_ready`, `on_client_disconnected`; `capabilities` flags: `SIMULCAST`, `SVC`, `SCREEN_SHARE`, `RECORDING`, `BANDWIDTH_ESTIMATION`.

| Backend | Class | Extra |
|---------|-------|-------|
| WebSocket | `WebSocketVideoBackend` | fastapi only |
| WebRTC | `FastRTCVideoBackend` (combined A/V) | `roomkit[fastrtc]` |
| RTP | `RTPVideoBackend` (combined A/V) | `roomkit[rtp]` |
| SIP | `SIPVideoBackend` (combined A/V) | `roomkit[sip]` |
| Webcam | `LocalVideoBackend(device=0, fps=30, width=640, height=480)` | `roomkit[local-video]` |
| Screen | `ScreenCaptureBackend(monitor=1, region=None, fps=5, scale=1.0, diff_threshold=0.0)` | `roomkit[screen-capture]` |
| Mock | `MockVideoBackend` | built-in |

`WebSocketVideoBackend` mounts on FastAPI via `mount_websocket_video(app, backend, path=...)` and auto-creates sessions via `set_session_factory()`. Each backend has a `get_*` lazy loader in `roomkit.video` (e.g. `get_sip_video_backend()`).

## Vision Providers

```python
from roomkit.video import OpenAIVisionConfig, OpenAIVisionProvider

# Defaults target Ollama: base_url="http://localhost:11434/v1", model="qwen3.5"
vision = OpenAIVisionProvider(OpenAIVisionConfig(
    api_key="sk-...", base_url="https://api.openai.com/v1",
    model="gpt-4o", detail="low", max_tokens=100,
))
# GeminiVisionProvider(GeminiVisionConfig(api_key="AIza...", model="gemini-3.1-flash-lite"))
```

`await provider.analyze_frame(frame, prompt=None)` returns `VisionResult(description, labels, confidence, faces, text, metadata)` — `text` is OCR, `faces` are `FaceDetection` boxes. Results are cached per session (`channel.get_last_vision_result(session_id)`), emitted as `video_vision_result` framework events, and auto-injected into the system prompt of any `AIChannel` in the same room. For realtime voice, `setup_realtime_vision(kit, room_id, voice_channel_id)` injects via `inject_text(silent=True)`. On-demand tools: `DescribeWebcamTool`, `DescribeScreenTool`, `ListWebcamsTool`.

## Video Hook Triggers

| Trigger | Fires When | Execution |
|---------|-----------|-----------|
| `ON_VIDEO_SESSION_STARTED` | Backend signals video path live and session bound | async |
| `ON_VIDEO_SESSION_ENDED` | Session unbound / client disconnected | async |
| `ON_VISION_RESULT` | Vision analysis completed (`VisionEvent`) — can block or modify the description | sync |
| `ON_VIDEO_DETECTION` | Pipeline filter emitted a detection (`VideoDetectionEvent`: `kind`, `labels`, `confidence`, `metadata`) | async |
| `BEFORE_BRIDGE_VIDEO` | Frame about to be bridge-forwarded (`BridgeVideoEvent`); `HookResult.block()` drops it | sync |
| `ON_SCREEN_SHARE_STARTED` / `STOPPED` | ConferenceChannel: `SCREEN_SHARE`-kind track published/unpublished | async |
| `ON_VIDEO_TRACK_ADDED` / `REMOVED` | Reserved — defined in `HookTrigger`, not yet fired by built-in channels | — |

## Video Bridge

`bridge=True` (or `VideoBridgeConfig(enabled=True, max_participants=10, forwarding_strategy="forward", keyframe_interval_s=5.0)`) forwards frames between sessions in the same room for human-to-human video, requesting keyframes (PLI) from sources periodically and when receivers join. `channel.set_bridge_filter(fn)` installs a synchronous per-frame filter `(source_session, frame) -> frame | None` — the fast-path alternative to `BEFORE_BRIDGE_VIDEO`.

## Avatars

`AvatarProvider` (`roomkit.video.avatar`) generates lip-synced video from TTS audio on `AudioVideoChannel`: `MuseTalkAvatarProvider` (local GPU), `WebSocketAvatarProvider` (remote inference), `MockAvatarProvider`. Pair with `avatar_encoder=PyAVVideoEncoder(...)` for RTP/SIP output. Cloud avatar A/V (Anam) uses `RealtimeAudioVideoChannel` instead.

## Optional Extras

`roomkit[video]` (PyAV stages: av, numpy) · `[local-video]` (webcam, effects, OpenCV recorder) · `[screen-capture]` (mss) · `[screen-input]` (pyautogui) · `[yolo]` (ultralytics) · `[mediapipe]` · `[video-overlay]` (Pillow) · `[fastrtc]`/`[rtp]`/`[sip]`/`[anam]` (transports / cloud avatar).
