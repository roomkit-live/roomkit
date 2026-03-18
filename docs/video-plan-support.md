# Video Support Plan for RoomKit

Last updated: 2026-03-18

## Current State

Real-time video is fully operational across multiple transport backends, with AI vision integration, a multi-stage processing pipeline, and room-level A/V recording. File-based video sharing (WhatsApp, Telegram) predates this work and continues unchanged.

---

## Scope Definition

| Tier | Description | Status |
|------|-------------|--------|
| **Tier 0** | File-based video messages (send/receive video clips) | **✅ Done** |
| **Tier 1** | Real-time video transport | **✅ Done** (RTP, SIP, Local, Screen, FastRTC, WebSocket) |
| **Tier 2** | Video processing pipeline | **✅ Done** (engine, decoder, encoder, resizer, filters, transforms, recording) |
| **Tier 3** | Video intelligence (vision AI) | **✅ Done** (OpenAI, Gemini, Mock + agent tools) |
| **Tier 4** | Combined A/V channel | **✅ Done** (AudioVideoChannel + RealtimeAudioVideoChannel) |
| **Tier 5** | Multi-party video (SFU) | **🔲 Not started** |

---

## Architecture Overview

```
VideoBackend (capture/transport)
    │
    ▼
VideoChannel (session lifecycle, hooks)
    │
    ▼
VideoPipeline (optional processing)
    │
    ├─ [Decoder] → [Resizer] → [Transforms] → [Filters]
    │
    ├─ VisionProvider (frame → text description)
    │       │
    │       ▼
    │   ON_VISION_RESULT hook / inject into AI context
    │
    └─ Recorder (frames → MP4)
```

---

## Phase 1 — Video Transport (Tier 1) ✅ Complete

### 1.1 Core Models ✅

- **`VideoFrame`** (`video/video_frame.py`) — inbound frame container with codec, dimensions, keyframe flag, sequence number. Supports encoded codecs (h264, vp8, vp9, av1) and raw codecs (raw_rgb24, raw_bgr24, raw_yuv420p, raw_nv12).
- **`VideoChunk`** (`video/base.py`) — outbound frame container.
- **`VideoSession`** (`video/base.py`) — session state with video track fields.
- **`VideoCapability`** flags — SIMULCAST, SVC, SCREEN_SHARE, RECORDING, BANDWIDTH_ESTIMATION.

### 1.2 VideoBackend ABC ✅

Located at `video/backends/base.py`. Methods: `connect()`, `disconnect()`, `send_video()`, `get_session()`, `get_video_session()`, `list_sessions()`, `close()`. Callbacks: `on_video_received()`, `on_session_ready()`, `on_client_disconnected()`.

### 1.3 VideoChannel ✅

Located at `channels/video.py`. Manages VideoBackend integration, session lifecycle, vision sampling at configurable intervals, recording handles per session, and media recording taps for room-level muxing.

### 1.4 Video Backends ✅

| Backend | Transport | Codecs | Status |
|---------|-----------|--------|--------|
| `LocalVideoBackend` | Webcam (OpenCV) | raw_rgb24 | ✅ Done |
| `ScreenCaptureBackend` | Screen capture (mss) | raw_rgb24 | ✅ Done |
| `RTPVideoBackend` | RTP/UDP (aiortp) | H.264, VP9, VP8 | ✅ Done |
| `SIPVideoBackend` | SIP+RTP (aiosipua) | H.264, VP9, VP8 | ✅ Done |
| `FastRTCVideoBackend` | WebRTC via FastRTC | H.264, VP8 | ✅ Done |
| `WebSocketVideoBackend` | Binary WebSocket protocol | H.264, VP8, raw | ✅ Done |
| `MockVideoBackend` | Simulated frames | — | ✅ Done |

**RTP/SIP highlights:**
- VP9 RTP depacketization (RFC 9628) with B/E-bit frame boundaries
- H.264 FU-A depacketization (RFC 6184)
- Jitter buffer with marker-bit frame delivery (instant, no next-timestamp wait)
- PLI-based keyframe recovery on packet loss
- NACK for packet retransmission
- PortAllocator for multi-session port management
- SDP codec negotiation (offerer preference order)

**Screen capture highlights:**
- Multi-monitor selection, region cropping
- Resolution downscaling (saves vision API tokens)
- Diff-based frame skipping for static screens

### 1.5 Hook Triggers ✅

```
ON_VIDEO_SESSION_STARTED, ON_VIDEO_SESSION_ENDED
ON_VIDEO_TRACK_ADDED, ON_VIDEO_TRACK_REMOVED
ON_SCREEN_SHARE_STARTED, ON_SCREEN_SHARE_STOPPED
ON_VISION_RESULT
```

---

## Phase 2 — Video Processing Pipeline (Tier 2) ✅ Complete

### 2.1 VideoPipeline Engine ✅

Located at `video/pipeline/engine.py`. Orchestrates inbound processing through decode → resize → transforms → filters chain. Includes async vision offload with result caching and `FilterContext` for passing vision state to filters.

### 2.2 Pipeline Stages ✅

| Stage | ABC | Implementations | Status |
|-------|-----|----------------|--------|
| **Decoder** | `VideoDecoderProvider` | `PyAVVideoDecoder` (H.264, VP8, VP9, AV1 → raw), `MockVideoDecoderProvider` | ✅ Done |
| **Encoder** | `VideoEncoderProvider` | `PyAVVideoEncoder` (raw → H.264 NAL units, libx264) | ✅ Done |
| **Resizer** | `VideoResizerProvider` | `PyAVVideoResizer` (downscale with aspect ratio), `MockVideoResizerProvider` | ✅ Done |
| **Filters** | `VideoFilterProvider` | `WatermarkFilter`, `YOLODetectorFilter`, `CensorVideoFilter`, `MockVideoFilterProvider` | ✅ Done |
| **Transforms** | `VideoTransformProvider` | `VideoEffectTransform` (8 effects: grayscale, sepia, invert, blur, cartoon, edges, sketch, pixelate), `MockVideoTransformProvider` | ✅ Done |
| **Recorder** | `VideoRecorder` | `PyAVVideoRecorder` (H.264/H.265/NVENC), `OpenCVVideoRecorder`, `MockVideoRecorder` | ✅ Done |

### 2.3 VideoPipelineConfig ✅

Located at `video/pipeline/config.py`. All stages optional — only configured stages run:

```python
@dataclass
class VideoPipelineConfig:
    decoder: VideoDecoderProvider | None = None
    resizer: VideoResizerProvider | None = None
    transforms: list[VideoTransformProvider] = field(default_factory=list)
    filters: list[VideoFilterProvider] = field(default_factory=list)
    vision: VisionProvider | None = None
    recorder: VideoRecorder | None = None
    recording_config: VideoRecordingConfig | None = None
```

### 2.4 VideoBridge (SFU) 🔲 Not Started

No `video/bridge.py` exists. Multi-party video forwarding is not implemented. See [Remaining Work](#remaining-work).

---

## Phase 2.5 — Room-Level Media Recording ✅ Complete

Mux audio and video from multiple channels into a single MP4 per room.

- **`MediaRecorder`** ABC (`recorder/base.py`) — event-driven multi-track recorder
- **`PyAVMediaRecorder`** (`recorder/pyav.py`) — production implementation with H.264+AAC muxing
- **`MockMediaRecorder`** (`recorder/mock.py`) — testing
- **`RoomRecorderManager`** — orchestrates per-room recording lifecycle

**Key design:**
- Declarative — channels declare what they contribute via `ChannelRecordingConfig`
- Room-level concern — recording configured on room, not individual channels
- Multi-recorder — local file + cloud upload, full quality + preview
- Per-participant audio tracks

**Implementation notes:**
- A/V sync via per-track PTS (each stream starts at PTS=0 from its own first frame)
- VP9 transcode: VP9 from RTP → decode → re-encode as H.264 for MP4
- libx264 zerolatency: immediate output from first frame
- Pre-roll buffering: all tracks must receive first frame before container creation
- Thread-safe: per-recording `threading.Lock` for concurrent audio/video writes

---

## Phase 3 — Video Intelligence (Tier 3) ✅ Complete

### 3.1 VisionProvider ABC ✅

Located at `video/vision/base.py`. Returns `VisionResult` with description, labels, confidence, faces, OCR text.

### 3.2 Implementations ✅

| Provider | Model | Status |
|----------|-------|--------|
| `GeminiVisionProvider` | Gemini Flash | ✅ Done |
| `OpenAIVisionProvider` | GPT-4o / Ollama / vLLM | ✅ Done |
| `MockVisionProvider` | — | ✅ Done |

### 3.3 AI Integration ✅

- `setup_video_vision()` — inject vision results into AIChannel context
- `setup_realtime_vision()` — inject into RealtimeVoiceChannel sessions
- Interval-based throttle in VideoChannel (configurable `vision_interval_ms`)

### 3.4 Agent Vision Tools ✅

| Tool | Purpose |
|------|---------|
| `DescribeScreenTool` | Capture screenshot → vision AI description |
| `DescribeWebcamTool` | Capture webcam frame → vision AI description |
| `ListWebcamsTool` | Enumerate available cameras |
| `ScreenInputTools` | Vision-assisted click, type, scroll, press keys |

---

## Phase 4 — Combined A/V Channel ✅ Complete

### 4.1 AudioVideoChannel ✅

Located at `channels/av.py`. Extends `VoiceChannel` with `VideoBackend` integration for SIP A/V calls.

### 4.2 RealtimeAudioVideoChannel ✅

Located at `channels/realtime_av.py`. Room-based channel for speech-to-speech + avatar providers (Anam AI).

### 4.3 RealtimeAVBridge ✅

Located at `voice/realtime/bridge.py`. Wires any VoiceBackend to any RealtimeAudioVideoProvider. Handles audio resampling (48kHz → codec rate), H.264 encoding, video pipeline (filters, watermark), and session lifecycle.

### 4.4 Avatar Providers ✅

| Provider | Type | Status |
|----------|------|--------|
| `AnamRealtimeProvider` | Cloud (WebRTC) | ✅ Done |
| `PersonaPlexRealtimeProvider` | Cloud (NVIDIA) | ✅ Done |
| `MuseTalkAvatarProvider` | Local (GPU) | ✅ Done |
| `WebSocketAvatarProvider` | Custom WebSocket | ✅ Done |

### 4.5 Bandwidth Adaptation 🔲 Not Started

See [Remaining Work](#remaining-work).

---

## Remaining Work

Three areas remain unimplemented:

### 1. VideoBridge / SFU — Multi-Party Video Forwarding

**Impact:** Medium — needed for video conferencing (3+ participants).

**What's needed:**
- `VideoBridge` class analogous to `AudioBridge`
- SFU mode: forward encoded frames without decode (efficient)
- MCU mode (optional): decode, composite, re-encode (CPU/GPU heavy)
- Per-receiver quality selection (simulcast layer switching)
- Track management: add/remove participants, screen share as separate track

**Estimated effort:** 2-3 weeks

### 2. Bandwidth Adaptation

**Impact:** Low-medium — needed for production WebRTC deployments over unreliable networks.

**What's needed:**
- Monitor network quality via RTCP feedback
- Degrade video quality before audio quality
- Switch simulcast layers based on available bandwidth
- Pause video entirely under severe congestion (audio-only fallback)

**Estimated effort:** 1-2 weeks

### 3. ML Pipeline Stages (Background Blur, Face Detection)

**Impact:** Low — most ML processing should be client-side. Server-side only needed for headless/bot participants.

**What exists:**
- `FaceDetection` dataclass in `vision/base.py` (bounding box + confidence)
- Generic Gaussian blur in `VideoEffectTransform` (not background-aware)

**What's needed:**
- `BackgroundBlurProvider` ABC + MediaPipe implementation
- `FaceDetectorProvider` ABC + MediaPipe/MTCNN implementation
- Optional `ONNXVisionProvider` for local inference (YOLO, Florence-2)

**Estimated effort:** 2-3 weeks

**Recommendation:** Defer — these are GPU-heavy server-side workloads. Client-side alternatives (browser MediaPipe, native SDKs) are more practical for most deployments.

---

## Success Criteria

- [x] Video frames flowing through framework end-to-end
- [x] 7 video backends (Local, Screen, RTP, SIP, FastRTC, WebSocket, Mock)
- [x] Video processing pipeline with decoder, encoder, resizer, filters, transforms
- [x] Video recording to MP4 (PyAV H.264/H.265/NVENC + OpenCV)
- [x] Combined A/V recording to single MP4 with per-participant audio tracks
- [x] SIP A/V call with VP9 video → H.264 recording with synced audio
- [x] VP9 RTP depacketization with instant frame delivery
- [x] PLI-based keyframe recovery on packet loss
- [x] AudioVideoChannel + RealtimeAudioVideoChannel for combined sessions
- [x] RealtimeAVBridge for avatar providers (Anam, PersonaPlex, MuseTalk)
- [x] Vision AI describes video content to AIChannel and RealtimeVoiceChannel
- [x] Screen capture with multi-monitor, region cropping, diff-based skipping
- [x] Agent vision tools (describe screen/webcam, list cameras, screen input)
- [x] YOLO object detection filter, watermark filter, censor filter
- [x] 8 video effect transforms (grayscale, sepia, blur, cartoon, etc.)
- [x] All components have tests, docs, and examples
- [x] 14 runnable examples
- [ ] Multi-party video forwarding via SFU
- [ ] Screen sharing as separate track in multi-party
- [ ] Graceful video → audio-only fallback under poor network
- [ ] Background blur (server-side, MediaPipe)
- [ ] Face detection provider (server-side, MediaPipe)

---

## File Structure

```
src/roomkit/
  channels/
    video.py                      # VideoChannel
    av.py                         # AudioVideoChannel (SIP A/V)
    realtime_av.py                # RealtimeAudioVideoChannel (avatar)
    _video_hooks.py               # VideoHooksMixin
  video/
    __init__.py                   # Exports + lazy loaders
    base.py                       # VideoChunk, VideoCapability, VideoSession
    video_frame.py                # VideoFrame dataclass
    ai_integration.py             # setup_video_vision(), setup_realtime_vision()
    utils.py                      # make_text_frame()
    backends/
      base.py                     # VideoBackend ABC
      local.py                    # LocalVideoBackend (webcam, OpenCV)
      screen.py                   # ScreenCaptureBackend (mss)
      rtp.py                      # RTPVideoBackend (aiortp, H.264/VP9)
      sip.py                      # SIPVideoBackend (aiosipua, SIP+RTP A/V)
      fastrtc.py                  # FastRTCVideoBackend (WebRTC)
      websocket.py                # WebSocketVideoBackend (binary protocol)
      mock.py                     # MockVideoBackend
    pipeline/
      config.py                   # VideoPipelineConfig
      engine.py                   # VideoPipeline orchestrator
      decoder/
        base.py                   # VideoDecoderProvider ABC
        pyav.py                   # PyAVVideoDecoder (H.264/VP8/VP9/AV1)
        mock.py
      encoder/
        base.py                   # VideoEncoderProvider ABC
        pyav.py                   # PyAVVideoEncoder (H.264 via libx264)
      resizer/
        base.py                   # VideoResizerProvider ABC
        pyav.py                   # PyAVVideoResizer
        mock.py
      filter/
        base.py                   # VideoFilterProvider ABC
        watermark.py              # WatermarkFilter (text overlay)
        yolo.py                   # YOLODetectorFilter (object detection)
        censor.py                 # CensorVideoFilter (frame replacement)
        mock.py
      transform/
        base.py                   # VideoTransformProvider ABC
        effects.py                # 8 effects (grayscale, sepia, blur, etc.)
        mock.py
    recorder/
      base.py                     # VideoRecorder ABC
      pyav.py                     # PyAVVideoRecorder (H.264/H.265/NVENC)
      opencv.py                   # OpenCVVideoRecorder
      mock.py
    vision/
      base.py                     # VisionProvider ABC, VisionResult, FaceDetection
      gemini.py                   # GeminiVisionProvider
      openai.py                   # OpenAIVisionProvider
      screen_tool.py              # DescribeScreenTool, capture_screen_frame()
      webcam_tool.py              # DescribeWebcamTool, ListWebcamsTool
      screen_input.py             # ScreenInputTools (click, type, scroll, press)
      mock.py                     # MockVisionProvider
    avatar/
      base.py                     # AvatarProvider ABC
      musetalk.py                 # MuseTalkAvatarProvider
      websocket.py                # WebSocketAvatarProvider
      mock.py
  recorder/                       # Room-level media recording
    base.py                       # MediaRecorder ABC, MediaRecordingConfig
    pyav.py                       # PyAVMediaRecorder (A/V muxing)
    mock.py                       # MockMediaRecorder
    _room_recorder_manager.py     # Per-room orchestration
  providers/
    anam/                         # Anam AI avatar provider
    personaplex/                  # PersonaPlex (NVIDIA) avatar provider
```

---

## Dependencies

| Package | Purpose | Install Extra | Status |
|---------|---------|---------------|--------|
| `aiortp>=0.3.0` | RTP video transport | `roomkit[rtp]` | ✅ Installed |
| `aiosipua>=0.4.0` | SIP signaling + RTP | `roomkit[sip]` | ✅ Installed |
| `av>=12.0.0` (PyAV) | FFmpeg decode/encode/record | `roomkit[video]` | ✅ Installed |
| `opencv-python-headless` | Webcam capture, image ops | `roomkit[local-video]` | ✅ Installed |
| `mss` | Screen capture | `roomkit[screen-capture]` | ✅ Installed |
| `pyautogui` | Screen input tools | `roomkit[screen-input]` | ✅ Installed |
| `ultralytics` | YOLO object detection | `roomkit[yolo]` | ✅ Installed |
| `mediapipe` | Background blur, face detection | — | 🔲 Not installed (future) |
| `aiortc` | Native WebRTC | — | 🔲 Not installed (FastRTC covers WebRTC use case) |
