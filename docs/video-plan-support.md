# Video Support Plan for RoomKit

## Current State

File-based video sharing is already implemented across the framework:

- **`VideoContent`** model — `url`, `mime_type`, `duration_seconds`, `thumbnail_url`, `size_bytes`
- **`ChannelMediaType.VIDEO`** enum value with `supports_video` capability flag
- **WhatsApp Personal** and **Telegram** providers send/receive video files natively
- **Content transcoder** falls back to `"[Video: {url}]"` for text-only channels
- **`ChannelCapabilities`** includes `supports_video`, `max_video_duration_seconds`, `supported_video_formats`

What is **not** supported is real-time video — video calls, live streaming, and server-side video processing pipelines.

---

## Scope Definition

| Tier | Description | Status |
|------|-------------|--------|
| **Tier 0** | File-based video messages (send/receive video clips) | **Done** |
| **Tier 1** | Real-time video transport (video calls via WebRTC/RTMP) | **In progress** |
| **Tier 2** | Video processing pipeline (transcoding, scaling, overlays, recording) | Not started |
| **Tier 3** | Video intelligence (vision AI, frame analysis, captioning) | **In progress** |

---

## Architecture Overview

The voice subsystem serves as the architectural template. Each voice component maps to a video analog:

| Voice Component | Video Analog | Purpose |
|----------------|-------------|---------|
| `AudioFrame` | `VideoFrame` | Inbound frame container (encoded NAL units or raw pixels) |
| `AudioChunk` | `VideoChunk` | Outbound frame container |
| `VoiceSession` | `MediaSession` (extend) | Session state with video track fields |
| `VoiceBackend` | `VideoBackend` | Transport ABC (WebRTC, RTMP, SRT) |
| `AudioPipeline` | `VideoPipeline` | Processing stage orchestrator |
| `VoiceChannel` | `VideoChannel` | Main channel orchestrator |
| VAD / AEC / AGC / Denoiser | Scale / Overlay / FaceBlur / BgBlur | Pipeline stages |
| STT (speech-to-text) | Vision AI (frame-to-text) | Intelligence provider |
| TTS (text-to-speech) | Image/video generation | AI-to-visual output |
| `AudioRecorder` | `VideoRecorder` | Recording stage |
| `AudioBridge` | `VideoBridge` | Multi-party video forwarding (SFU) |

### Data Flow

```
Inbound Video:
  Participant (camera)
    -> VideoBackend.on_video_received()       [encoded frame -> VideoFrame]
    -> VideoPipeline.process_inbound():
         [Decoder] -> [Resizer] -> [Recorder tap] -> [FaceDetector] -> [BgBlur]
    -> VisionProvider.analyze(frame)          [periodic key-frame analysis]
    -> RoomKit.process_inbound()              [creates RoomEvent with VideoContent or text description]

Outbound Video:
  VideoBackend.send_video(session, frame)     [forward/relay to participants]
    <- VideoPipeline.process_outbound():
         [Overlay] -> [Encoder] -> [Recorder tap] -> [Resizer]
```

### Multi-Party Topology

```
SFU (Selective Forwarding Unit):
  Participant A -> Server -> Participant B
                          -> Participant C
  Participant B -> Server -> Participant A
                          -> Participant C

  Each participant sends 1 stream, receives N-1 streams.
  Server forwards without decoding (lightweight).

MCU (Multipoint Control Unit):
  All participants -> Server [decode + mix + encode] -> All participants

  Server composes a single stream. Heavy on server, simple on clients.
```

**Recommendation:** SFU for scalability. MCU only as an optional fallback for constrained clients.

---

## Phased Implementation Plan

### Phase 1 — Core Models and Video Transport (Tier 1)

**Goal:** Real-time video frames flowing through the framework.

**Status:** 1.1 ✅ | 1.2 ✅ | 1.3 ✅ | 1.4 partial (LocalVideoBackend) | 1.5 ✅ | 1.6 ✅

#### 1.1 Core Models ✅

**`VideoFrame`** dataclass (`video/video_frame.py`):

```python
@dataclass
class VideoFrame:
    data: bytes                          # Encoded frame (H.264/VP8/VP9 NAL unit) or raw YUV
    codec: str = "h264"                  # h264, vp8, vp9, av1, raw
    width: int = 640
    height: int = 480
    timestamp_ms: float | None = None    # Relative to session start
    keyframe: bool = False               # Is this an IDR/keyframe?
    sequence: int = 0                    # Frame sequence number
    metadata: dict[str, Any] = field(default_factory=dict)
```

**`VideoChunk`** dataclass (`video/base.py`):

```python
@dataclass
class VideoChunk:
    data: bytes
    codec: str = "h264"
    width: int = 640
    height: int = 480
    timestamp_ms: int | None = None
    keyframe: bool = False
    is_final: bool = False
```

**`VideoCapability`** flags:

```python
class VideoCapability(IntFlag):
    SIMULCAST = auto()        # Multiple resolution streams
    SVC = auto()              # Scalable Video Coding layers
    SCREEN_SHARE = auto()     # Separate screen share track
    RECORDING = auto()        # Server-side recording
    BANDWIDTH_ESTIMATION = auto()
```

#### 1.2 VideoBackend ABC ✅

Located at `video/backends/base.py`, following the `VoiceBackend` pattern:

```python
class VideoBackend(ABC):
    @abstractmethod
    async def connect(self, room_id: str, participant_id: str, channel_id: str) -> MediaSession: ...

    @abstractmethod
    async def send_video(self, session: MediaSession, frame: VideoFrame | AsyncIterator[VideoChunk]) -> None: ...

    @abstractmethod
    async def disconnect(self, session: MediaSession) -> None: ...

    # Callbacks
    def on_video_received(self, callback: Callable[[MediaSession, VideoFrame], None]) -> None: ...
    def on_session_ready(self, callback: Callable[[MediaSession], None]) -> None: ...
    def on_client_disconnected(self, callback: Callable[[MediaSession], None]) -> None: ...

    @property
    def capabilities(self) -> VideoCapability: ...
```

#### 1.3 VideoChannel ✅

Located at `channels/video.py`, minimal orchestrator:

- Registers with `VideoBackend` callbacks
- Manages session lifecycle (`bind_session`, `unbind_session`)
- Fires hook triggers on video events
- Forwards frames to `VideoBridge` for multi-party

#### 1.4 WebRTC Video Backend 🔲

Extend `FastRTCVoiceBackend` or build on `aiortc`:

- Video track negotiation (codec, resolution, framerate)
- ICE/DTLS/SRTP for video
- Simulcast support (multiple quality layers)
- Bandwidth estimation and adaptation

#### 1.5 New Hook Triggers ✅

```python
# Video session lifecycle
ON_VIDEO_SESSION_STARTED = "on_video_session_started"
ON_VIDEO_SESSION_ENDED = "on_video_session_ended"

# Video events
ON_VIDEO_FRAME = "on_video_frame"              # Per-frame (use sparingly)
ON_VIDEO_TRACK_ADDED = "on_video_track_added"
ON_VIDEO_TRACK_REMOVED = "on_video_track_removed"
ON_SCREEN_SHARE_STARTED = "on_screen_share_started"
ON_SCREEN_SHARE_STOPPED = "on_screen_share_stopped"

# Quality adaptation
ON_VIDEO_QUALITY_CHANGED = "on_video_quality_changed"
```

#### 1.6 RFC Update ✅

Add video transport section to `roomkit-specs/roomkit-rfc.md`. Either:
- Expand Level 3 ("Voice") to "Level 3: Real-Time Media" covering audio + video
- Add Level 4: Video (OPTIONAL)

**Effort estimate: 2-4 weeks**

Key risks: WebRTC video negotiation is significantly more complex than audio. Codec licensing (H.264) and cross-browser compatibility require careful handling.

---

### Phase 2 — Video Processing Pipeline (Tier 2)

**Goal:** Pluggable server-side video processing stages.

**Status:** 2.1 🔲 | 2.2 partial (Recorder ✅) | 2.3 🔲 | 2.4 🔲

#### 2.1 VideoPipeline Engine

Located at `video/pipeline/engine.py`, analogous to `AudioPipeline`:

```python
class VideoPipeline:
    async def process_inbound(self, session: MediaSession, frame: VideoFrame) -> VideoFrame: ...
    async def process_outbound(self, session: MediaSession, frame: VideoFrame) -> VideoFrame: ...
```

#### 2.2 Pipeline Stages

Each stage follows the ABC + mock + implementation pattern under `video/pipeline/<stage>/`:

| Stage | ABC | Purpose | Implementations |
|-------|-----|---------|----------------|
| **Decoder** | `VideoDecoderProvider` | Encoded -> raw pixels | FFmpeg, OpenCV |
| **Encoder** | `VideoEncoderProvider` | Raw pixels -> encoded | FFmpeg, OpenCV, hardware (NVENC) |
| **Resizer** | `VideoResizerProvider` | Resolution scaling | OpenCV, Pillow |
| **Overlay** | `VideoOverlayProvider` | Watermarks, names, timestamps | OpenCV, Pillow |
| **BackgroundBlur** | `BackgroundBlurProvider` | Blur/replace background | MediaPipe, ONNX |
| **FaceDetector** | `FaceDetectorProvider` | Detect and optionally blur faces | MediaPipe, MTCNN |
| **Recorder** | `VideoRecorder` | Record to MP4/WebM file or S3 | **OpenCV ✅**, FFmpeg |
| **NoiseFilter** | `VideoNoiseFilter` | Reduce compression artifacts | OpenCV |

#### 2.3 Pipeline Contract

```python
@dataclass
class VideoPipelineContract:
    transport_inbound_codec: str       # From backend (e.g., "vp8")
    transport_outbound_codec: str      # To backend (e.g., "h264")
    internal_format: str               # Inside pipeline (e.g., "raw_rgb24")
    max_resolution: tuple[int, int]    # Pipeline cap
    target_fps: int                    # Frame rate target
```

#### 2.4 VideoBridge (SFU)

Located at `video/bridge.py`:

```python
class VideoBridge:
    def add_session(self, session: MediaSession, room_id: str, backend: VideoBackend) -> None: ...
    def forward(self, source_session: MediaSession, frame: VideoFrame) -> None: ...
    def set_quality_selector(self, fn: Callable) -> None: ...  # Per-receiver quality selection
```

SFU mode: forward encoded frames without decode (efficient).
MCU mode (optional): decode, composite, re-encode (CPU/GPU heavy).

**Effort estimate: 3-5 weeks**

Key risks: Server-side video processing requires significant CPU/GPU. Background blur and face detection are real-time ML workloads. Consider making most stages client-side by default.

---

### Phase 3 — Video Intelligence (Tier 3)

**Goal:** AI understanding of video content for conversation context.

**Status:** 3.1 ✅ | 3.2 ✅ (OpenAI + Gemini) | 3.3 ✅ | 3.4 ✅

#### 3.1 VisionProvider ABC ✅

Located at `video/vision/base.py`:

```python
class VisionProvider(ABC):
    @abstractmethod
    async def analyze_frame(self, frame: VideoFrame) -> VisionResult: ...

    @abstractmethod
    async def analyze_stream(
        self, frames: AsyncIterator[VideoFrame], interval_ms: int = 1000
    ) -> AsyncIterator[VisionResult]: ...

@dataclass
class VisionResult:
    description: str                    # Natural language description
    labels: list[str]                   # Detected objects/scenes
    confidence: float                   # Overall confidence
    faces: list[FaceDetection] | None   # Detected faces
    text: str | None                    # OCR text (for screen shares)
    metadata: dict[str, Any]
```

#### 3.2 Implementations ✅

| Provider | Model | Speed | Status |
|----------|-------|-------|--------|
| `GeminiVisionProvider` | Gemini 3.1 Flash-Lite | ~1-2s (cloud) | **Done** |
| `OpenAIVisionProvider` | GPT-4o / Ollama / vLLM | varies | **Done** |
| `MockVisionProvider` | — | instant | **Done** |
| Ollama + moondream | moondream 1.8B | <1s (local) | **Works via OpenAIVisionProvider** |
| Ollama + qwen3-vl | qwen3-vl:4b | ~6-10s (local) | **Works but slow** |
| `ONNXVisionProvider` | YOLO / Florence-2 | ~30ms (local) | Not started |

#### 3.3 Key-Frame Sampling Strategy ✅

Implemented as interval-based throttle in VideoChannel._on_video_received:

- **Periodic sampling** — 1 frame every N seconds (configurable via `vision_interval_ms`, default 2s)
- Throttle check runs before task creation (no O(fps) task allocation)
- Timing logged per frame (elapsed_ms in framework events)

Future: scene change detection, on-demand triggers.

#### 3.4 AIChannel Integration ✅

Implemented via `setup_video_vision()`:

```python
from roomkit import setup_video_vision

setup_video_vision(kit, room_id="r1", ai_channel_id="ai")
# AI's system prompt is now live-updated with vision descriptions
```

- Injects vision descriptions into AI binding metadata (system_prompt)
- Preserves base system prompt across updates (no stacking)
- Filters by room_id
- Configurable context_prefix

---

### Phase 4 — Combined Audio+Video Channel

**Goal:** Unified media session for voice + video calls.

#### 4.1 Approach Options

**Option A — Separate channels, shared session:**

```python
voice = VoiceChannel("voice", backend=audio_backend, stt=stt, tts=tts)
video = VideoChannel("video", backend=video_backend, vision=vision)
# Both bound to same room, share MediaSession
```

**Option B — Unified MediaChannel:**

```python
media = MediaChannel(
    "media",
    audio_backend=audio_backend,
    video_backend=video_backend,
    stt=stt, tts=tts, vision=vision,
    pipeline=MediaPipelineConfig(audio=audio_config, video=video_config),
)
```

**Recommendation:** Option A for flexibility. A `MediaSession` model extends `VoiceSession` with video track state, and both channels reference the same session. A convenience `MediaChannel` factory can compose them.

#### 4.2 Track Management

```python
@dataclass
class MediaSession(VoiceSession):
    video_enabled: bool = False
    screen_share_enabled: bool = False
    video_track_id: str | None = None
    screen_share_track_id: str | None = None
    video_codec: str | None = None
    video_resolution: tuple[int, int] | None = None
```

#### 4.3 Bandwidth Adaptation

- Monitor network quality via RTCP feedback
- Degrade video quality before audio quality
- Switch simulcast layers based on available bandwidth
- Pause video entirely under severe congestion (audio-only fallback)

**Effort estimate: 2-3 weeks**

---

## Effort Summary

| Phase | Scope | Effort | Dependency |
|-------|-------|--------|------------|
| **Phase 1** | Core models + WebRTC video transport | 2-4 weeks | None |
| **Phase 2** | Video pipeline + recording + SFU | 3-5 weeks | Phase 1 |
| **Phase 3** | Vision AI providers | 1-2 weeks | Phase 1 |
| **Phase 4** | Combined audio+video sessions | 2-3 weeks | Phase 1 |

**Total for full video support: 8-14 weeks**
**MVP (Phase 1 only): 2-4 weeks** — basic real-time video transport over WebRTC.
**High-value fast path (Phase 1 + Phase 3): 3-6 weeks** — video calls with AI vision.

---

## Key Architectural Decisions

These decisions should be made before implementation begins:

### 1. SFU vs MCU

| | SFU | MCU |
|---|-----|-----|
| Server load | Low (forward only) | High (decode + mix + encode) |
| Client load | Higher (decode N streams) | Lower (decode 1 stream) |
| Latency | Lower | Higher |
| Flexibility | Per-stream quality | Single composite |
| Industry standard | Yes (Zoom, Meet, Teams) | Legacy |

**Recommendation:** SFU as default. MCU as optional provider for constrained clients.

### 2. Server-Side vs Client-Side Processing

Video processing is 10-100x more expensive than audio processing.

| Processing | Server-side | Client-side |
|-----------|-------------|-------------|
| Background blur | GPU required | Browser MediaPipe / native SDK |
| Face detection | GPU required | Browser MediaPipe / native SDK |
| Recording | Server (reliable) | Client (unreliable) |
| Transcoding | Server (for SFU relay) | Not applicable |
| Overlays | Server | CSS/Canvas overlay |

**Recommendation:** Recording and transcoding on server. Background blur, face detection, and overlays on client. Server-side ML stages available as opt-in for headless/bot participants.

### 3. Codec Strategy

| Codec | Pros | Cons |
|-------|------|------|
| H.264 | Universal, hardware acceleration | Licensing (Cisco OpenH264 is free) |
| VP8 | Free, WebRTC mandatory-to-implement | Older, less efficient |
| VP9 | Free, better compression | Not universal hardware support |
| AV1 | Best compression, royalty-free | Limited hardware encode support |

**Recommendation:** VP8 as baseline (mandatory WebRTC codec). H.264 preferred when available. AV1 as future option.

### 4. Channel Architecture

**Recommendation:** Separate `VideoChannel` and `VoiceChannel` sharing a `MediaSession`. A `MediaChannel` convenience factory composes both. This keeps each channel focused and testable while allowing combined use.

### 5. RFC Conformance Level

**Recommendation:** Expand Level 3 from "Voice" to "Real-Time Media" encompassing both audio and video pipelines. Video-specific stages are OPTIONAL within Level 3.

---

## File Structure

```
src/roomkit/
  video/                          # New top-level video subsystem
    __init__.py
    base.py                       # VideoChunk, VideoCapability, MediaSession
    video_frame.py                # VideoFrame dataclass
    backends/
      __init__.py
      base.py                     # VideoBackend ABC
      webrtc.py                   # WebRTC video backend (aiortc or extend FastRTC)
      mock.py                     # MockVideoBackend for testing
    pipeline/
      __init__.py
      engine.py                   # VideoPipeline orchestrator
      config.py                   # VideoPipelineConfig
      decoder/
        base.py                   # VideoDecoderProvider ABC
        ffmpeg.py
        mock.py
      encoder/
        base.py                   # VideoEncoderProvider ABC
        ffmpeg.py
        mock.py
      resizer/
        base.py                   # VideoResizerProvider ABC
        opencv.py
        mock.py
      overlay/
        base.py                   # VideoOverlayProvider ABC
        opencv.py
        mock.py
      background/
        base.py                   # BackgroundBlurProvider ABC
        mediapipe.py
        mock.py
      face/
        base.py                   # FaceDetectorProvider ABC
        mediapipe.py
        mock.py
      recorder/
        base.py                   # VideoRecorder ABC
        ffmpeg.py
        mock.py
    vision/
      __init__.py
      base.py                     # VisionProvider ABC
      openai.py                   # OpenAI GPT-4o Vision
      gemini.py                   # Gemini Pro Vision
      mock.py                     # MockVisionProvider
    bridge.py                     # VideoBridge (SFU forwarding)
  channels/
    video.py                      # VideoChannel (new)
    __init__.py                   # Add VideoChannel factory
  models/
    enums.py                      # Add video hook triggers
```

---

## Dependencies

Required new dependencies (all optional extras):

| Package | Purpose | Phase |
|---------|---------|-------|
| `aiortc` | WebRTC video transport | Phase 1 |
| `av` (PyAV) | FFmpeg bindings for encode/decode | Phase 2 |
| `opencv-python-headless` | Image processing stages | Phase 2 |
| `mediapipe` | Background blur, face detection | Phase 2 |
| `pillow` | Lightweight image operations | Phase 2 |

All should be optional extras in `pyproject.toml`:

```toml
[project.optional-dependencies]
video = ["aiortc>=1.9.0", "av>=12.0.0"]
video-processing = ["opencv-python-headless>=4.9", "mediapipe>=0.10"]
video-vision = []  # Provider SDKs (openai, google-genai) already in ai extras
```

---

## Recommended Execution Order

1. **Start with Phase 1 + Phase 3 in parallel**
   - Phase 1 delivers the transport layer (video calls work end-to-end)
   - Phase 3 delivers AI video understanding (high value, lower effort, follows existing patterns)

2. **Phase 4 after Phase 1 stabilizes**
   - Combined audio+video sessions depend on a stable video transport

3. **Phase 2 last (or defer)**
   - Server-side video processing is the heaviest investment
   - Most video processing can be pushed to clients initially
   - Implement only recording first (highest demand), defer ML stages

---

## Success Criteria

- [ ] WebRTC video call between two participants in a room
- [ ] Video forwarding to 4+ participants via SFU
- [x] Video recording to file (MP4)
- [x] Vision AI can describe video call content to AIChannel
- [ ] Screen sharing works as a separate video track
- [ ] Graceful fallback: video degrades to audio-only under poor network
- [x] All new components have tests, docs, and examples
- [x] RFC updated with video transport specification
- [x] `make all` passes with video extras installed
