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
| **Tier 1** | Real-time video transport (RTP/SIP video calls) | **Done** (RTP, SIP, Local, Screen) |
| **Tier 2** | Video processing pipeline (transcoding, scaling, overlays, recording) | **Partial** (Recording + VP9 transcode done) |
| **Tier 3** | Video intelligence (vision AI, frame analysis, captioning) | **Done** (OpenAI, Gemini, Mock) |
| **Tier 4** | Combined A/V channel | **Done** (AudioVideoChannel for SIP) |

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

**Status:** 1.1 ✅ | 1.2 ✅ | 1.3 ✅ | 1.4 ✅ (Local + RTP + SIP) | 1.5 ✅ | 1.6 ✅

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

#### 1.4 Video Backends ✅

| Backend | Transport | Codecs | Status |
|---------|-----------|--------|--------|
| `LocalVideoBackend` | Webcam (OpenCV) | raw_rgb24 | ✅ Done |
| `RTPVideoBackend` | RTP/UDP (aiortp) | H.264, VP9, VP8 | ✅ Done |
| `SIPVideoBackend` | SIP+RTP (aiosipua) | H.264, VP9, VP8 | ✅ Done |
| `ScreenCaptureBackend` | Screen capture (mss) | raw_rgb24 | ✅ Done |
| WebRTC Backend | WebRTC (aiortc) | VP8/VP9/H.264 | 🔲 Future |

**RTP/SIP highlights:**
- VP9 RTP depacketization (RFC 9628) with B/E-bit frame boundaries
- H.264 FU-A depacketization (RFC 6184)
- Jitter buffer with marker-bit frame delivery (instant, no next-timestamp wait)
- PLI-based keyframe recovery on packet loss
- NACK for packet retransmission
- PortAllocator for multi-session port management
- SDP codec negotiation (offerer preference order)

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

**Status:** 2.1 🔲 | 2.2 partial (Recorder: OpenCV ✅ + PyAV/FFmpeg ✅ + VP9 transcode ✅) | 2.3 🔲 | 2.4 🔲

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
| **Recorder** | `VideoRecorder` | Record to MP4/WebM file or S3 | **OpenCV ✅**, **PyAV/FFmpeg ✅** (H.264, H.265, NVENC) |
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

### Phase 2.5 — Room-Level Media Recording (Audio+Video Muxing)

**Goal:** Record audio and video into a single container file (MP4) with per-participant track attribution. Independent of Phase 4 — works with today's separate VoiceChannel + VideoChannel.

**Status:** ✅ Complete — `MediaRecorder` ABC, `PyAVMediaRecorder`, `MockMediaRecorder`, `RoomRecorderManager`, framework wiring, tests, docs, example

#### Design Principles

- **Event-driven** — the recorder subscribes to room lifecycle events, no explicit `tap_xxx()` methods per media type
- **Declarative** — channels declare what they contribute via `ChannelRecordingConfig`, the framework wires everything
- **Room-level concern** — recording is configured on the room, not on individual channels
- **Multi-recorder** — a room can have multiple recorders (local file + cloud upload, full quality + preview)
- **Participant attribution** — each participant gets a separate audio track, identified by `participant_id`

#### Data Models

All in `recorder/base.py`:

```python
@dataclass
class RecordingTrack:
    """A media track in a recording."""
    id: str
    kind: str                        # "audio", "video", "screen_share"
    channel_id: str
    participant_id: str | None = None
    codec: str = ""                  # "raw_rgb24", "pcm_s16le", etc.
    sample_rate: int | None = None   # audio only
    width: int | None = None         # video only
    height: int | None = None        # video only


@dataclass
class ChannelRecordingConfig:
    """What a channel contributes to room recording."""
    audio: bool = False
    video: bool = False
    screen_share: bool = False
    per_participant: bool = True     # separate track per participant, or mixed


@dataclass
class MediaRecordingConfig:
    """Room-level recording output configuration."""
    storage: str = ""
    video_codec: str = "libx264"     # libx264 (default), h264_nvenc, libx265
    audio_codec: str = "aac"         # aac, opus
    audio_sample_rate: int = 16000
    format: str = "mp4"


@dataclass
class RoomRecorderBinding:
    """Binds a recorder to a room with its output configuration."""
    recorder: MediaRecorder
    config: MediaRecordingConfig
    enabled: bool = True             # pause/resume without removing
    name: str = ""                   # optional label ("archive", "preview", "debug")
```

#### MediaRecorder ABC

```python
class MediaRecorder(ABC):
    """Event-driven multi-track recorder."""

    @abstractmethod
    def on_recording_start(self, config: MediaRecordingConfig) -> RecordingHandle: ...

    @abstractmethod
    def on_recording_stop(self, handle: RecordingHandle) -> RecordingResult: ...

    @abstractmethod
    def on_track_added(self, handle: RecordingHandle, track: RecordingTrack) -> None: ...

    @abstractmethod
    def on_track_removed(self, handle: RecordingHandle, track: RecordingTrack) -> None: ...

    @abstractmethod
    def on_data(self, handle: RecordingHandle, track: RecordingTrack,
                data: bytes, timestamp_ms: float) -> None: ...

    def close(self) -> None: ...
```

The recorder is **generic** — no media-type-specific methods. `on_data()` receives bytes with a `RecordingTrack` that describes the format. This extends naturally to screen share, subtitles, or any future track type without new methods.

#### Framework Wiring (automatic)

```
attach_channel("door", "cam")          → cam.recording.video == True
  connect_video(session)               → recorder.on_track_added(video_track)
    backend delivers frame             → recorder.on_data(video_track, pixels, ts)

attach_channel("door", "voice")        → voice.recording.audio == True
  voice session starts                 → recorder.on_track_added(audio_track)
    pipeline delivers PCM              → recorder.on_data(audio_track, pcm, ts)

disconnect / detach                    → recorder.on_track_removed(track)
room close                             → recorder.on_recording_stop(handle)
```

#### Output Structure

With `per_participant=True`, the MP4 container holds separate tracks per participant:

```
recordings/door_20260312T143022.mp4
  Track 0: H.264 video  (cam — visitor's camera)
  Track 1: AAC audio    (voice — participant "visitor")
  Track 2: AAC audio    (voice — participant "ai-agent")
```

#### Multi-Recorder Example

A room can bind multiple recorders for different purposes:

```python
await kit.create_room(
    room_id="door",
    recorders=[
        RoomRecorderBinding(
            recorder=PyAVMediaRecorder(),
            config=MediaRecordingConfig(
                storage="./recordings",
                video_codec="auto",
                audio_codec="aac",
            ),
            name="local",
        ),
        RoomRecorderBinding(
            recorder=PyAVMediaRecorder(),
            config=MediaRecordingConfig(
                storage="./archive",
                video_codec="libx265",
                audio_codec="opus",
            ),
            name="archive",
            enabled=False,  # activate later on demand
        ),
    ],
)
```

#### Usage Example

```python
video = VideoChannel("cam", backend=video_backend,
    recording=ChannelRecordingConfig(video=True),
)
voice = VoiceChannel("voice", backend=audio_backend, stt=stt, tts=tts,
    recording=ChannelRecordingConfig(audio=True, per_participant=True),
)

kit.register_channel(video)
kit.register_channel(voice)

await kit.create_room(
    room_id="door",
    recorders=[
        RoomRecorderBinding(
            recorder=PyAVMediaRecorder(),
            config=MediaRecordingConfig(storage="./recordings", video_codec="auto", audio_codec="aac"),
            name="local",
        ),
    ],
)

await kit.attach_channel("door", "cam")
await kit.attach_channel("door", "voice")

# Recording starts automatically when sessions connect
session = await kit.connect_video("door", "visitor", "cam")
await kit.connect_voice("door", "visitor", "voice")
```

Zero wiring code — channels declare intent, room owns the recorders, framework delivers events.

#### Relationship to Existing Recorders

| Recorder | Scope | Status |
|----------|-------|--------|
| `VideoRecorder` (ABC) | Video-only, channel-level | **Done** (backward compat) |
| `OpenCVVideoRecorder` | Video-only, raw MP4 | **Done** |
| `PyAVVideoRecorder` | Video-only, H.264/NVENC | **Done** |
| `WavFileRecorder` | Audio-only, voice pipeline | **Done** |
| `MediaRecorder` (ABC) | Multi-track, room-level | **Done** |
| `PyAVMediaRecorder` | A/V muxed, H.264+AAC | **Done** |

The existing `VideoRecorder` and `WavFileRecorder` continue to work for single-media channel-level recording. `MediaRecorder` is the evolution for room-level multi-track recording.

**Effort estimate: 1-2 weeks** (MediaRecorder ABC + PyAVMediaRecorder + framework wiring + tests) — **Completed.**

**Dependencies:** None — works with current separate VoiceChannel + VideoChannel. Does NOT require Phase 4 (MediaSession).

#### Implementation Notes

- **A/V sync**: Per-track PTS — each stream starts at PTS=0 from its own first frame timestamp. Eliminates startup offset between mic and camera (or audio/video RTP).
- **VP9 transcode**: VP9 frames from RTP are decoded then re-encoded as H.264 for MP4. Dimensions probed from the first decodable VP9 keyframe before creating the encoder stream.
- **libx264 zerolatency**: Forces immediate output from first frame to prevent MP4 muxer EINVAL (the muxer rejects audio when video stream has zero packets).
- **Encoded frame handling**: H.264 gets Annex B start codes; VP9/VP8/AV1 are fed as raw bitstream. Codec-specific start code detection is idempotent.
- **Pre-roll buffering**: Data buffered per-track until all registered tracks receive at least one frame. Container + all streams created at once with known parameters.
- **Keyframe gating**: For encoded video, if the probe can't decode frames (e.g., first frame is a P-frame), the video stream is skipped and audio records solo.
- **Thread safety**: Per-recording `threading.Lock` — video frames from capture thread and audio from event loop can write concurrently.
- **Mux error gating**: First mux error logged with full traceback; subsequent errors suppressed to prevent log flooding.
- **Recording lifecycle**: Tied to `close_room()` — `disconnect_voice()` removes tracks but does NOT stop the recording.
- **Files**: `recorder/base.py` (models + ABC), `recorder/pyav.py` (PyAV muxer), `recorder/_pyav_mux.py` (mux helpers — probe, stream creation, PTS, safe_mux), `recorder/mock.py` (testing), `recorder/_room_recorder_manager.py` (orchestration), framework wiring in `core/framework.py` and `core/_room_lifecycle.py`.
- **Docs**: Guide at `roomkit-docs/docs/guides/room-media-recorder.md`, features.md updated.
- **Examples**: `examples/room_media_recorder.py` (mic + webcam → MP4), `examples/sip_video_call.py` (SIP A/V call recording).

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

**Status:** ✅ Core done (AudioVideoChannel for SIP A/V) | 4.3 🔲

#### 4.1 AudioVideoChannel ✅

Implemented as `AudioVideoChannel` (`channels/av.py`), extending `VoiceChannel` with `VideoBackend` integration:

```python
av = AudioVideoChannel(
    "voice",
    stt=stt, tts=tts,
    backend=SIPVideoBackend(
        local_sip_addr=("0.0.0.0", 5060),
        supported_video_codecs=["H264", "VP9"],
    ),
    pipeline=AudioPipelineConfig(),
    recording=ChannelRecordingConfig(audio=True, video=True),
)
```

- SIP backend handles both audio and video RTP sessions
- Video frames delivered via backend taps (not a separate channel)
- Recording wired automatically for both audio and video tracks
- Works with SIPVideoBackend (combined A/V) or separate backends

#### 4.2 Track Management

Video session state is tracked per-voice-session in the backend:

- `SIPVideoBackend._video_call_sessions` — VP9/H.264 RTP sessions
- `SIPVideoBackend._video_sessions` — VideoSession objects
- Video codec and resolution learned from first decoded frame
- Video track added/removed alongside audio track in recording

#### 4.3 Bandwidth Adaptation 🔲

- Monitor network quality via RTCP feedback
- Degrade video quality before audio quality
- Switch simulcast layers based on available bandwidth
- Pause video entirely under severe congestion (audio-only fallback)

**Remaining effort: 1-2 weeks** (bandwidth adaptation only)

---

## Effort Summary

| Phase | Scope | Effort | Status |
|-------|-------|--------|--------|
| **Phase 1** | Core models + video transport | 2-4 weeks | **✅ Done** (RTP/SIP/Local/Screen) |
| **Phase 2** | Video pipeline + recording + SFU | 3-5 weeks | **Partial** (recording done, pipeline/SFU remaining) |
| **Phase 2.5** | Room-level media recording (A/V muxing) | 1-2 weeks | **✅ Done** (VP9 transcode, per-track sync) |
| **Phase 3** | Vision AI providers | 1-2 weeks | **✅ Done** (OpenAI, Gemini, Mock) |
| **Phase 4** | Combined audio+video sessions | 2-3 weeks | **✅ Done** (AudioVideoChannel, SIP A/V) |

**Remaining work:**
- Phase 2: VideoPipeline engine, SFU/VideoBridge, processing stages (decoder, resizer, overlay, background blur, face detection) — **3-5 weeks**
- WebRTC backend (aiortc) — **2-3 weeks**
- Bandwidth adaptation — **1-2 weeks**

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
  channels/
    av.py                         # AudioVideoChannel (combined A/V for SIP)
    video.py                      # VideoChannel
  video/                          # Video subsystem
    __init__.py
    base.py                       # VideoChunk, VideoCapability, VideoSession
    video_frame.py                # VideoFrame dataclass
    backends/
      __init__.py
      base.py                     # VideoBackend ABC + get_video_session()
      local.py                    # LocalVideoBackend (webcam via OpenCV)
      rtp.py                      # RTPVideoBackend (aiortp, H.264/VP9)
      sip.py                      # SIPVideoBackend (aiosipua, SIP+RTP A/V)
      screen.py                   # ScreenCaptureBackend (mss)
      mock.py                     # MockVideoBackend for testing
      webrtc.py                   # 🔲 WebRTC video backend (future)
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

| Package | Purpose | Phase | Status |
|---------|---------|-------|--------|
| `aiortp>=0.3.0` | RTP video transport (VP9/H.264/VP8) | Phase 1 | **Installed** (`roomkit[rtp]`) |
| `aiosipua>=0.4.0` | SIP signaling + RTP bridge | Phase 1 | **Installed** (`roomkit[sip]`) |
| `av>=12.0.0` (PyAV) | FFmpeg bindings for encode/decode/record | Phase 2 | **Installed** (`roomkit[video]`) |
| `opencv-python-headless` | Image processing + basic recording | Phase 2 | **Installed** (`roomkit[local-video]`) |
| `aiortc` | WebRTC video transport | Future | Not started |
| `mediapipe` | Background blur, face detection | Future | Not started |
| `pillow` | Lightweight image operations | Future | Not started |

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
- [x] Video recording to file (MP4) — OpenCV + PyAV/FFmpeg (H.264, NVENC)
- [x] Combined A/V recording to single MP4 with per-participant audio tracks
- [x] SIP A/V call with VP9 video → H.264 recording with synced audio
- [x] VP9 RTP depacketization with instant frame delivery (marker-bit jitter buffer)
- [x] PLI-based keyframe recovery on packet loss
- [x] AudioVideoChannel for combined SIP audio+video sessions
- [x] Vision AI can describe video call content to AIChannel
- [x] Screen capture backend for screen sharing
- [ ] Screen sharing works as a separate video track in multi-party
- [ ] Graceful fallback: video degrades to audio-only under poor network
- [x] All new components have tests, docs, and examples
- [x] RFC updated with video transport specification
- [x] `make all` passes with video extras installed
