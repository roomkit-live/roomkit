# Voice Pipeline

The audio pipeline sits between the voice backend and STT/TTS, processing audio through pluggable stages.

## Pipeline Architecture

```
Inbound:   Backend -> [Resampler] -> [Recorder] -> [AEC] -> [AGC] -> [Denoiser] -> VAD -> [Diarization] + [DTMF]
Outbound:  TTS -> [PostProcessors] -> [Recorder] -> AEC.feed_reference -> [Resampler] -> Backend
```

All stages are optional except VAD (required for speech detection). Stages in brackets are skipped if not configured.

## Pipeline Configuration

```python
from roomkit import VoiceChannel
from roomkit.voice.pipeline import AudioPipelineConfig, VADConfig

pipeline = AudioPipelineConfig(
    resampler=my_resampler,           # Sample rate conversion
    vad=my_vad_provider,              # Voice activity detection (required)
    denoiser=my_denoiser,             # Background noise removal
    diarization=my_diarizer,          # Speaker identification
    aec=my_aec,                       # Echo cancellation
    agc=my_agc,                       # Automatic gain control
    dtmf=my_dtmf_detector,            # DTMF tone detection
    recorder=my_recorder,             # Audio recording
    recording_config=my_rec_config,   # Recording settings
    turn_detector=my_turn_detector,   # Turn-taking detection
    vad_config=VADConfig(
        silence_threshold_ms=500,     # Silence before speech_end
    ),
)

voice = VoiceChannel(
    "voice",
    stt=stt,
    tts=tts,
    backend=backend,
    pipeline=pipeline,
)
```

## Capability-Aware Skipping

AEC and AGC stages automatically skip when the backend declares native capabilities:

```python
from roomkit.voice.base import VoiceCapability

# If backend has NATIVE_AEC, pipeline skips the AEC stage
# If backend has NATIVE_AGC, pipeline skips the AGC stage
```

## Pipeline Stages Reference

### Resampler

Converts audio between sample rates (e.g., 8kHz SIP to 16kHz STT).

```python
from roomkit.voice.pipeline.resampler import LinearResamplerProvider

resampler = LinearResamplerProvider()
# Or use SincResampler for higher quality
```

### VAD (Voice Activity Detection)

Detects speech start/end events. Required for the pipeline.

```python
from roomkit.voice.pipeline import MockVADProvider, VADEvent, VADEventType, VADConfig

# Mock for testing
vad = MockVADProvider(events=[
    VADEvent(type=VADEventType.SPEECH_START),
    None,  # No event for this frame
    VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"speech-data"),
])

# Production: SherpaOnnx VAD (local, offline)
from roomkit.voice import get_sherpa_onnx_vad_provider, get_sherpa_onnx_vad_config

SherpaVAD = get_sherpa_onnx_vad_provider()
SherpaVADConfig = get_sherpa_onnx_vad_config()
vad = SherpaVAD(SherpaVADConfig(threshold=0.5))
```

### AEC (Acoustic Echo Cancellation)

Removes echo from the microphone signal caused by speaker output.

```python
# Speex AEC
from roomkit.voice import get_speex_aec_provider

SpeexAEC = get_speex_aec_provider()
aec = SpeexAEC(sample_rate=16000, frame_size=160, filter_length=1024)

# WebRTC AEC
# pip install roomkit[webrtc-aec]
```

The pipeline feeds TTS audio as reference to the AEC via `process_outbound()`.
Reference audio is converted to the exact post-resampler capture format, and
AEC state and playback activation are isolated per voice session.
While playback is active, realtime backends keep the render timeline aligned
with capture by feeding hardware silence during jitter gaps. Ending playback
bypasses AEC but preserves its converged filter until the session ends.

### AGC (Automatic Gain Control)

Normalizes audio volume levels. AGC does not remove background noise; it makes
quiet and loud speech reach downstream denoising, VAD, and STT at a predictable
level.

```python
from roomkit.voice.pipeline import AGCConfig, AudioPipelineConfig, SimpleAGCProvider

agc = SimpleAGCProvider(AGCConfig(target_level_dbfs=-12.0))

# Or let the pipeline create the built-in provider from the config:
pipeline = AudioPipelineConfig(agc_config=AGCConfig(target_level_dbfs=-12.0))
```

Gain state is isolated per stream, near-silence is not amplified, and a peak
limiter prevents PCM clipping. Processed frames expose
`metadata.gain_applied_db`. Backends declaring `NATIVE_AGC` skip this stage.

### Denoiser

Removes background noise from audio.

```python
# Continuous WebRTC NS (same optional dependency as WebRTC AEC)
from roomkit.voice.pipeline import WebRTCNoiseSuppressorProvider

denoiser = WebRTCNoiseSuppressorProvider(sample_rate=24000)

# RNNoise (local, CPU-based)
from roomkit.voice import get_rnnoise_denoiser_provider

RNNoise = get_rnnoise_denoiser_provider()
denoiser = RNNoise()

# ai|coustics Quail (cloud API)
# pip install roomkit[aicoustics]

# SherpaOnnx denoiser (local ONNX model)
from roomkit.voice import get_sherpa_onnx_denoiser_provider, get_sherpa_onnx_denoiser_config

SherpaDenoiser = get_sherpa_onnx_denoiser_provider()
SherpaDenoiserConfig = get_sherpa_onnx_denoiser_config()
denoiser = SherpaDenoiser(SherpaDenoiserConfig())
```

WebRTC NS, RNNoise, and ai|coustics accept arbitrary caller chunk sizes without
duplicating or dropping audio; an irregular stream uses a fixed one-native-block
delay (10 ms for WebRTC NS and RNNoise). All built-in denoisers validate PCM
format before creating native state and keep model history isolated per stream.

### Diarization

Identifies different speakers in multi-speaker audio.

```python
from roomkit.voice.pipeline.diarization.mock import MockDiarizationProvider

diarizer = MockDiarizationProvider()
```

### DTMF Detection

Detects dual-tone multi-frequency signals (phone keypad tones).

```python
from roomkit.voice.pipeline.dtmf.mock import MockDTMFDetector

dtmf = MockDTMFDetector()
```

DTMF runs in parallel with other pipeline stages (before AEC/AGC/denoiser).

### Audio Recorder

Records inbound and outbound audio.

```python
from roomkit.voice.pipeline.recorder.mock import MockAudioRecorder
from roomkit.voice.pipeline.recorder import RecordingConfig

recorder = MockAudioRecorder()
config = RecordingConfig(
    format="wav",
    sample_rate=16000,
    channels=1,
)
```

### Turn Detector

Determines when a speaker's turn is complete (post-STT):

```python
from roomkit.voice.pipeline.turn.mock import MockTurnDetector

turn = MockTurnDetector()

# Production: Smart turn detection (ML-based)
# pip install roomkit[smart-turn]
```

Turn detection accumulates transcription fragments until `is_complete=True`, then routes the combined text.

### Backchannel Detector

Classifies short utterances as backchannel (e.g., "uh-huh", "yeah") to prevent false interruptions:

```python
from roomkit.voice.pipeline.backchannel.mock import MockBackchannelDetector

bc = MockBackchannelDetector()
```

Used with `InterruptionStrategy.SEMANTIC`.

### Post-Processors

Custom audio transformations on outbound TTS audio:

```python
from roomkit.voice.pipeline.postprocessor.base import AudioPostProcessor
```

## Interruption Handling

The `InterruptionHandler` manages what happens when the user speaks during TTS playback:

```python
from roomkit.voice.interruption import InterruptionConfig, InterruptionStrategy

config = InterruptionConfig(
    strategy=InterruptionStrategy.CONFIRMED,
    min_speech_ms=300,
    allow_during_first_ms=600,
)
```

For `RealtimeVoiceChannel` with provider-side VAD, `allow_during_first_ms`
guards the provider input after physical playback begins. RoomKit continues to
process the real microphone signal through AEC, AGC, denoising, and recording,
but forwards equal-duration PCM silence to the provider until the guard expires.
This gives an acoustic echo canceller time to converge without letting residual
onset echo trigger a server-side interruption. The guard requires a transport
with playback callbacks; set it to `0` on a headset or another already-clean
audio path when immediate barge-in is more important.

| Strategy | When to Use |
|----------|-------------|
| `IMMEDIATE` | Fast response, accept false positives |
| `CONFIRMED` | Balanced — waits for sustained speech |
| `SEMANTIC` | Ignore backchannel ("uh-huh") using BackchannelDetector |
| `DISABLED` | Never interrupt (e.g., announcements) |

## AudioFrame

Inbound audio is represented as `AudioFrame`:

```python
from roomkit.voice.audio_frame import AudioFrame

frame = AudioFrame(
    data=b"\x00" * 320,   # Raw PCM bytes
    sample_rate=16000,     # Hz
    channels=1,            # Mono
    sample_width=2,        # 16-bit PCM
    timestamp_ms=0.0,
)
```

Pipeline stages annotate `frame.metadata` as they process: `denoiser`, `vad`, `aec`, `agc`, `diarization`, `dtmf` keys.

## Mock Providers for Testing

Every pipeline stage has a mock provider with pre-configured event sequences:

```python
from roomkit.voice.pipeline import (
    MockVADProvider,
    MockDenoiserProvider,
    MockDiarizationProvider,
    MockAGCProvider,
    MockAECProvider,
    MockDTMFDetector,
    MockAudioRecorder,
    MockTurnDetector,
    MockBackchannelDetector,
)
```

Example with mock VAD:

```python
from roomkit.voice.pipeline import MockVADProvider, VADEvent, VADEventType

vad = MockVADProvider(events=[
    VADEvent(type=VADEventType.SPEECH_START),
    None,
    VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"speech"),
])
```
