# Voice Channels

VoiceChannel handles real-time audio conversations with speech-to-text, text-to-speech, and an audio processing pipeline.

## Basic Voice Setup

```python
from roomkit import RoomKit, VoiceChannel
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider, VADEvent, VADEventType
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider
from roomkit.voice.backends.mock import MockVoiceBackend

kit = RoomKit()

# Create providers
backend = MockVoiceBackend()
stt = MockSTTProvider(transcripts=["Hello, how can I help?"])
tts = MockTTSProvider()
vad = MockVADProvider(events=[
    VADEvent(type=VADEventType.SPEECH_START),
    None,
    VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio"),
])

# Create voice channel with pipeline
voice = VoiceChannel(
    "voice-agent",
    stt=stt,
    tts=tts,
    backend=backend,
    pipeline=AudioPipelineConfig(vad=vad),
)
kit.register_channel(voice)
```

## Joining a Voice Session

```python
# Create room and attach voice channel
await kit.create_room(room_id="call-room")
await kit.attach_channel("call-room", "voice-agent")

# Join a participant to the voice session
session = await kit.join(
    room_id="call-room",
    channel_id="voice-agent",
    participant_id="caller-1",
)

# Leave when done
await kit.leave(session)
```

## STT Providers

| Provider | Class | Config | Extra |
|----------|-------|--------|-------|
| Deepgram | `DeepgramSTTProvider` | `DeepgramConfig` | `roomkit[deepgram]` |
| SherpaOnnx | `SherpaOnnxSTTProvider` | `SherpaOnnxSTTConfig` | `roomkit[sherpa-onnx]` |
| Gradium | `GradiumSTTProvider` | `GradiumSTTConfig` | `roomkit[gradium]` |
| Qwen3 ASR | `Qwen3ASRProvider` | `Qwen3ASRConfig` | `roomkit[qwen-asr]` |
| Mock | `MockSTTProvider` | — | built-in |

Use lazy loaders to avoid import-time dependency checks:

```python
from roomkit.voice import get_deepgram_provider, get_deepgram_config

DeepgramSTTProvider = get_deepgram_provider()
DeepgramConfig = get_deepgram_config()

stt = DeepgramSTTProvider(DeepgramConfig(
    api_key="...",
    model="nova-2",
    language="en",
))
```

### STT language per session

A streaming STT fixes its language when the stream opens. RoomKit opens one
stream per utterance (VAD mode) or per turn (continuous mode), so the
language can change between them: `VoiceChannel.set_stt_language(session,
"fr-CA")` applies from the session's next stream, `None` returns to the
provider's configuration, `get_stt_language(session)` reads it back. Only
providers with `supports_language_override` (Deepgram, the mock) are handed
a language. `TranscriptionResult.language`, `TranscriptionEvent.language`
and `PartialTranscriptionEvent.language` carry what the provider reported —
Deepgram Nova-3 in `language="multi"` reports the language most words were
tagged with; a pinned stream reports nothing.

`STTLanguageLock` packages the detect-then-lock loop:

```python
from roomkit import STTLanguageLock, VoiceChannel

voice = VoiceChannel(
    "voice",
    stt=DeepgramSTTProvider(DeepgramConfig(api_key="...", model="nova-3", language="multi")),
    backend=backend,
    pipeline=AudioPipelineConfig(),
    stt_language_lock=STTLanguageLock(
        detect_language="multi",   # every session starts here
        prefer={"fr": "fr-CA"},    # reported -> locked
        lock_after=1,              # agreeing finals before locking
        release_after=2,           # consecutive misses (no text, low confidence,
    ),                             # another language) before detecting again
)
```

Or from a hook, with the same primitives:

```python
@kit.hook(HookTrigger.ON_TRANSCRIPTION)
async def pin_language(event, ctx):
    if event.language == "fr":
        voice.set_stt_language(event.session, "fr-CA")
    return HookResult.allow()
```

## TTS Providers

| Provider | Class | Config | Extra |
|----------|-------|--------|-------|
| ElevenLabs | `ElevenLabsTTSProvider` | `ElevenLabsConfig` | `roomkit[elevenlabs]` |
| SherpaOnnx | `SherpaOnnxTTSProvider` | `SherpaOnnxTTSConfig` | `roomkit[sherpa-onnx]` |
| Gradium | `GradiumTTSProvider` | `GradiumTTSConfig` | `roomkit[gradium]` |
| Qwen3 | `Qwen3TTSProvider` | `Qwen3TTSConfig` | `roomkit[qwen-tts]` |
| NeuTTS | `NeuTTSProvider` | `NeuTTSConfig` | `roomkit[neutts]` |
| Grok TTS | `GrokTTSProvider` | `GrokTTSConfig` | xAI |
| Gemini TTS | `GeminiTTSProvider` | `GeminiTTSConfig` | `roomkit[gemini]` |
| Mock | `MockTTSProvider` | — | built-in |

```python
from roomkit.voice import get_elevenlabs_provider, get_elevenlabs_config

ElevenLabsTTSProvider = get_elevenlabs_provider()
ElevenLabsConfig = get_elevenlabs_config()

tts = ElevenLabsTTSProvider(ElevenLabsConfig(
    api_key="...",
    voice_id="21m00Tcm4TlvDq8ikWAM",
    model="eleven_turbo_v2",
))
```

## Voice Backends

Backends handle audio transport between the framework and participants:

| Backend | Class | Extra | Use Case |
|---------|-------|-------|----------|
| Local mic/speaker | `LocalAudioBackend` | `roomkit[local-audio]` | Development/testing |
| FastRTC (WebRTC) | `FastRTCVoiceBackend` | `roomkit[fastrtc]` | Browser-based voice |
| RTP | `RTPVoiceBackend` | `roomkit[rtp]` | VoIP integration |
| SIP | `SIPVoiceBackend` | `roomkit[sip]` | Telephony |
| WebTransport | `WebTransportBackend` | `roomkit[webtransport]` | Low-latency web |
| Mock | `MockVoiceBackend` | built-in | Testing |

```python
from roomkit.voice import get_local_audio_backend

LocalAudioBackend = get_local_audio_backend()
backend = LocalAudioBackend(sample_rate=16000, channels=1)
```

## Shared Capture Sources

A backend takes the microphone when a session starts and releases it when the
session ends. An `AudioCaptureSource` owns the device instead, so anything that
must listen *before* a session exists — a wake word, a level meter — keeps
running, and the session becomes one subscriber among several.

| Source | Class | Extra | Use Case |
|--------|-------|-------|----------|
| Local microphone | `LocalMicSource` | `roomkit[local-audio]` | Wake word, always-on capture |
| Mock | `MockCaptureSource` | built-in | Testing |

```python
from roomkit.voice.capture import LocalMicSource
from roomkit.voice.backends.local import LocalAudioBackend

mic = LocalMicSource(sample_rate=24000, backlog_seconds=10)
mic.start()                                        # the device opens once
detector = mic.subscribe(enqueue, name="wakeword") # no session in sight

transport = LocalAudioBackend(source=mic)          # a subscriber, not the owner

mark = mic.mark()                                  # at SPEECH_START
await channel.start_session(                       # once the trigger matched
    room_id, participant_id, connection=None,
    metadata={"capture_since": mark},
)
```

The source retains recent audio in a bounded ring. `mark()` names a position in
it and `subscribe(since=mark)` replays from there before going live; a mark that
has aged out replays what remains and reports `truncated`. Replayed frames take
the ordinary inbound path — no new control point.

Fan-out is synchronous on the capture thread, which keeps AEC's capture and
reference timing in step, so a subscriber must enqueue the frame and return.
`start()` / `stop()` alone acquire and release the device: dropping to zero
subscribers never stops capture, and a detector can safely detach for the
duration of a call — which it should, since source frames are pre-AEC.

`LocalAudioBackend(source=...)` derives its input format from the source and
rejects a contradicting explicit argument; with no source its behaviour is
unchanged.

## Interruption Handling

Four strategies for handling user speech during TTS playback:

```python
from roomkit.voice.interruption import InterruptionConfig, InterruptionStrategy

voice = VoiceChannel(
    "voice",
    stt=stt,
    tts=tts,
    backend=backend,
    pipeline=AudioPipelineConfig(vad=vad),
    interruption=InterruptionConfig(
        strategy=InterruptionStrategy.CONFIRMED,
        min_speech_ms=300,  # Wait 300ms of sustained speech before interrupting
    ),
)
```

| Strategy | Behavior |
|----------|----------|
| `IMMEDIATE` | Interrupt on any detected speech |
| `CONFIRMED` | Wait for sustained speech (min_speech_ms). Default. |
| `SEMANTIC` | Use BackchannelDetector to ignore "uh-huh", "yeah" |
| `DISABLED` | Never interrupt TTS playback |

## Voice Greeting

Send a greeting when a session starts:

```python
await kit.send_greeting(
    room_id="call-room",
    channel_id="voice-agent",
    greeting="Welcome! How can I help you today?",
    session=session,
)
```

Or configure on the Agent:

```python
from roomkit import Agent

agent = Agent(
    "voice-agent",
    provider=provider,
    greeting="Welcome! How can I help you today?",
    stt=stt,
    tts=tts,
    backend=backend,
    pipeline=AudioPipelineConfig(vad=vad),
)
```

## Voice Hooks

```python
from roomkit import HookTrigger, HookExecution

@kit.hook(HookTrigger.ON_SPEECH_START, execution=HookExecution.ASYNC)
async def on_speech(event, ctx):
    print("User started speaking")

@kit.hook(HookTrigger.ON_TRANSCRIPTION, execution=HookExecution.ASYNC)
async def on_transcription(event, ctx):
    print(f"Transcription: {event.content.body}")

@kit.hook(HookTrigger.BEFORE_TTS)
async def before_tts(event, ctx):
    # Can modify or block TTS text
    return HookResult.allow()

@kit.hook(HookTrigger.ON_BARGE_IN, execution=HookExecution.ASYNC)
async def on_barge_in(event, ctx):
    print("User interrupted the AI")
```

## Audio Bridging

Bridge audio between sessions for human-to-human voice calls:

```python
voice = VoiceChannel("voice", backend=backend, bridge=True)

# With bridge + STT for live transcription
voice = VoiceChannel("voice", stt=stt, backend=backend, bridge=True)
```

Audio bridge supports N-party calls with mixing and cross-rate resampling.

## DTMF

Send and detect DTMF tones:

```python
# Send DTMF
await voice.send_dtmf(session, digit="1", duration_ms=160)

# Detect DTMF via hook
@kit.hook(HookTrigger.ON_DTMF, execution=HookExecution.ASYNC)
async def on_dtmf(event, ctx):
    print(f"DTMF digit: {event.data['digit']}")
```
