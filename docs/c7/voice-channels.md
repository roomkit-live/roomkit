# Voice Channels

VoiceChannel handles real-time audio conversations with speech-to-text, text-to-speech, and an audio processing pipeline.

## Basic Voice Setup

```python
from roomkit import RoomKit, VoiceChannel
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider, VADEvent, VADEventType
from roomkit.voice.stt import MockSTTProvider
from roomkit.voice.tts import MockTTSProvider
from roomkit.voice.backends import MockVoiceBackend

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

A host serving several organizations passes `organization_id=` to `join()`:
the room is read with that scope first (RFC §17.2), so another
organization's room raises `RoomNotFoundError` and no session joins it or
reaches its recordings. Left unset, the read is unscoped.

## STT Providers

| Provider | Class | Config | Extra |
|----------|-------|--------|-------|
| Deepgram | `DeepgramSTTProvider` | `DeepgramConfig` | `roomkit[deepgram]` |
| SherpaOnnx | `SherpaOnnxSTTProvider` | `SherpaOnnxSTTConfig` | `roomkit[sherpa-onnx]` |
| Gradium | `GradiumSTTProvider` | `GradiumSTTConfig` | `roomkit[gradium]` |
| Qwen3 ASR | `Qwen3ASRProvider` | `Qwen3ASRConfig` | `roomkit[qwen-asr]` |
| Voxtral 3 (Mistral, file + realtime) | `VoxtralSTTProvider` | `VoxtralSTTConfig` | `roomkit[mistral-voice]` |
| Mock | `MockSTTProvider` | — | built-in |
| Rime | `RimeSTTProvider` | `RimeSTTConfig` | `roomkit[rime]` |

Rime STT accepts English or Spanish explicitly. It yields revised partials
and one final when the audio iterator ends. Configure a pipeline VAD, or
`batch_mode=True` with explicit `flush_stt()`; it has no native endpointing,
language detection or diarization. Streams last at most 120 seconds.

Use lazy loaders to avoid import-time dependency checks:

```python
from roomkit.voice.stt import DeepgramConfig, DeepgramSTTProvider

stt = DeepgramSTTProvider(DeepgramConfig(
    api_key="...",
    model="nova-2",
    language="en",
))
```

### STT language per session

A streaming STT fixes its language when the stream opens. RoomKit opens one
stream per utterance (VAD mode) or per turn (continuous mode; a provider that
diarizes or reports `keeps_stream_across_turns` keeps one stream across turns,
ended for a new language), so the language can change between them: `VoiceChannel.set_stt_language(session,
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
| SherpaOnnx (VITS/Piper, Kokoro) | `SherpaOnnxTTSProvider` | `SherpaOnnxTTSConfig` | `roomkit[sherpa-onnx]` |
| Gradium | `GradiumTTSProvider` | `GradiumTTSConfig` | `roomkit[gradium]` |
| Qwen3 | `Qwen3TTSProvider` | `Qwen3TTSConfig` | `roomkit[qwen-tts]` |
| NeuTTS | `NeuTTSProvider` | `NeuTTSConfig` | `roomkit[neutts]` |
| Grok TTS | `GrokTTSProvider` | `GrokTTSConfig` | xAI |
| Gemini TTS | `GeminiTTSProvider` | `GeminiTTSConfig` | `roomkit[gemini]` |
| Fish Audio (Drama 3, S2) | `FishAudioTTSProvider` | `FishAudioTTSConfig` | `roomkit[fish-audio]` |
| Phonon (Gradium, local CPU) | `PhononTTSProvider` | `PhononTTSConfig` | `roomkit[phonon]` |
| Voxtral 3 (Mistral) | `VoxtralTTSProvider` | `VoxtralTTSConfig` | `roomkit[mistral-voice]` |
| Mock | `MockTTSProvider` | — | built-in |
| Rime Coda / Mist v3 | `RimeTTSProvider` | `RimeTTSConfig` | `roomkit[rime]` |

Rime TTS streams mono PCM16 at 8/16/24/48 kHz and supports streamed text input.
`synthesize()` returns a WAV data URL. Each synthesis owns its connection, so
cancelling one leaves other calls active. Use `list_voices(language="fr")` to
query voices for the configured model; select a matching voice and language.
The standalone `examples/voice_rime.py` combines synthesis and transcription.

```python
from roomkit.voice.tts import ElevenLabsConfig, ElevenLabsTTSProvider

tts = ElevenLabsTTSProvider(ElevenLabsConfig(
    api_key="...",
    voice_id="21m00Tcm4TlvDq8ikWAM",
    model_id="eleven_flash_v2_5",
    output_format="pcm_16000",  # a VoiceChannel plays PCM only
))
```

A Voice Channel plays decoded 16-bit PCM (RFC §12.2): a TTS chunk in MP3,
Opus or G.711 is refused before a byte plays, with an error naming the
format. ElevenLabs streams MP3 by default, so give it `output_format="pcm_<rate>"`.

Kokoro runs locally through sherpa-onnx (`kokoro-int8-multi-lang-v1_0` from
sherpa-onnx's `tts-models` release). `lang` names the espeak-ng language of
text that is neither English nor Chinese; without it French is read as English.

Fish Audio's Drama 3 (`model="drama-3"`, the default) is directed in the text:
with `FishAudioTTSConfig(expressive=True)` it performs `[calm]`, `[sigh]` and
`<whisper>…</whisper>`; without it the text is spoken as written. A sentence
with no word to say is never sent (the model invents speech for one): on a
stream it goes with the next sentence. Do not put `StripBrackets` in front of
an expressive provider: it removes the directions.

```python
from roomkit.voice.tts import FishAudioTTSConfig, FishAudioTTSProvider

tts = FishAudioTTSProvider(FishAudioTTSConfig(api_key="...", voice="<voice id>", expressive=True))
```
A voice is an index into `voices.bin` (`ff_siwis` is 30). It speaks 24 kHz.

```python
from roomkit.voice.tts import SherpaOnnxTTSConfig, SherpaOnnxTTSProvider

d = "kokoro-int8-multi-lang-v1_0"
tts = SherpaOnnxTTSProvider(SherpaOnnxTTSConfig(
    model_type="kokoro", model=f"{d}/model.int8.onnx", voices=f"{d}/voices.bin",
    tokens=f"{d}/tokens.txt", data_dir=f"{d}/espeak-ng-data",
    lexicon=f"{d}/lexicon-us-en.txt,{d}/lexicon-zh.txt",
    lang="fr", speaker_id=30, sample_rate=24000,
))
```

### Voices, dialogue and custom voices

Every TTS lists its voices as `VoiceInfo` (`available_voices()` offline,
`list_voices(language=, gender=, query=)` live); `VoiceInfo.id` is what
`synthesize(voice=...)` takes, and the filters behave the same on every
provider (`"fr"` finds `fr-CA` and `fr-FR`). Gemini voices a two-speaker
dialogue in one clip, and `GeminiVoiceLibrary` designs or replicates custom
voices, replication requiring the person's recorded consent.

```python
from roomkit.voice.tts import GeminiTTSConfig, GeminiTTSProvider
from roomkit.voice import DialogueTurn

tts = GeminiTTSProvider(GeminiTTSConfig(api_key="..."))
quebec = await tts.list_voices(language="fr-CA", gender="female")   # 37 voices
audio = await tts.synthesize_dialogue(
    [DialogueTurn(speaker="A", text="Bonjour!"), DialogueTurn(speaker="B", text="Salut!")],
    {"A": quebec[0].id, "B": "fr-ca-advisor-2"},
)
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
from roomkit.voice.backends import LocalAudioBackend

backend = LocalAudioBackend(input_sample_rate=16000, channels=1)
```

`LocalAudioBackend` plays every response through one PortAudio output stream
whose callback is Python. `output_latency_ms` sets that stream's margin before
the device runs dry; the default (`None`) asks PortAudio for "low" with an
`aec`, "high" without it, and "high" on macOS:

```python
backend = LocalAudioBackend(aec=aec, output_latency_ms=100)
```

A wider margin stops the underflows, heard as clicks, that a stalled callback
causes: with an AEC, a C call holding the GIL for 50 ms every 400 ms gave 84
underflows over three responses at "low" and none at 100 ms. It costs latency:
a wider margin delays each response and each cut by as much.

With an `aec`, the margin adds about as much to the echo delay (100 ms asked:
echo at about 149 ms instead of 38); keep it inside the canceller's range:
WebRTC about 500 ms in all, Speex its `filter_length` (200 ms by default). A
delay past WebRTC's 500 ms is logged as a warning, once per session. The AEC's
echo delay can still drift under such stalls, more with the wider margin, and
can let more echo through (RMK-973). A value that is not positive and finite
raises `ValueError`.

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
from roomkit.voice.backends import LocalAudioBackend

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
from roomkit.voice import InterruptionConfig, InterruptionStrategy

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

Several transports in one room: build the channel on one backend and add the
others with `voice.add_backend(sip_backend)`. The added backend's audio goes
through the same pipeline, bridge and STT; everything sent to one of its
sessions (bridged audio, TTS, transcriptions, the hang-up of `kit.leave()`)
goes out on it. A session finds its transport through `kit.join(..., backend=)`
or by the added backend reporting it holds the session. The channel needs a primary
backend (else `RuntimeError`); closing it closes the added backends.

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
