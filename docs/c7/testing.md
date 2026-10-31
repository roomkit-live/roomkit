# Testing

RoomKit provides mock implementations for every pluggable component: AI providers, voice backends, pipeline stages, identity resolvers, and telemetry.

## Test Setup

```bash
pip install roomkit[dev]
uv run pytest              # Run all tests
uv run pytest tests/test_framework.py -v  # Specific file
```

Configuration in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"  # No @pytest.mark.asyncio needed
```

## Basic Test Pattern

```python
from roomkit import RoomKit, InboundMessage, TextContent, WebSocketChannel


class TestMyFeature:
    async def test_message_delivery(self) -> None:
        kit = RoomKit()

        alice = WebSocketChannel("ws-alice")
        bob = WebSocketChannel("ws-bob")
        kit.register_channel(alice)
        kit.register_channel(bob)

        await kit.create_room(room_id="test-room")
        await kit.attach_channel("test-room", "ws-alice")
        await kit.attach_channel("test-room", "ws-bob")

        bob_inbox: list = []
        async def on_recv(_conn: str, event) -> None:
            bob_inbox.append(event)

        bob.register_connection("bob-conn", on_recv, room_id="test-room")

        result = await kit.process_inbound(
            InboundMessage(
                channel_id="ws-alice",
                sender_id="alice",
                content=TextContent(body="Hello"),
            )
        )

        assert not result.blocked
        # The room delivers to the other channels, never back to the sender's
        assert len(bob_inbox) == 1
        assert bob_inbox[0].content.body == "Hello"
```

## Recording Transport and Echo Agent

`roomkit.testing` gives a test or an example the two channels it needs to see
a room work without a provider, as roomkit-rs's `roomkit::testing` does:

```python
from roomkit import ChannelCategory, ChannelType, RoomKit
from roomkit.testing import EchoAgent, RecordingTransport

kit = RoomKit()
ws = RecordingTransport("ws", ChannelType.WEBSOCKET)
kit.register_channel(ws)
kit.register_channel(EchoAgent("echo", "You said: ", quote=True))
await kit.create_room(room_id="r1")
await kit.attach_channel("r1", "ws")
await kit.attach_channel("r1", "echo", category=ChannelCategory.INTELLIGENCE)
# ... a message into the room ...
assert ws.texts == ["Hello.", "You said: Hello."]
```

- `RecordingTransport(channel_id, channel_type, *, capabilities=None)` records
  every event delivered to it: `deliveries` holds each one as a
  `Delivery(event, text)`, the event as transcoded for it and its text as a
  person on the transport reads it; `texts` the texts alone. It needs no
  recipient, and addresses its senders as its type does (an SMS one in E.164).
  `capabilities` are those it declares: one that edits natively receives an
  edit as it is.
- `EchoAgent(channel_id, reply, *, quote=False)` answers every event it is
  asked with `reply`, followed by the event's text when `quote` is set.

## Mock AI Provider

```python
from roomkit.providers.ai import MockAIProvider
from roomkit import AIChannel, ChannelCategory

# Responds with pre-configured messages in order
provider = MockAIProvider(responses=["Response 1", "Response 2"])

ai = AIChannel("ai", provider=provider)
kit.register_channel(ai)
await kit.attach_channel("room", "ai", category=ChannelCategory.INTELLIGENCE)

# After processing, check what the AI was called with:
assert len(provider.calls) == 1
last_call = provider.calls[-1]
print(last_call.system_prompt)
print(last_call.temperature)
print([t.name for t in last_call.tools])
```

## Mock Voice Backend

```python
from roomkit.voice.backends import MockVoiceBackend

backend = MockVoiceBackend()

# Simulate audio input
backend.simulate_audio_received(session, audio_frame)
```

## Voice Test Bench

`roomkit.voice.testing` replaces frame-by-frame injection and `asyncio.sleep`
with a simulated phone and a hook timeline. `ScenarioVoiceBackend` extends
`MockVoiceBackend`: `play()` cuts a WAV or a `PCMAudio` clip into 20 ms frames
delivered at a transport's cadence (or back to back with `realtime=False`) and
captures what the bot sends per session. `VoiceTrace` records when each voice
hook fired, so a test waits for the hook it needs and reads the turn's order and
latencies off the timeline.

```python
from roomkit import HookTrigger, RoomKit, ScenarioVoiceBackend, VoiceTrace
from roomkit.voice.testing import read_wav, tone, write_wav

backend = ScenarioVoiceBackend()
kit = RoomKit(stt=stt, tts=tts, voice=backend)
trace = VoiceTrace(kit)                     # before the first session
# ... register the VoiceChannel, create the room, join the session ...

await backend.play(session, tone(300), realtime=False)   # or a WAV path
heard = await trace.wait_for(HookTrigger.ON_TRANSCRIPTION, timeout=2)
await trace.wait_for(HookTrigger.AFTER_TTS)              # TimeoutError names what did fire

speech_end = trace.first(HookTrigger.ON_SPEECH_END)
assert speech_end is not None
trace.elapsed_ms(speech_end, heard)                      # STT latency, ms
trace.sequence()                                         # the hooks, in order
backend.write_capture(session, "reports/bot.wav")        # what the bot said
```

`ScenarioVoiceBackend(mute_mic_during_playback=True)` drops the caller's frames
while the bot is speaking, as `LocalAudioBackend` does by default;
`is_speaking(session)` holds for as long as the bot's audio lasts. Use it to
test what a continuous STT stream does when the audio stops mid-session.

`read_wav`, `write_wav`, `pcm_frames`, `silence` and `tone` are the stdlib WAV
and PCM helpers around `PCMAudio`. Example: `examples/voice_scenario_backend.py`.

## Mock Pipeline Providers

Every pipeline stage has a mock that accepts pre-configured event sequences:

```python
from roomkit.voice.pipeline import (
    MockVADProvider,
    VADEvent,
    VADEventType,
    MockDenoiserProvider,
    MockDiarizationProvider,
    MockAGCProvider,
    MockAECProvider,
    MockDTMFDetector,
    MockAudioRecorder,
    MockTurnDetector,
    MockBackchannelDetector,
)

# VAD with event sequence
vad = MockVADProvider(events=[
    VADEvent(type=VADEventType.SPEECH_START),
    None,  # No event for this frame
    VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"speech"),
])

# Other mocks
denoiser = MockDenoiserProvider()
diarizer = MockDiarizationProvider()
agc = MockAGCProvider()
aec = MockAECProvider()
dtmf = MockDTMFDetector()
recorder = MockAudioRecorder()
turn = MockTurnDetector()
backchannel = MockBackchannelDetector()
```

## Mock STT/TTS

```python
from roomkit.voice.stt import MockSTTProvider
from roomkit.voice.tts import MockTTSProvider

stt = MockSTTProvider(transcripts=["Hello", "How are you?"])
tts = MockTTSProvider()
```

## Mock Identity Resolver

```python
from roomkit.identity import MockIdentityResolver

resolver = MockIdentityResolver()
kit = RoomKit(identity_resolver=resolver)
```

## Mock Realtime Provider

```python
from roomkit.voice.realtime import MockRealtimeProvider, MockRealtimeTransport

provider = MockRealtimeProvider()
transport = MockRealtimeTransport()
```

## Testing Hooks

```python
from roomkit import HookTrigger, HookResult, HookExecution

async def test_hook_blocks_message() -> None:
    kit = RoomKit()
    ws = WebSocketChannel("ws")
    kit.register_channel(ws)

    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "ws")

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def blocker(event, ctx):
        return HookResult.block("blocked")

    result = await kit.process_inbound(
        InboundMessage(
            channel_id="ws",
            sender_id="user",
            content=TextContent(body="test"),
        )
    )

    assert result.blocked
    assert result.reason == "blocked"
```

## Testing Voice Pipeline

```python
from roomkit import VoiceChannel
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider, VADEvent, VADEventType
from roomkit.voice.stt import MockSTTProvider
from roomkit.voice.tts import MockTTSProvider
from roomkit.voice.backends import MockVoiceBackend
from roomkit.voice import AudioFrame

async def test_voice_pipeline() -> None:
    kit = RoomKit()

    backend = MockVoiceBackend()
    stt = MockSTTProvider(transcripts=["Hello"])
    tts = MockTTSProvider()
    vad = MockVADProvider(events=[
        VADEvent(type=VADEventType.SPEECH_START),
        None,
        VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio"),
    ])

    voice = VoiceChannel(
        "voice", stt=stt, tts=tts, backend=backend,
        pipeline=AudioPipelineConfig(vad=vad),
    )
    kit.register_channel(voice)

    await kit.create_room(room_id="call")
    await kit.attach_channel("call", "voice")

    # Simulate audio input
    frame = AudioFrame(data=b"\x00" * 320, sample_rate=16000)
    backend.simulate_audio_received(None, frame)
```

## Testing Orchestration

```python
from roomkit import Agent, Pipeline, RoomKit, WebSocketChannel
from roomkit.providers.ai import MockAIProvider

async def test_pipeline_orchestration() -> None:
    agent1 = Agent("a1", provider=MockAIProvider(responses=["Transferring..."]))
    agent2 = Agent("a2", provider=MockAIProvider(responses=["Resolved!"]))

    kit = RoomKit(orchestration=Pipeline(agents=[agent1, agent2]))

    ws = WebSocketChannel("ws")
    kit.register_channel(ws)

    await kit.create_room(room_id="test")
    await kit.attach_channel("test", "ws")

    # First message goes to agent1
    result = await kit.process_inbound(
        InboundMessage(channel_id="ws", sender_id="user", content=TextContent(body="Help"))
    )
    assert not result.blocked
```

## Test Utilities

```python
# Pydantic model updates (immutable)
modified = event.model_copy(update={"content": TextContent(body="new")})

# Never mutate models directly — always use model_copy
```
