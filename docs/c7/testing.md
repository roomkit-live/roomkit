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

        ws = WebSocketChannel("ws-user")
        kit.register_channel(ws)

        inbox: list = []
        async def on_recv(_conn: str, event) -> None:
            inbox.append(event)

        ws.register_connection("conn", on_recv)

        await kit.create_room(room_id="test-room")
        await kit.attach_channel("test-room", "ws-user")

        result = await kit.process_inbound(
            InboundMessage(
                channel_id="ws-user",
                sender_id="user",
                content=TextContent(body="Hello"),
            )
        )

        assert not result.blocked
        assert len(inbox) == 1
        assert inbox[0].content.body == "Hello"
```

## Mock AI Provider

```python
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.channels.ai import AIChannel
from roomkit import ChannelCategory

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
from roomkit.voice.backends.mock import MockVoiceBackend

backend = MockVoiceBackend()

# Simulate audio input
backend.simulate_audio_received(session, audio_frame)
```

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
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

stt = MockSTTProvider(transcriptions=["Hello", "How are you?"])
tts = MockTTSProvider()
```

## Mock Identity Resolver

```python
from roomkit.identity.mock import MockIdentityResolver

resolver = MockIdentityResolver()
kit = RoomKit(identity_resolver=resolver)
```

## Mock Realtime Provider

```python
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport

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
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.audio_frame import AudioFrame

async def test_voice_pipeline() -> None:
    kit = RoomKit()

    backend = MockVoiceBackend()
    stt = MockSTTProvider(transcriptions=["Hello"])
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
from roomkit.providers.ai.mock import MockAIProvider

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
