# Realtime Voice (Speech-to-Speech)

RealtimeVoiceChannel connects to speech-to-speech AI models that handle audio directly — no separate STT/TTS needed. The AI model receives audio and responds with audio.

## Basic Setup

```python
from roomkit import RoomKit, ChannelCategory
from roomkit.channels.realtime_voice import RealtimeVoiceChannel

kit = RoomKit()

# Gemini Live example
from roomkit.voice import get_gemini_live_provider

GeminiLiveProvider = get_gemini_live_provider()

realtime = RealtimeVoiceChannel(
    "realtime-voice",
    provider=GeminiLiveProvider(
        api_key="...",
        model="gemini-2.0-flash-live-001",
    ),
    system_prompt="You are a helpful voice assistant. Keep responses brief.",
)
kit.register_channel(realtime)

await kit.create_room(room_id="voice-room")
await kit.attach_channel("voice-room", "realtime-voice", category=ChannelCategory.INTELLIGENCE)
```

## Providers

| Provider | Class | Extra | Description |
|----------|-------|-------|-------------|
| Google Gemini Live | `GeminiLiveProvider` | `roomkit[realtime-gemini]` | Gemini 2.0 speech-to-speech |
| OpenAI Realtime | `OpenAIRealtimeProvider` | `roomkit[realtime-openai]` | GPT-4o realtime audio |
| xAI Grok | `XAIRealtimeProvider` | — | Grok speech-to-speech |
| Mock | `MockRealtimeProvider` | built-in | Testing |

```python
# OpenAI Realtime
from roomkit.voice import get_openai_realtime_provider

OpenAIRealtime = get_openai_realtime_provider()
provider = OpenAIRealtime(api_key="sk-...", model="gpt-4o-realtime-preview")

# xAI Grok
from roomkit.voice import get_xai_realtime_provider, get_xai_realtime_config

XAIRealtime = get_xai_realtime_provider()
XAIConfig = get_xai_realtime_config()
provider = XAIRealtime(XAIConfig(api_key="..."))
```

## Audio Transports

Transports handle the client-side audio (browser or device):

| Transport | Class | Extra | Use Case |
|-----------|-------|-------|----------|
| WebSocket | `WebSocketRealtimeTransport` | `roomkit[websocket]` | Browser via WebSocket |
| FastRTC (WebRTC) | `FastRTCRealtimeTransport` | `roomkit[fastrtc]` | Browser via WebRTC |
| Local mic/speaker | `LocalAudioBackend` | `roomkit[local-audio]` | Local development |
| Mock | `MockRealtimeTransport` | built-in | Testing |

```python
from roomkit.voice import get_websocket_realtime_transport

WSTransport = get_websocket_realtime_transport()
transport = WSTransport(host="0.0.0.0", port=8765)

realtime = RealtimeVoiceChannel(
    "realtime-voice",
    provider=provider,
    transport=transport,
    system_prompt="You are a helpful assistant.",
)
```

## Joining a Session

```python
session = await kit.join(
    room_id="voice-room",
    channel_id="realtime-voice",
    participant_id="caller-1",
)

# Leave when done
await kit.leave(session)
```

## Tool Calling

Realtime voice channels support tool calling during conversations:

```python
from roomkit.channels.realtime_voice import RealtimeVoiceChannel, ToolHandler

async def handle_tool(name: str, arguments: dict) -> str:
    if name == "get_weather":
        return '{"temperature": 22, "condition": "sunny"}'
    return '{"error": "unknown tool"}'

realtime = RealtimeVoiceChannel(
    "realtime-voice",
    provider=provider,
    system_prompt="You help with weather. Use the get_weather tool.",
    tools=[
        {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    ],
    tool_handler=handle_tool,
)
```

## Text Injection

Inject text into an active realtime session (appears as if the AI said it):

```python
session = await kit.join("room", "realtime-voice", participant_id="user")
await realtime.inject_text(session, "Let me check that for you...")
```

## VAD Configuration

Configure voice activity detection for realtime providers:

```python
realtime = RealtimeVoiceChannel(
    "realtime-voice",
    provider=provider,
    system_prompt="...",
    vad_config={
        "threshold": 0.5,
        "silence_duration_ms": 500,
    },
)
```

## Session Resumption

Some providers support session resumption after disconnection:

```python
# OpenAI Realtime supports session resumption
provider = OpenAIRealtime(
    api_key="sk-...",
    model="gpt-4o-realtime-preview",
)
# Sessions automatically resume when possible
```

## Hooks

```python
from roomkit import HookTrigger, HookExecution

@kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC)
async def on_tool(event, ctx):
    print(f"Realtime tool call: {event}")

@kit.hook(HookTrigger.ON_REALTIME_TEXT_INJECTED, execution=HookExecution.ASYNC)
async def on_inject(event, ctx):
    print(f"Text injected: {event}")
```
