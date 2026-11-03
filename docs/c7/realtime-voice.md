# Realtime Voice (Speech-to-Speech)

RealtimeVoiceChannel connects to speech-to-speech AI models that handle audio directly — no separate STT/TTS needed. The AI model receives audio and responds with audio.

## Basic Setup

```python
from roomkit import ChannelCategory, RealtimeVoiceChannel, RoomKit

kit = RoomKit()

# Gemini Live example
from roomkit.providers.gemini import GeminiLiveProvider

realtime = RealtimeVoiceChannel(
    "realtime-voice",
    provider=GeminiLiveProvider(
        api_key="...",
        model="gemini-3.8-live",
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
| OpenAI Realtime | `OpenAIRealtimeProvider` | `roomkit[realtime-openai]` | GPT Realtime audio (reasoning-capable) |
| OpenAI GPT-Live | `OpenAILiveProvider` | `roomkit[realtime-openai]` | Full-duplex speech-to-speech; reasoning and tools delegated to a backend model, hosted or integrator-side |
| xAI Grok | `XAIRealtimeProvider` | — | Grok speech-to-speech |
| Deepgram Voice Agent | `DeepgramAgentProvider` | `roomkit[realtime-deepgram]` | Nova listen + LLM think + Aura speak, each chosen separately |
| Mock | `MockRealtimeProvider` | built-in | Testing |

```python
# OpenAI Realtime
from roomkit.providers.openai import OpenAIRealtimeProvider

provider = OpenAIRealtimeProvider(api_key="sk-...")  # defaults to gpt-realtime-2.1

# OpenAI GPT-Live (full-duplex; the model listens and speaks at once)
from roomkit.providers.openai import HostedReasoning, OpenAILiveProvider

provider = OpenAILiveProvider(
    api_key="sk-...",  # defaults to gpt-live-1
    delegation=HostedReasoning(model="gpt-5.6-terra"),  # backend runs at OpenAI
)
# IntegratorReasoning() (the default) serves delegations through
# RealtimeVoiceChannel(reasoning_backend=AIProviderReasoningBackend(...))

# xAI Grok
from roomkit.providers.xai import XAIRealtimeConfig, XAIRealtimeProvider

provider = XAIRealtimeProvider(XAIRealtimeConfig(api_key="..."))
```

## Audio Transports

Transports handle the client-side audio (browser or device):

| Transport | Class | Extra | Use Case |
|-----------|-------|-------|----------|
| WebSocket | `WebSocketRealtimeTransport` | `roomkit[websocket]` | Browser via WebSocket |
| FastRTC (WebRTC) | `FastRTCRealtimeTransport` | `roomkit[fastrtc]` | Browser via WebRTC |
| Local mic/speaker | `LocalAudioBackend` | `roomkit[local-audio]` | Local development |
| Mock | `MockRealtimeTransport` | built-in | Testing |

The WebSocket transport does not listen itself: the application serves the
sockets (here with `websockets`) and starts a session on each connection.

```python
from websockets.asyncio.server import serve

from roomkit.voice.realtime import WebSocketRealtimeTransport

realtime = RealtimeVoiceChannel(
    "realtime-voice",
    provider=provider,
    transport=WebSocketRealtimeTransport(),  # audio_format="base64_json" for JSON frames
    system_prompt="You are a helpful assistant.",
)
kit.register_channel(realtime)

async def handle_client(ws):
    room = await kit.create_room()
    await kit.attach_channel(room.id, "realtime-voice")
    await realtime.start_session(room.id, "caller", ws)
    await ws.wait_closed()  # the transport reads the socket until the client leaves

server = await serve(handle_client, "0.0.0.0", 8765)
```

`examples/realtime_voice_gemini.py` serves one, with its page and an origin
check.

A host that will not serve a WebRTC peer (no session waits for it, its call
expired) refuses it with `await transport.reject_connection(webrtc_id,
message=...)` on a `FastRTCRealtimeTransport`: the message sent as is on the
peer's data channel, the peer connection (or a websocket client's socket)
closed, the stream cleaned and the
handler unregistered, each step even when an earlier one fails. A peer the
transport's own `auth` refuses is closed the same way.

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

`join()` and `RealtimeVoiceChannel.start_session()` take `organization_id=`:
the room is read with that scope before the session exists (RFC §17.2), so
another organization's room raises `RoomNotFoundError` and no session joins
it or reaches its recordings. Left unset, `join()` reads the room unscoped and
`start_session()` does not read it.

## Tool Calling

Realtime voice channels support tool calling during conversations:

```python
from roomkit.channels import RealtimeVoiceChannel
from roomkit import ToolHandler

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

A message another channel broadcasts enters each session of the room as its author's quoted words (`@ws: “...”`), and the model answers it. An edit, a delete, a notice (a `SYSTEM` event) or a reaction enters silently: the session learns the correction, the deletion or the notice, and the model answers the next message. To have the model speak about a notice, deliver it an instruction (`kit.deliver(..., instruction=True)`).

## VAD Configuration

The provider's own turn detection is tuned per session through
`provider_config` in the session metadata (keys as the provider reads them, here
OpenAI's `server_vad`):

```python
session = await kit.join(
    "room",
    "realtime-voice",
    participant_id="user",
    metadata={
        "provider_config": {
            "turn_detection_type": "server_vad",
            "threshold": 0.5,
            "silence_duration_ms": 500,
        }
    },
)
```

A local VAD instead goes in `pipeline=AudioPipelineConfig(vad=..., vad_config=...)`.

## Session Resumption

Some providers support session resumption after disconnection:

```python
# OpenAI Realtime supports session resumption
provider = OpenAIRealtime(
    api_key="sk-...",
    model="gpt-realtime-2.1",
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
