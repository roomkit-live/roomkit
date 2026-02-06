# RoomKit

[![PyPI](https://img.shields.io/pypi/v/roomkit)](https://pypi.org/project/roomkit/)
[![Python](https://img.shields.io/pypi/pyversions/roomkit)](https://pypi.org/project/roomkit/)
[![License](https://img.shields.io/github/license/roomkit-live/roomkit)](LICENSE)

Pure async Python 3.12+ library for multi-channel conversations.

RoomKit gives you a single abstraction — the **room** — to orchestrate messages across SMS, RCS, Email, WhatsApp, Messenger, Voice, WebSocket, HTTP webhooks, and AI channels. Events flow in through any channel, get validated by hooks, and broadcast to every other channel in the room with automatic content transcoding.

```
Inbound ──► Hook pipeline ──► Store ──► Broadcast to all channels
                                             │
        ┌──────────┬──────────┬────────┬─────┼─────┬────────┬────────┬────────┐
        ▼          ▼          ▼        ▼     ▼     ▼        ▼        ▼        ▼
     SMS/RCS   WhatsApp    Email    Teams  Msgr   Voice     WS       AI    Webhook
```

**Website:** [www.roomkit.live](https://www.roomkit.live) | **Docs:** [www.roomkit.live/docs](https://www.roomkit.live/docs/)

## Quickstart

```bash
pip install roomkit
```

```python
import asyncio
from roomkit import (
    ChannelCategory, HookResult, HookTrigger,
    InboundMessage, MockAIProvider, RoomContext,
    RoomEvent, RoomKit, TextContent, WebSocketChannel,
)
from roomkit.channels.ai import AIChannel

async def main():
    kit = RoomKit()

    # Register channels
    ws = WebSocketChannel("ws-user")
    ai = AIChannel("ai-bot", provider=MockAIProvider(responses=["Hello!"]))
    kit.register_channel(ws)
    kit.register_channel(ai)

    # Create a room and attach channels
    await kit.create_room(room_id="room-1")
    await kit.attach_channel("room-1", "ws-user")
    await kit.attach_channel("room-1", "ai-bot", category=ChannelCategory.INTELLIGENCE)

    # Add a broadcast hook
    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="filter")
    async def block_spam(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if isinstance(event.content, TextContent) and "spam" in event.content.body:
            return HookResult.block("spam detected")
        return HookResult.allow()

    # Process a message — it gets stored, broadcast, and the AI responds
    result = await kit.process_inbound(
        InboundMessage(channel_id="ws-user", sender_id="user-1", content=TextContent(body="Hi"))
    )
    print(result.blocked)  # False

    # View conversation history
    for event in await kit.store.list_events("room-1"):
        print(f"[{event.source.channel_id}] {event.content.body}")

asyncio.run(main())
```

More examples in [`examples/`](examples/).

## Installation

RoomKit's core has a single dependency (pydantic). Providers that call external APIs need optional extras:

```bash
pip install roomkit                    # core only
pip install roomkit[httpx]             # HTTP-based providers (SMS, RCS, Email)
pip install roomkit[websocket]         # WebSocket event source
pip install roomkit[anthropic]         # Anthropic Claude AI
pip install roomkit[openai]            # OpenAI GPT
pip install roomkit[gemini]            # Google Gemini AI
pip install roomkit[teams]             # Microsoft Teams (Bot Framework)
pip install roomkit[neonize]           # WhatsApp Personal (neonize)
pip install roomkit[fastrtc]           # FastRTC voice backend
pip install roomkit[realtime-gemini]   # Gemini Live speech-to-speech
pip install roomkit[realtime-openai]   # OpenAI Realtime speech-to-speech
pip install roomkit[providers]         # all transport providers
pip install roomkit[all]               # everything
```

For development:

```bash
git clone https://github.com/sboily/roomkit.git
cd roomkit
uv sync --extra dev
make all                               # lint + typecheck + test
```

Requires **Python 3.12+**.

## Channels

Each channel is a thin adapter between the room and an external transport. All channels implement the same interface: `handle_inbound()` converts a provider message into a `RoomEvent`, and `deliver()` pushes events out.

| Channel | Type | Media | Notes |
|---------|------|-------|-------|
| **SMS** | `sms` | text, MMS | Max 1600 chars, delivery receipts |
| **RCS** | `rcs` | text, rich, media | Rich cards, carousels, suggested actions |
| **Email** | `email` | text, rich, media | Threading support |
| **WebSocket** | `websocket` | text, rich, media | Real-time with typing, reactions |
| **Messenger** | `messenger` | text, rich, media, template | Buttons, quick replies |
| **Teams** | `teams` | text, rich | Bot Framework SDK, proactive messaging, 28K chars |
| **WhatsApp** | `whatsapp` | text, rich, media, location, template | Buttons, templates |
| **WhatsApp Personal** | `whatsapp_personal` | text, media, audio, location | Typing indicators, read receipts, neonize |
| **Voice** | `voice` | audio, text | STT/TTS, barge-in, FastRTC streaming |
| **Realtime Voice** | `realtime_voice` | audio, text | Speech-to-speech AI (Gemini Live, OpenAI Realtime) |
| **HTTP** | `webhook` | text, rich | Generic webhook for any system |
| **AI** | `ai` | text, rich | Intelligence layer (not transport) |

Channels have two categories: **transport** (delivers to external systems) and **intelligence** (generates content, like AI). The Voice channel bridges real-time audio with the room-based conversation model.

## Providers

Providers handle the actual API calls. Every provider has a mock counterpart for testing.

### SMS Providers

| Provider | Features | Dependency |
|----------|----------|------------|
| `TwilioSMSProvider` | SMS, MMS, delivery status | `roomkit[httpx]` |
| `TelnyxSMSProvider` | SMS, MMS, delivery status | `roomkit[httpx]` |
| `SinchSMSProvider` | SMS, delivery status | `roomkit[httpx]` |
| `VoiceMeUpSMSProvider` | SMS, MMS aggregation | `roomkit[httpx]` |

### RCS Providers

| Provider | Features | Dependency |
|----------|----------|------------|
| `TwilioRCSProvider` | Rich cards, carousels, actions | `roomkit[httpx]` |
| `TelnyxRCSProvider` | Rich cards, carousels, actions | `roomkit[httpx]` |

### AI Providers

| Provider | Features | Dependency |
|----------|----------|------------|
| `AnthropicAIProvider` | Claude, vision, tools | `roomkit[anthropic]` |
| `OpenAIAIProvider` | GPT-4, vision, tools | `roomkit[openai]` |
| `GeminiAIProvider` | Gemini, vision, tools | `roomkit[gemini]` |
| `create_vllm_provider` | Local LLM, OpenAI-compatible | `roomkit[vllm]` |

### Voice Providers

| Provider | Role | Dependency |
|----------|------|------------|
| `DeepgramSTTProvider` | Speech-to-text | `roomkit[httpx]` |
| `ElevenLabsTTSProvider` | Text-to-speech | `roomkit[httpx]` |
| `SherpaOnnxSTTProvider` | Local STT (transducer/Whisper) | `roomkit[sherpa-onnx]` |
| `SherpaOnnxTTSProvider` | Local TTS (VITS/Piper) | `roomkit[sherpa-onnx]` |
| `FastRTCVoiceBackend` | WebRTC audio transport | `roomkit[fastrtc]` |

### Realtime Voice (Speech-to-Speech)

| Component | Role | Dependency |
|-----------|------|------------|
| `GeminiLiveProvider` | Speech-to-speech AI provider | `roomkit[realtime-gemini]` |
| `OpenAIRealtimeProvider` | Speech-to-speech AI provider | `roomkit[realtime-openai]` |
| `WebSocketRealtimeTransport` | Browser-to-server audio (WebSocket) | `roomkit[websocket]` |
| `FastRTCRealtimeTransport` | Browser-to-server audio (WebRTC) | `roomkit[fastrtc]` |

### Teams Providers

| Provider | Features | Dependency |
|----------|----------|------------|
| `BotFrameworkTeamsProvider` | Proactive messaging, conversation references, bot mention detection | `roomkit[teams]` |

### WhatsApp Personal Providers

| Provider | Features | Dependency |
|----------|----------|------------|
| `NeonizeWhatsAppProvider` | Multidevice, typing indicators, read receipts, media | `roomkit[neonize]` |

### Other Providers

| Provider | Channel | Dependency |
|----------|---------|------------|
| `ElasticEmailProvider` | Email | `roomkit[httpx]` |
| `FacebookMessengerProvider` | Messenger | `roomkit[httpx]` |
| `WebhookHTTPProvider` | HTTP | `roomkit[httpx]` |

Each HTTP-based provider lazy-imports `httpx` so the core library stays lightweight.

## Hooks

Hooks intercept events at specific points in the pipeline. Sync hooks can block or modify events; async hooks run after the fact for logging or side effects.

```python
@kit.hook(HookTrigger.BEFORE_BROADCAST, name="compliance_check")
async def check(event: RoomEvent, ctx: RoomContext) -> HookResult:
    # Block, allow, or modify the event
    return HookResult.allow()
```

**Triggers:** `BEFORE_BROADCAST`, `AFTER_BROADCAST`, `ON_ROOM_CREATED`, `ON_ROOM_PAUSED`, `ON_ROOM_CLOSED`, `ON_CHANNEL_ATTACHED`, `ON_CHANNEL_DETACHED`, `ON_CHANNEL_MUTED`, `ON_CHANNEL_UNMUTED`, `ON_IDENTITY_AMBIGUOUS`, `ON_IDENTITY_UNKNOWN`, `ON_PARTICIPANT_IDENTIFIED`, `ON_TASK_CREATED`, `ON_DELIVERY_STATUS`, `ON_ERROR`, `ON_SPEECH_START`, `ON_SPEECH_END`, `ON_TRANSCRIPTION`, `BEFORE_TTS`, `AFTER_TTS`.

Hooks support **filtering** by channel type, channel ID, and direction:

```python
@kit.hook(
    HookTrigger.BEFORE_BROADCAST,
    channel_types={ChannelType.SMS},
    directions={ChannelDirection.INBOUND},
)
async def sms_only_hook(event, ctx):
    return HookResult.allow()
```

Hooks can also inject side-effect events, create tasks, and record observations.

## AI Integration

### Per-Room AI Configuration

Configure AI behavior per room with custom system prompts, temperature, and tools:

```python
from roomkit import AIConfig, AITool

room = await kit.create_room(
    room_id="support-room",
    ai_config=AIConfig(
        system_prompt="You are a helpful support agent.",
        temperature=0.7,
        tools=[
            AITool(
                name="lookup_order",
                description="Look up order status",
                parameters={"type": "object", "properties": {"order_id": {"type": "string"}}}
            )
        ],
    ),
)
```

### Function Calling

AI providers support function calling with automatic tool result handling:

```python
response = await ai_provider.generate(context)
if response.tool_calls:
    for call in response.tool_calls:
        result = await execute_tool(call.name, call.arguments)
        # Feed result back to AI
```

## Realtime Events

Handle ephemeral events like typing indicators, presence, and read receipts:

```python
from roomkit import EphemeralEvent, EphemeralEventType

# Subscribe to realtime events
async def handle_realtime(event: EphemeralEvent):
    if event.type == EphemeralEventType.TYPING_START:
        print(f"{event.user_id} is typing...")

sub_id = await kit.subscribe_room("room-1", handle_realtime)

# Publish typing indicator
await kit.publish_typing("room-1", "user-1")

# Publish presence
await kit.publish_presence("room-1", "user-1", "online")

# Publish read receipt
await kit.publish_read_receipt("room-1", "user-1", "event-123")
```

For distributed deployments, implement a custom `RealtimeBackend` (e.g., Redis pub/sub).

## Identity Resolution

Resolve unknown senders to known identities with a pluggable pipeline:

```python
from roomkit import IdentityResolver, IdentityResult, IdentificationStatus, Identity

class MyResolver(IdentityResolver):
    async def resolve(self, message, context):
        user = await lookup(message.sender_id)
        if user:
            return IdentityResult(
                status=IdentificationStatus.IDENTIFIED,
                identity=Identity(id=user.id, display_name=user.name),
            )
        return IdentityResult(status=IdentificationStatus.UNKNOWN)

kit = RoomKit(identity_resolver=MyResolver())
```

### Channel Type Filtering

Restrict identity resolution to specific channel types:

```python
kit = RoomKit(
    identity_resolver=MyResolver(),
    identity_channel_types={ChannelType.SMS},  # Only resolve for SMS
)
```

Supports identified, pending, ambiguous, challenge, and rejected outcomes with hook-based customization.

## Webhook Processing

Process provider webhooks with automatic parsing and delivery status tracking:

```python
# Generic webhook processing
result = await kit.process_webhook(
    channel_id="sms-channel",
    raw_payload=request_body,
    headers=request_headers,
)

# Handle delivery status updates
@kit.on_delivery_status
async def handle_status(status: DeliveryStatus):
    print(f"Message {status.provider_message_id}: {status.status}")
```

## Event-Driven Sources

For persistent connections (WebSocket, NATS, SSE), use **SourceProviders** instead of webhooks:

```python
from roomkit import RoomKit, BaseSourceProvider, SourceStatus
from roomkit.sources import WebSocketSource

kit = RoomKit()

# Built-in WebSocket source
source = WebSocketSource(
    url="wss://chat.example.com/events",
    channel_id="websocket-chat",
)

# Attach with resilience options
await kit.attach_source(
    "websocket-chat",
    source,
    auto_restart=True,           # Restart on failure
    max_restart_attempts=10,     # Give up after 10 failures
    max_concurrent_emits=20,     # Backpressure control
)

# Monitor health
health = await kit.source_health("websocket-chat")
print(f"Status: {health.status}, Messages: {health.messages_received}")

# Detach when done
await kit.detach_source("websocket-chat")
```

**Webhook vs Event-Driven:**

| Aspect | Webhooks | Event Sources |
|--------|----------|---------------|
| Connection | Stateless HTTP | Persistent (WS, TCP, etc.) |
| Initiative | External system pushes | RoomKit subscribes |
| Use cases | Twilio, SendGrid | WebSocket, NATS, SSE |

Create custom sources by extending `BaseSourceProvider`:

```python
class NATSSource(BaseSourceProvider):
    @property
    def name(self) -> str:
        return "nats:events"

    async def start(self, emit) -> None:
        self._set_status(SourceStatus.CONNECTED)
        async for msg in self.subscribe():
            await emit(parse_message(msg))
            self._record_message()
```

Features: exponential backoff, max restart attempts, backpressure control, health monitoring.

## Resilience

Built-in patterns for production reliability:

- **Retry with backoff** — configurable per-channel retry policy with exponential backoff
- **Circuit breaker** — isolates failing providers so one broken channel doesn't bring down the room
- **Rate limiting** — token bucket limiter with per-second/minute/hour limits per channel
- **Content transcoding** — automatic conversion between channel capabilities (e.g. rich to text fallback)
- **Chain depth tracking** — prevents infinite event loops between channels

Configure per binding:

```python
from roomkit import RetryPolicy, RateLimit

await kit.attach_channel("room-1", "sms-out",
    metadata={"phone_number": "+15551234567"},
    retry_policy=RetryPolicy(max_retries=3, base_delay_seconds=1.0),
    rate_limit=RateLimit(max_per_second=5.0),
)
```

## Scaling

### Per-room locking

RoomKit serializes event processing per room using a `RoomLockManager`. The default `InMemoryLockManager` works for single-process deployments. For multi-process or distributed setups, subclass `RoomLockManager` with a distributed lock (Redis, Postgres advisory locks, etc.):

```python
from roomkit import RoomKit, RoomLockManager

# Single process (default)
kit = RoomKit()

# Distributed
kit = RoomKit(lock_manager=MyRedisLockManager())
```

### Backpressure

RoomKit does not enforce a global concurrency limit on `process_inbound` calls. Each call acquires a per-room lock internally, but across different rooms, processing is fully concurrent.

To prevent resource exhaustion, add concurrency control upstream:

```python
import asyncio
from roomkit import RoomKit, InboundMessage, InboundResult

kit = RoomKit()
semaphore = asyncio.Semaphore(100)  # max 100 concurrent rooms processing

async def handle_webhook(message: InboundMessage) -> InboundResult:
    async with semaphore:
        return await kit.process_inbound(message)
```

## Room Lifecycle

Rooms transition through states automatically based on activity timers:

```
ACTIVE ──(inactive timeout)──► PAUSED ──(closed timeout)──► CLOSED
```

```python
from roomkit import RoomTimers

await kit.create_room(
    room_id="support-123",
    timers=RoomTimers(inactive_after_seconds=300, closed_after_seconds=3600),
)

# Check and apply timer transitions
transitioned = await kit.sweep_room_timers()
```

## Storage

The `ConversationStore` ABC defines the persistence interface. `InMemoryStore` is included for development and testing. Implement the ABC to use any database.

```python
kit = RoomKit()                          # uses InMemoryStore by default
kit = RoomKit(store=MyPostgresStore())   # plug in your own
```

The store handles rooms, events, bindings, participants, identities, tasks, and observations.

## AI Assistant Support

RoomKit includes files to help AI coding assistants understand the library:

- **[llms.txt](https://www.roomkit.live/llms.txt)** — Structured documentation for LLM context windows
- **[AGENTS.md](AGENTS.md)** — Coding guidelines and patterns for AI assistants
- **[MCP Integration](https://www.roomkit.live/docs/mcp/)** — Model Context Protocol support

Access programmatically:

```python
from roomkit import get_llms_txt, get_agents_md

llms_content = get_llms_txt()
agents_content = get_agents_md()
```

## Project Structure

```
src/roomkit/
  channels/        Channel implementations (sms, rcs, email, websocket, ai, ...)
  core/            Framework, hooks, routing, retry, circuit breaker
  identity/        Identity resolution pipeline
  models/          Pydantic data models and enums
  providers/       Provider implementations (sms/, rcs/, email/, teams/, messenger/, ...)
  realtime/        Ephemeral events (typing, presence, read receipts)
  sources/         Event-driven sources (WebSocket, neonize, custom)
  store/           Storage abstraction and in-memory implementation
  voice/           Voice subsystem (stt/, tts/, backends/, realtime/)
```

## Documentation

- **[Website](https://www.roomkit.live)** — Landing page and overview
- **[Documentation](https://www.roomkit.live/docs/)** — Full documentation
- **[API Reference](https://www.roomkit.live/docs/api/)** — Complete API docs
- **[RFC](https://www.roomkit.live/docs/roomkit-rfc/)** — Design document

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Quick version:

```bash
uv sync --extra dev
make all                # ruff check + mypy --strict + pytest
```

All new code needs tests. Aim for >90% coverage.

## License

[MIT](LICENSE)
