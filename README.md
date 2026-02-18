# RoomKit

[![PyPI](https://img.shields.io/pypi/v/roomkit)](https://pypi.org/project/roomkit/)
[![Python](https://img.shields.io/pypi/pyversions/roomkit)](https://pypi.org/project/roomkit/)
[![License](https://img.shields.io/github/license/roomkit-live/roomkit)](LICENSE)

Pure async Python 3.12+ framework for multi-channel conversation orchestration.

RoomKit gives you a single abstraction — the **room** — to orchestrate conversations across SMS, RCS, Email, WhatsApp, Messenger, Teams, Telegram, Voice, WebSocket, HTTP, and AI channels. Define agents, wire them into pipelines, and let the framework handle routing, handoffs, audio processing, and content transcoding.

```
                                ┌──────────────────────┐
                                │        Room          │
                                │                      │
Inbound ──► Hook pipeline ──►   │  ConversationRouter   │  ──► Broadcast to all channels
                                │  ConversationState    │
                                │  HandoffHandler       │        │
                                └──────────────────────┘   ┌────┴────┐
                                                           │         │
        ┌──────────┬──────────┬────────┬─────┬─────┬───────┤         ├────────┐
        ▼          ▼          ▼        ▼     ▼     ▼       ▼         ▼        ▼
     SMS/RCS   WhatsApp    Email    Teams  Msgr   Voice  Realtime    WS    Agents
                                                  (STT/   Voice            (AI)
                                                   TTS)  (S2S AI)
```

**Website:** [www.roomkit.live](https://www.roomkit.live) | **Docs:** [www.roomkit.live/docs](https://www.roomkit.live/docs/)

## Quickstart

```bash
pip install roomkit
```

```python
import asyncio
from roomkit import (
    Agent, ChannelCategory, ConversationPipeline, ConversationState,
    HandoffMemoryProvider, InboundMessage, MockAIProvider, PipelineStage,
    RoomKit, SlidingWindowMemory, TextContent, WebSocketChannel,
    set_conversation_state,
)

async def main():
    kit = RoomKit()

    # Transport channel for user messages
    ws = WebSocketChannel("ws-user")
    kit.register_channel(ws)

    # AI agents with identity metadata
    triage = Agent(
        "agent-triage",
        provider=MockAIProvider(responses=["I'll transfer you to our specialist."]),
        role="Triage agent",
        description="Routes incoming requests",
        system_prompt="You triage incoming requests.",
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=50)),
    )
    handler = Agent(
        "agent-handler",
        provider=MockAIProvider(responses=["Let me help you with that."]),
        role="Support specialist",
        description="Handles customer requests",
        system_prompt="You handle requests.",
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=50)),
    )
    kit.register_channel(triage)
    kit.register_channel(handler)

    # Define a pipeline: triage -> handling
    pipeline = ConversationPipeline(stages=[
        PipelineStage(phase="triage", agent_id="agent-triage", next="handling"),
        PipelineStage(phase="handling", agent_id="agent-handler", next=None),
    ])
    router, handoff = pipeline.install(kit, [triage, handler])

    # Create room and attach channels
    await kit.create_room(room_id="room-1")
    await kit.attach_channel("room-1", "ws-user")
    await kit.attach_channel("room-1", "agent-triage", category=ChannelCategory.INTELLIGENCE)
    await kit.attach_channel("room-1", "agent-handler", category=ChannelCategory.INTELLIGENCE)

    # Initialize conversation state
    room = await kit.get_room("room-1")
    room = set_conversation_state(room, ConversationState(phase="triage", active_agent_id="agent-triage"))
    await kit.store.update_room(room)

    # Process a message — routed to the active agent
    await kit.process_inbound(
        InboundMessage(channel_id="ws-user", sender_id="user-1", content=TextContent(body="I need help"))
    )

    # Handoff to the next agent
    result = await handoff.handle(
        room_id="room-1",
        calling_agent_id="agent-triage",
        arguments={"target": "agent-handler", "reason": "User needs support"},
    )
    print(f"Handoff accepted: {result.accepted}")

asyncio.run(main())
```

More examples in [`examples/`](examples/).

## Installation

RoomKit's core has a single dependency (pydantic). Providers that call external APIs need optional extras:

```bash
pip install roomkit                    # core only

# AI providers
pip install roomkit[anthropic]         # Anthropic Claude
pip install roomkit[openai]            # OpenAI GPT
pip install roomkit[gemini]            # Google Gemini
pip install roomkit[pydantic-ai]       # Pydantic AI

# Voice backends
pip install roomkit[fastrtc]           # FastRTC WebRTC backend
pip install roomkit[sip]               # SIP voice backend
pip install roomkit[rtp]               # RTP voice backend
pip install roomkit[local-audio]       # Local mic/speaker backend

# Speech-to-speech AI
pip install roomkit[realtime-gemini]   # Gemini Live
pip install roomkit[realtime-openai]   # OpenAI Realtime

# STT / TTS providers
pip install roomkit[deepgram]          # Deepgram STT
pip install roomkit[elevenlabs]        # ElevenLabs TTS
pip install roomkit[sherpa-onnx]       # SherpaOnnx (local STT/TTS/VAD/Denoiser)
pip install roomkit[gradium]           # Gradium STT + TTS

# Communication providers
pip install roomkit[httpx]             # HTTP-based providers (SMS, RCS, Email)
pip install roomkit[websocket]         # WebSocket event source
pip install roomkit[teams]             # Microsoft Teams
pip install roomkit[telegram]          # Telegram
pip install roomkit[neonize]           # WhatsApp Personal

# Infrastructure
pip install roomkit[postgres]          # PostgreSQL storage
pip install roomkit[opentelemetry]     # OpenTelemetry tracing
pip install roomkit[mcp]               # Model Context Protocol tools

# Meta extras
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

## Multi-Agent Orchestration

RoomKit's orchestration layer lets you define agents, route conversations through pipelines, and hand off between agents — including mid-call voice handoffs where the caller never hears a disconnect.

### Agent

`Agent` extends `AIChannel` with identity metadata that gets auto-injected into the system prompt:

```python
from roomkit import Agent, GeminiAIProvider, GeminiConfig

triage = Agent(
    "agent-triage",
    provider=GeminiAIProvider(config),
    role="Triage receptionist",
    description="Routes callers to the right specialist",
    scope="Financial advisory services only",
    voice="21m00Tcm4TlvDq8ikWAM",     # TTS voice ID
    greeting="Greet the caller warmly and ask how you can help.",
    language="French",                  # language hint injected into prompt
    system_prompt="You triage incoming requests.",
    memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=20)),
)
```

For speech-to-speech orchestration (Gemini Live, OpenAI Realtime), agents can be **config-only** — no AI provider needed since the realtime provider handles reasoning:

```python
triage = Agent(
    "agent-triage",
    role="Triage receptionist",
    voice="Zephyr",
    system_prompt="Greet callers warmly.",
)
```

### ConversationPipeline

Define a multi-stage pipeline and install it with one call:

```python
from roomkit import ConversationPipeline, PipelineStage

pipeline = ConversationPipeline(stages=[
    PipelineStage(phase="intake", agent_id="agent-triage", next="handling"),
    PipelineStage(phase="handling", agent_id="agent-advisor", next=None),
])

# Wires routing hook + handoff tool on all agents
router, handler = pipeline.install(
    kit, [triage, advisor],
    greet_on_handoff=True,         # send agent greeting on handoff
    voice_channel_id="voice",      # voice channel for spoken greetings
)
```

`install()` auto-wires:
- A `ConversationRouter` (BEFORE_BROADCAST hook) that routes messages to the active agent
- A `HandoffHandler` with the `handoff_conversation` tool registered on each agent
- Voice map from each agent's `voice` field for per-agent TTS voices
- Greeting injection on handoff (spoken via TTS or injected into realtime sessions)
- System event filtering to keep orchestration internals out of transcripts

### Voice Orchestration

Agent handoffs work seamlessly on live voice calls. The voice/realtime channel is a transport — swapping the active agent doesn't touch the audio session:

```
SIP Call → VoiceChannel (STT) → transcript → ConversationRouter → Active Agent
                                                                       │
                                                                  text response
                                                                       │
                                              VoiceChannel (TTS) ← ────┘
```

For speech-to-speech mode, the realtime session is reconfigured on handoff (system prompt, voice, tools) using session resumption — the conversation context is preserved with ~200-500ms latency:

```
SIP Call → RealtimeVoiceChannel → GeminiLiveProvider
                                        │
                                  handoff_conversation tool call
                                        │
                                  reconfigure_session() (voice, prompt, tools)
```

### Conversation State

Track conversation phase, active agent, handoff history, and custom context per room:

```python
from roomkit import ConversationState, get_conversation_state, set_conversation_state

state = get_conversation_state(room)
print(state.phase, state.active_agent_id, state.handoff_count)

for t in state.phase_history:
    print(f"{t.from_phase} -> {t.to_phase} ({t.reason})")
```

### Language-Aware Handoffs

Set a per-room language that gets injected into agent prompts and realtime sessions:

```python
await handler.set_language("room-1", "French", channel_id="voice")
```

## Channels

Each channel is a thin adapter between the room and an external transport. All channels implement the same interface: `handle_inbound()` converts a provider message into a `RoomEvent`, and `deliver()` pushes events out.

| Channel | Type | Media | Notes |
|---------|------|-------|-------|
| **SMS** | `sms` | text, MMS | Max 1600 chars, delivery receipts |
| **RCS** | `rcs` | text, rich, media | Rich cards, carousels, suggested actions |
| **Email** | `email` | text, rich, media | Threading support |
| **WebSocket** | `websocket` | text, rich, media | Real-time with typing, reactions |
| **Messenger** | `messenger` | text, rich, media, template | Buttons, quick replies |
| **Teams** | `teams` | text, rich | Bot Framework SDK, proactive messaging |
| **Telegram** | `telegram` | text, rich, media | Bot API |
| **WhatsApp** | `whatsapp` | text, rich, media, location, template | Buttons, templates |
| **WhatsApp Personal** | `whatsapp_personal` | text, media, audio, location | Typing indicators, read receipts |
| **Voice** | `voice` | audio, text | STT/TTS, audio pipeline, barge-in |
| **Realtime Voice** | `realtime_voice` | audio, text | Speech-to-speech AI (Gemini Live, OpenAI Realtime) |
| **HTTP** | `webhook` | text, rich | Generic webhook for any system |
| **AI / Agent** | `ai` | text, rich | Intelligence layer (not transport) |

Channels have two categories: **transport** (delivers to external systems) and **intelligence** (generates content, like AI). The Voice and Realtime Voice channels bridge real-time audio with the room-based conversation model.

## Providers

Providers handle the actual API calls. Every provider has a mock counterpart for testing.

### AI Providers

| Provider | Features | Dependency |
|----------|----------|------------|
| `AnthropicAIProvider` | Claude, vision, tools, streaming | `roomkit[anthropic]` |
| `OpenAIAIProvider` | GPT-4, vision, tools, streaming | `roomkit[openai]` |
| `GeminiAIProvider` | Gemini, vision, tools, streaming | `roomkit[gemini]` |
| `PydanticAIProvider` | Pydantic AI agent integration | `roomkit[pydantic-ai]` |
| `create_vllm_provider` | Local LLM, OpenAI-compatible | `roomkit[vllm]` |

### Voice Backends

| Backend | Role | Dependency |
|---------|------|------------|
| `FastRTCVoiceBackend` | WebRTC audio transport | `roomkit[fastrtc]` |
| `SIPVoiceBackend` | SIP/RTP voice transport | `roomkit[sip]` |
| `RTPBackend` | Raw RTP audio transport | `roomkit[rtp]` |
| `LocalAudioBackend` | Local mic/speaker for testing | `roomkit[local-audio]` |

### STT Providers

| Provider | Features | Dependency |
|----------|----------|------------|
| `DeepgramSTTProvider` | Cloud streaming + batch | `roomkit[deepgram]` |
| `SherpaOnnxSTTProvider` | Local ONNX (transducer/Whisper) | `roomkit[sherpa-onnx]` |
| `QwenSTTProvider` | Qwen ASR | `roomkit[qwen-asr]` |
| `GradiumSTTProvider` | Gradium | `roomkit[gradium]` |

### TTS Providers

| Provider | Features | Dependency |
|----------|----------|------------|
| `ElevenLabsTTSProvider` | Cloud streaming, low latency | `roomkit[elevenlabs]` |
| `SherpaOnnxTTSProvider` | Local ONNX (VITS/Piper) | `roomkit[sherpa-onnx]` |
| `QwenTTSProvider` | Qwen TTS | `roomkit[qwen-tts]` |
| `GradiumTTSProvider` | Gradium | `roomkit[gradium]` |
| `NeuTTSProvider` | Neural TTS | built-in |

### Realtime Voice (Speech-to-Speech)

| Component | Role | Dependency |
|-----------|------|------------|
| `GeminiLiveProvider` | Speech-to-speech AI | `roomkit[realtime-gemini]` |
| `OpenAIRealtimeProvider` | Speech-to-speech AI | `roomkit[realtime-openai]` |
| `WebSocketRealtimeTransport` | Browser-to-server audio (WS) | `roomkit[websocket]` |
| `FastRTCRealtimeTransport` | Browser-to-server audio (WebRTC) | `roomkit[fastrtc]` |
| `SIPRealtimeTransport` | SIP audio to realtime AI | `roomkit[sip]` |

### SMS Providers

| Provider | Features | Dependency |
|----------|----------|------------|
| `TwilioSMSProvider` | SMS, MMS, delivery status | `roomkit[httpx]` |
| `TelnyxSMSProvider` | SMS, MMS, delivery status | `roomkit[httpx]` |
| `SinchSMSProvider` | SMS, delivery status | `roomkit[httpx]` |
| `VoiceMeUpSMSProvider` | SMS, MMS aggregation | `roomkit[httpx]` |

### Other Providers

| Provider | Channel | Dependency |
|----------|---------|------------|
| `TwilioRCSProvider` | RCS | `roomkit[httpx]` |
| `TelnyxRCSProvider` | RCS | `roomkit[httpx]` |
| `ElasticEmailProvider` | Email | `roomkit[httpx]` |
| `SendGridProvider` | Email | `roomkit[httpx]` |
| `FacebookMessengerProvider` | Messenger | `roomkit[httpx]` |
| `TelegramBotProvider` | Telegram | `roomkit[telegram]` |
| `BotFrameworkTeamsProvider` | Teams | `roomkit[teams]` |
| `NeonizeWhatsAppProvider` | WhatsApp Personal | `roomkit[neonize]` |
| `WebhookHTTPProvider` | HTTP | `roomkit[httpx]` |

Each HTTP-based provider lazy-imports `httpx` so the core library stays lightweight.

## Audio Pipeline

The audio pipeline sits between the voice backend and STT, processing raw audio through pluggable inbound and outbound chains:

```
Inbound:   Backend → [Resampler] → [Recorder] → [AEC] → [AGC] → [Denoiser] → VAD → [Diarization] + [DTMF]
Outbound:  TTS → [PostProcessors] → [Recorder] → AEC.feed_reference → [Resampler] → Backend
```

AEC and AGC stages are automatically skipped when the backend declares `NATIVE_AEC` / `NATIVE_AGC` capabilities.

| Stage | Role | Implementations |
|-------|------|-----------------|
| `VADProvider` | Voice activity detection | `SherpaOnnxVADProvider`, `EnergyVADProvider` |
| `DenoiserProvider` | Noise reduction | `RNNoiseDenoiserProvider`, `SherpaOnnxDenoiserProvider` |
| `AECProvider` | Acoustic echo cancellation | `SpeexAECProvider` |
| `AGCProvider` | Automatic gain control | configurable |
| `DiarizationProvider` | Speaker identification | pluggable |
| `DTMFDetector` | DTMF tone detection (parallel) | pluggable |
| `AudioRecorder` | Record inbound/outbound audio | `WavFileRecorder` |
| `AudioPostProcessor` | Custom outbound transforms | pluggable |
| `TurnDetector` | Post-STT turn completion | pluggable |
| `BackchannelDetector` | Distinguish interruptions from "uh-huh" | pluggable |

All stages are optional — configure what you need:

```python
from roomkit import VoiceChannel
from roomkit.voice.pipeline import AudioPipelineConfig, VADConfig
from roomkit.voice.interruption import InterruptionConfig, InterruptionStrategy

pipeline = AudioPipelineConfig(
    vad=my_vad, denoiser=my_denoiser, aec=my_aec,
    agc=my_agc, diarization=my_diarizer, dtmf=my_dtmf,
    recorder=my_recorder, turn_detector=my_turn_detector,
    vad_config=VADConfig(silence_threshold_ms=500),
)
voice = VoiceChannel(
    "voice", stt=stt, tts=tts, backend=backend,
    pipeline=pipeline,
    interruption=InterruptionConfig(strategy=InterruptionStrategy.CONFIRMED, min_speech_ms=300),
)
```

**Interruption strategies** control how user speech during TTS playback is handled: `IMMEDIATE` (interrupt on any speech), `CONFIRMED` (wait for sustained speech), `SEMANTIC` (use backchannel detection to ignore "uh-huh"), `DISABLED` (ignore speech during playback).

## Hooks

Hooks intercept events at specific points in the pipeline. Sync hooks can block or modify events; async hooks run after the fact for logging or side effects.

```python
@kit.hook(HookTrigger.BEFORE_BROADCAST, name="compliance_check")
async def check(event: RoomEvent, ctx: RoomContext) -> HookResult:
    return HookResult.allow()
```

**35 hook triggers** covering the full lifecycle:

| Category | Triggers |
|----------|----------|
| Event pipeline | `BEFORE_BROADCAST`, `AFTER_BROADCAST` |
| Room lifecycle | `ON_ROOM_CREATED`, `ON_ROOM_PAUSED`, `ON_ROOM_CLOSED` |
| Channel lifecycle | `ON_CHANNEL_ATTACHED`, `ON_CHANNEL_DETACHED`, `ON_CHANNEL_MUTED`, `ON_CHANNEL_UNMUTED` |
| Identity | `ON_IDENTITY_AMBIGUOUS`, `ON_IDENTITY_UNKNOWN`, `ON_PARTICIPANT_IDENTIFIED` |
| Voice (audio) | `ON_SPEECH_START`, `ON_SPEECH_END`, `ON_TRANSCRIPTION`, `ON_PARTIAL_TRANSCRIPTION`, `ON_VAD_SILENCE`, `ON_VAD_AUDIO_LEVEL`, `ON_INPUT_AUDIO_LEVEL`, `ON_OUTPUT_AUDIO_LEVEL` |
| Voice (TTS) | `BEFORE_TTS`, `AFTER_TTS`, `ON_TTS_CANCELLED`, `ON_BARGE_IN` |
| Voice (pipeline) | `ON_SPEAKER_CHANGE`, `ON_DTMF`, `ON_TURN_COMPLETE`, `ON_TURN_INCOMPLETE`, `ON_BACKCHANNEL`, `ON_RECORDING_STARTED`, `ON_RECORDING_STOPPED` |
| Realtime voice | `ON_REALTIME_TOOL_CALL`, `ON_REALTIME_TEXT_INJECTED` |
| Orchestration | `ON_PHASE_TRANSITION`, `ON_HANDOFF`, `ON_HANDOFF_REJECTED` |
| Side effects | `ON_TASK_CREATED`, `ON_DELIVERY_STATUS`, `ON_ERROR`, `ON_OBSERVATION`, `ON_PROTOCOL_TRACE` |

Hooks support filtering by channel type, channel ID, and direction:

```python
@kit.hook(HookTrigger.BEFORE_BROADCAST, channel_types={ChannelType.SMS}, directions={ChannelDirection.INBOUND})
async def sms_only_hook(event, ctx):
    return HookResult.allow()
```

## AI Integration

### Per-Room AI Configuration

```python
from roomkit import AIConfig, AITool

room = await kit.create_room(
    room_id="support-room",
    ai_config=AIConfig(
        system_prompt="You are a helpful support agent.",
        temperature=0.7,
        tools=[AITool(name="lookup_order", description="Look up order status", parameters={...})],
    ),
)
```

### Memory Providers

Control what context the AI sees with pluggable memory:

```python
from roomkit import SlidingWindowMemory, HandoffMemoryProvider

# Recent N messages
memory = SlidingWindowMemory(max_events=50)

# Handoff-aware: injects handoff context (reason, summary) from previous agent
memory = HandoffMemoryProvider(SlidingWindowMemory(max_events=50))

agent = Agent("agent", provider=my_provider, memory=memory, system_prompt="...")
```

### Function Calling

AI providers support function calling with automatic tool result handling:

```python
response = await ai_provider.generate(context)
if response.tool_calls:
    for call in response.tool_calls:
        result = await execute_tool(call.name, call.arguments)
```

### MCP Tools

Integrate Model Context Protocol tools into AI agents:

```python
from roomkit import MCPToolProvider, compose_tool_handlers

mcp = MCPToolProvider(server_url="http://localhost:3000")
handler = compose_tool_handlers(mcp.handler, my_custom_handler)
```

## Realtime Events

Handle ephemeral events like typing indicators, presence, and read receipts:

```python
from roomkit import EphemeralEvent, EphemeralEventType

async def handle_realtime(event: EphemeralEvent):
    if event.type == EphemeralEventType.TYPING_START:
        print(f"{event.user_id} is typing...")

sub_id = await kit.subscribe_room("room-1", handle_realtime)
await kit.publish_typing("room-1", "user-1")
await kit.publish_presence("room-1", "user-1", "online")
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
            return IdentityResult(status=IdentificationStatus.IDENTIFIED, identity=Identity(id=user.id, display_name=user.name))
        return IdentityResult(status=IdentificationStatus.UNKNOWN)

kit = RoomKit(identity_resolver=MyResolver())
```

## Event-Driven Sources

For persistent connections (WebSocket, NATS, SSE), use **SourceProviders** instead of webhooks:

```python
from roomkit.sources import WebSocketSource

source = WebSocketSource(url="wss://chat.example.com/events", channel_id="websocket-chat")
await kit.attach_source("websocket-chat", source, auto_restart=True, max_restart_attempts=10)

health = await kit.source_health("websocket-chat")
print(f"Status: {health.status}, Messages: {health.messages_received}")
```

## Resilience

Built-in patterns for production reliability:

- **Retry with backoff** — configurable per-channel retry policy with exponential backoff
- **Circuit breaker** — isolates failing providers so one broken channel doesn't bring down the room
- **Rate limiting** — token bucket limiter with per-second/minute/hour limits per channel
- **Content transcoding** — automatic conversion between channel capabilities (rich to text fallback)
- **Chain depth tracking** — prevents infinite event loops between channels (default max=5)
- **Idempotency** — idempotency keys prevent duplicate event processing

```python
from roomkit import RetryPolicy, RateLimit

await kit.attach_channel("room-1", "sms-out",
    retry_policy=RetryPolicy(max_retries=3, base_delay_seconds=1.0),
    rate_limit=RateLimit(max_per_second=5.0),
)
```

## Scaling

### Per-room locking

RoomKit serializes event processing per room using a `RoomLockManager`. The default `InMemoryLockManager` works for single-process deployments. For distributed setups, subclass with a distributed lock:

```python
kit = RoomKit()                                    # default in-memory
kit = RoomKit(lock_manager=MyRedisLockManager())   # distributed
```

### Room Lifecycle

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
transitioned = await kit.sweep_room_timers()
```

## Storage

The `ConversationStore` ABC defines the persistence interface. `InMemoryStore` is included for development; `PostgresStore` for production.

```python
kit = RoomKit()                          # uses InMemoryStore by default
kit = RoomKit(store=PostgresStore(...))   # PostgreSQL
```

The store handles rooms, events, bindings, participants, identities, tasks, and observations.

## Telemetry

Built-in tracing with pluggable backends:

```python
from roomkit import TelemetryConfig, ConsoleTelemetryProvider

kit = RoomKit(telemetry=TelemetryConfig(provider=ConsoleTelemetryProvider()))

# Or use OpenTelemetry for production
from roomkit import OpenTelemetryProvider
kit = RoomKit(telemetry=TelemetryConfig(provider=OpenTelemetryProvider()))
```

## Skills

Extensible AI capabilities via a skill registry:

```python
from roomkit import Skill, SkillMetadata, SkillRegistry

registry = SkillRegistry()
registry.register(Skill(
    metadata=SkillMetadata(name="weather", description="Get weather forecasts"),
    handler=my_weather_handler,
))
```

## AI Assistant Support

RoomKit includes files to help AI coding assistants understand the library:

- **[llms.txt](https://www.roomkit.live/llms.txt)** — structured documentation for LLM context windows
- **[AGENTS.md](AGENTS.md)** — coding guidelines and patterns for AI assistants
- **[MCP Integration](https://www.roomkit.live/docs/mcp/)** — Model Context Protocol support

```python
from roomkit import get_llms_txt, get_agents_md, get_ai_context

llms_content = get_llms_txt()
agents_content = get_agents_md()
combined = get_ai_context()
```

## Project Structure

```
src/roomkit/
  core/            Framework, hooks, routing, retry, circuit breaker
  channels/        Channel implementations (Voice, AI, Agent, WebSocket, etc.)
  orchestration/   Multi-agent routing, handoff, pipeline, conversation state
  providers/       Provider implementations (AI, SMS, Email, Teams, etc.)
  voice/           Voice subsystem
    backends/        Audio transports (FastRTC, RTP, SIP, Local)
    stt/             Speech-to-text (Deepgram, SherpaOnnx, Qwen, Gradium)
    tts/             Text-to-speech (ElevenLabs, SherpaOnnx, Qwen, Gradium, Neu)
    pipeline/        10 audio processing stages (VAD, AEC, AGC, Denoiser, etc.)
    realtime/        Speech-to-speech (Gemini Live, OpenAI Realtime)
  models/          Pydantic data models and enums
  memory/          AI context construction (SlidingWindow, Handoff-aware)
  skills/          Extensible AI capabilities
  tools/           MCP tool integration
  store/           Conversation persistence (Memory, Postgres)
  identity/        User identification resolution
  realtime/        Ephemeral events (typing, presence, reactions)
  sources/         Event-driven sources (WebSocket, SSE, Neonize)
  telemetry/       Tracing (Console, OpenTelemetry)
```

## Documentation

- **[Website](https://www.roomkit.live)** — landing page and overview
- **[Documentation](https://www.roomkit.live/docs/)** — full documentation
- **[API Reference](https://www.roomkit.live/docs/api/)** — complete API docs
- **[RFC](https://www.roomkit.live/docs/roomkit-rfc/)** — design document

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Quick version:

```bash
uv sync --extra dev
make all                # ruff check + mypy --strict + pytest
```

All new code needs tests. Aim for >90% coverage.

## License

[MIT](LICENSE)
