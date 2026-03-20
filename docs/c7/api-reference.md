# API Reference

RoomKit exports **71 symbols** from `roomkit`. Providers and voice types import from subpackages.

## Top-Level Imports (`from roomkit import ...`)

### Framework

| Symbol | Description |
|--------|-------------|
| `RoomKit` | Central orchestrator — rooms, channels, hooks, storage |

### Channels

| Symbol | Description |
|--------|-------------|
| `Agent` | AI agent with role, description, greeting, tools |
| `AIChannel` | Intelligence layer for AI responses |
| `AudioVideoChannel` | Combined audio + video channel |
| `Channel` | Base class for all channels |
| `EmailChannel` | Email transport channel factory |
| `HTTPChannel` | HTTP webhook transport channel factory |
| `MessengerChannel` | Facebook Messenger transport channel factory |
| `RCSChannel` | RCS transport channel factory |
| `RealtimeAudioVideoChannel` | Realtime speech-to-speech with video |
| `RealtimeVoiceChannel` | Speech-to-speech AI channel |
| `SMSChannel` | SMS transport channel factory |
| `TeamsChannel` | Microsoft Teams transport channel factory |
| `TelegramChannel` | Telegram Bot transport channel factory |
| `TransportChannel` | Generic transport channel wrapper |
| `VideoChannel` | Video channel with vision pipeline |
| `VoiceChannel` | Real-time audio with STT/TTS/pipeline |
| `WebSocketChannel` | WebSocket bidirectional channel |
| `WhatsAppChannel` | WhatsApp Business API channel factory |
| `WhatsAppPersonalChannel` | WhatsApp Personal (neonize) channel factory |

### Enums

| Symbol | Description |
|--------|-------------|
| `Access` | Channel access levels: READ_WRITE, READ_ONLY, WRITE_ONLY, NONE |
| `ChannelCategory` | TRANSPORT or INTELLIGENCE |
| `ChannelType` | SMS, EMAIL, WHATSAPP, VOICE, AI, WEBSOCKET, etc. |
| `EventStatus` | DELIVERED, BLOCKED, etc. |
| `EventType` | MESSAGE, SYSTEM, EDIT, DELETE, etc. |
| `HookExecution` | SYNC or ASYNC |
| `HookTrigger` | 40+ hook triggers (BEFORE_BROADCAST, AFTER_BROADCAST, etc.) |
| `RoomStatus` | ACTIVE, PAUSED, CLOSED, ARCHIVED |

### Orchestration

| Symbol | Description |
|--------|-------------|
| `Loop` | Producer/reviewer cycle strategy |
| `Orchestration` | ABC for orchestration strategies |
| `Pipeline` | Linear agent chain strategy |
| `Supervisor` | Supervisor delegates to workers strategy |
| `Swarm` | Bidirectional handoff strategy |

### Models

| Symbol | Description |
|--------|-------------|
| `ChannelBinding` | Binding of a channel to a room |
| `ChannelCapabilities` | Declared capabilities of a channel |
| `ChannelOutput` | Output of a channel delivery |
| `DeliveryResult` | Result of delivering a message |
| `DeliveryStatus` | Delivery status from provider webhook |
| `EventSource` | Source attribution for an event |
| `FrameworkEvent` | Lightweight framework lifecycle event |
| `HookResult` | Result from sync hooks: `.allow()`, `.block(reason)`, `.modify(event)` |
| `InjectedEvent` | Event injected by a hook |
| `InboundMessage` | Incoming message from a provider |
| `InboundResult` | Result of processing an inbound message |
| `Participant` | Participant data model |
| `ProviderResult` | Result from a provider operation |
| `Room` | Room data model |
| `RoomContext` | Context passed to hooks (room, bindings, participants, events) |
| `RoomEvent` | Core event stored in the timeline |
| `RoomTimers` | Timer configuration for room inactivity |
| `SessionStartedEvent` | Event fired when a voice session starts |
| `TextContent` | Plain text content |
| `Tool` | Base class for tool definitions |
| `ToolCallCallback` | Callback type for tool call events |
| `ToolCallEvent` | Tool call event model |
| `ToolHandler` | Tool handler type for realtime voice |
| `get_current_voice_session` | Get the current voice session from context |

### Errors

| Symbol | Description |
|--------|-------------|
| `RoomKitError` | Base exception |
| `RoomNotFoundError` | Room does not exist |
| `ChannelNotFoundError` | Channel not attached to room |
| `ChannelNotRegisteredError` | Channel not registered with framework |
| `ParticipantNotFoundError` | Participant not found in room |
| `IdentityNotFoundError` | Identity not found |
| `SourceAlreadyAttachedError` | Source already attached |
| `SourceNotFoundError` | No source attached |
| `VoiceBackendNotConfiguredError` | Voice backend not configured |
| `VoiceNotConfiguredError` | Voice (STT/TTS) not configured |

### AI Documentation Helpers

| Symbol | Description |
|--------|-------------|
| `get_llms_txt()` | Get llms.txt content |
| `get_llms_full_txt()` | Get llms-full.txt content (comprehensive) |
| `get_agents_md()` | Get AGENTS.md content |
| `get_ai_context()` | Get combined AI context |

## RoomKit Constructor

```python
kit = RoomKit(
    store=InMemoryStore(),                 # ConversationStore implementation
    identity_resolver=None,                # IdentityResolver implementation
    identity_channel_types=None,           # Channel types to resolve identity for
    inbound_router=None,                   # InboundRoomRouter implementation
    lock_manager=None,                     # RoomLockManager implementation
    realtime=None,                         # RealtimeBackend implementation
    max_chain_depth=5,                     # AI-to-AI loop prevention
    identity_timeout=10.0,                 # Identity resolution timeout (seconds)
    process_timeout=30.0,                  # Inbound processing timeout (seconds)
    task_runner=None,                      # Background task runner
    delivery_strategy=None,                # Delivery strategy
    status_bus=None,                       # Status bus for orchestration
    telemetry=None,                        # TelemetryProvider
    inbound_rate_limit=None,               # Framework-level rate limit
    orchestration=None,                    # Orchestration strategy
)
```

## Key RoomKit Methods

### Room Lifecycle

| Method | Description |
|--------|-------------|
| `create_room(room_id?, metadata?, orchestration?)` | Create a room |
| `get_room(room_id)` | Get room by ID |
| `close_room(room_id)` | Close a room |
| `update_room_metadata(room_id, metadata)` | Update room metadata |
| `check_room_timers(room_id)` | Check timer transitions for one room |
| `check_all_timers()` | Check all room timers |

### Channel Operations

| Method | Description |
|--------|-------------|
| `register_channel(channel)` | Register a channel |
| `attach_channel(room_id, channel_id, category?, access?, ...)` | Attach channel to room |
| `detach_channel(room_id, channel_id)` | Detach channel from room |
| `mute(room_id, channel_id)` | Mute a channel |
| `unmute(room_id, channel_id)` | Unmute a channel |
| `set_access(room_id, channel_id, access)` | Set channel access level |

### Voice/Video

| Method | Description |
|--------|-------------|
| `join(room_id, channel_id, participant_id?, ...)` | Join voice/video session |
| `leave(session)` | Leave voice/video session |
| `transcribe(audio)` | Speech-to-text |
| `synthesize(text, voice?)` | Text-to-speech |

### Inbound Pipeline

| Method | Description |
|--------|-------------|
| `process_inbound(message, room_id?)` | Process an inbound message |

### Hooks

| Method | Description |
|--------|-------------|
| `hook(trigger, execution?, priority?, ...)` | Decorator to register a hook |
| `on(event_type)` | Decorator for framework events |
| `identity_hook(trigger, ...)` | Decorator for identity hooks |
| `on_delivery_status(fn)` | Decorator for delivery status |
| `add_room_hook(room_id, trigger, execution, fn, ...)` | Add room-scoped hook |
| `remove_room_hook(room_id, name)` | Remove room-scoped hook |

### Realtime

| Method | Description |
|--------|-------------|
| `publish_typing(room_id, user_id, is_typing?)` | Typing indicator |
| `publish_presence(room_id, user_id, status)` | Presence update |
| `publish_reaction(room_id, user_id, target_event_id, emoji)` | Reaction |
| `publish_read_receipt(room_id, user_id, event_id)` | Read receipt |
| `subscribe_room(room_id, callback)` | Subscribe to ephemeral events |
| `unsubscribe_room(subscription_id)` | Unsubscribe |

### Sources

| Method | Description |
|--------|-------------|
| `attach_source(channel_id, source, auto_restart?, ...)` | Attach event source |
| `detach_source(channel_id)` | Detach event source |
| `source_health(channel_id)` | Get source health |

### Other

| Method | Description |
|--------|-------------|
| `delegate(room_id, agent_id, task, ...)` | Delegate to background agent |
| `send_greeting(room_id, channel_id?, greeting?, ...)` | Send greeting |
| `send_event(room_id, channel_id, content, ...)` | Send event directly |
| `get_timeline(room_id, offset?, limit?)` | Query event timeline |
| `close()` | Shutdown framework |

## Provider Subpackage Imports

### AI Providers

```python
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig
from roomkit.providers.gemini.ai import GeminiAIProvider
from roomkit.providers.gemini.config import GeminiConfig
from roomkit.providers.mistral.ai import MistralAIProvider
from roomkit.providers.mistral.config import MistralConfig
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.ai.base import AIProvider, AIContext, AIResponse, AITool, AIToolCall
```

### SMS Providers

```python
from roomkit.providers.twilio.sms import TwilioSMSProvider
from roomkit.providers.twilio.config import TwilioConfig
from roomkit.providers.telnyx.sms import TelnyxSMSProvider
from roomkit.providers.telnyx.config import TelnyxConfig
from roomkit.providers.sinch.sms import SinchSMSProvider
from roomkit.providers.sinch.config import SinchConfig
from roomkit.providers.sms.mock import MockSMSProvider
```

### Voice (Lazy Loaders)

```python
from roomkit.voice import (
    get_deepgram_provider, get_deepgram_config,
    get_elevenlabs_provider, get_elevenlabs_config,
    get_sherpa_onnx_stt_provider, get_sherpa_onnx_tts_provider,
    get_local_audio_backend,
    get_fastrtc_backend,
    get_rtp_backend,
    get_sip_backend,
    get_gemini_live_provider,
    get_openai_realtime_provider,
    get_xai_realtime_provider,
    get_websocket_realtime_transport,
    get_speex_aec_provider,
    get_rnnoise_denoiser_provider,
)
```

### Voice Mocks

```python
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
```

### Pipeline

```python
from roomkit.voice.pipeline import (
    AudioPipelineConfig, VADConfig,
    MockVADProvider, VADEvent, VADEventType,
    MockDenoiserProvider, MockDiarizationProvider,
    MockAGCProvider, MockAECProvider, MockDTMFDetector,
    MockAudioRecorder, MockTurnDetector, MockBackchannelDetector,
)
from roomkit.voice.interruption import InterruptionConfig, InterruptionStrategy
from roomkit.voice.audio_frame import AudioFrame
```

### Orchestration

```python
from roomkit.orchestration.state import get_conversation_state, ConversationState
from roomkit.orchestration.router import ConversationRouter, RoutingRule
from roomkit.orchestration.pipeline import ConversationPipeline, PipelineStage
from roomkit.orchestration.handoff import HandoffHandler, HandoffMemoryProvider
```

### Storage

```python
from roomkit.store.base import ConversationStore
from roomkit.store.memory import InMemoryStore
from roomkit.store.postgres import PostgresStore
```

### Content Types

```python
from roomkit.models.event import (
    TextContent, RichContent, MediaContent, AudioContent, VideoContent,
    LocationContent, CompositeContent, TemplateContent, SystemContent,
    EditContent, DeleteContent,
)
```
