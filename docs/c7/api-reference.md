# API Reference

RoomKit exports **170 symbols** from `roomkit`. Providers and voice types import from subpackages.

## Top-Level Imports (`from roomkit import ...`)

### Framework

| Symbol | Description |
|--------|-------------|
| `RoomKit` | Central orchestrator — rooms, channels, hooks, storage |
| `RoomKitConsole` | Full-screen terminal dashboard for voice agent development (optional, requires `rich`) |
| `__version__` | Package version string |
| `content_logging_enabled` | Whether raw message content may be written to logs (default False) |
| `set_content_logging` | Enable/disable process-wide logging of raw message content |

### Channels

| Symbol | Description |
|--------|-------------|
| `ACPChannel` | Connects a room to an external ACP coding agent over stdio |
| `Agent` | AI agent with role, description, greeting, tools |
| `AIChannel` | Intelligence layer for AI responses |
| `AIChannelTurnConfig` | Per-turn generation overrides for AIChannel (None fields keep channel defaults) |
| `AudioVideoChannel` | Combined audio + video channel |
| `BuzzChannel` | Buzz (Nostr relay) transport channel factory |
| `Channel` | Base class for all channels |
| `CLIChannel` | Interactive terminal channel |
| `ConferenceChannel` | Multi-party conference channel backed by an external SFU |
| `DiscordChannel` | Discord bot transport channel factory |
| `EmailChannel` | Email transport channel factory |
| `FrameworkAwareChannel` | Channel base handed the framework it is registered with |
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

### Conference

| Symbol | Description |
|--------|-------------|
| `BotSession` | The framework's own connection to a conference |
| `ConferenceAccess` | Credentials a client uses to join the conference directly |
| `ConferenceBackend` | ABC for SFU conference backends |
| `ConferenceBargeIn` | Event: a participant spoke over the bot and was allowed to interrupt it |
| `ConferenceCapability` | Flag enum of capabilities a ConferenceBackend can support |
| `ConferenceGrants` | Permissions encoded into a participant's conference access |
| `ConferenceInterruptionConfig` | Multi-party interruption policy |
| `ConferenceInterruptionScope` | Who may interrupt the bot while it is speaking |
| `ConferenceParticipant` | A participant's media presence in a conference |
| `ConferenceRealtimeConfig` | Composes a speech-to-speech provider with a conference |
| `ConferenceRecordingConfig` | Configuration for recording a conference |
| `ConferenceRecordingMode` | Where a conference recording is produced |
| `ConferenceRecordingStarted` | Event: a track's recording has opened |
| `ConferenceRecordingStopped` | Event: a track's recording has closed, with its destination |
| `ConferenceToolHandler` | Callable type for conference tool invocation |
| `ConferenceTrack` | A single media stream published by a conference participant |
| `ConferenceTranscription` | What a lane produced, before it enters the room |
| `TrackKind` | Kind of media carried by a conference track |
| `LiveKitConferenceBackend` | ConferenceBackend backed by a LiveKit SFU |
| `LiveKitConfig` | Connection and behaviour settings for LiveKitConferenceBackend |
| `MockConferenceBackend` | Conference backend that scripts SFU events for tests |
| `MockDelivery` | Mock media timing — how long one frame took to reach every subscriber |
| `MockFaults` | Per-operation failures and delays for the mock backend |
| `MockTrackFormat` | Audio format a participant negotiated for one track (mock) |
| `MockUtterance` | Chunks published for one utterance on one bot's track (mock) |
| `CONFERENCE_ADDRESS_KEYS` | Participant-attribute keys read as a caller's address, most specific first |
| `CONFERENCE_METADATA_KEY` | `Participant.metadata` key a conference nests provider data under (`"conference"`) |
| `CONFERENCE_UNASSERTED_METADATA_KEY` | Metadata key nesting client-claimed (unverified) participant attributes |

### Video

| Symbol | Description |
|--------|-------------|
| `VideoDetectionEvent` | Detection event emitted by video pipeline filters |
| `FaceTouchFilter` | Detects hand-to-face contact using MediaPipe landmarks |
| `FaceTouchConfig` | Configuration for face touch detection |
| `FaceTouchSensitivity` | Sensitivity presets controlling detection thresholds |
| `FaceZone` | Face zones that can be monitored for touch detection |
| `MockFaceTouchFilter` | Mock filter emitting pre-configured detection events at specific frames |

### Enums

| Symbol | Description |
|--------|-------------|
| `Access` | Channel access levels: READ_WRITE, READ_ONLY, WRITE_ONLY, NONE |
| `ChannelCategory` | TRANSPORT or INTELLIGENCE |
| `ChannelType` | 23 values: SMS, MMS, RCS, EMAIL, WHATSAPP, WHATSAPP_PERSONAL, WEBSOCKET, AI, VOICE, REALTIME_VOICE, REALTIME_AUDIO_VIDEO, PUSH, MESSENGER, TELEGRAM, TEAMS, DISCORD, BUZZ, WEBHOOK, VIDEO, AUDIO_VIDEO, CONFERENCE, CLI, SYSTEM |
| `EventStatus` | PENDING, DELIVERED, READ, FAILED, BLOCKED |
| `EventType` | 27 values (MESSAGE, SYSTEM, EDIT, DELETE, TOOL_CALL_START, DTMF, etc.) |
| `HookExecution` | SYNC or ASYNC |
| `HookTrigger` | 76 hook triggers — full list in hooks.md |
| `RoomStatus` | ACTIVE, PAUSED, CLOSED, ARCHIVED |
| `Visibility` | Scope keywords for an event's `visibility` field: ALL, NONE, TRANSPORT, INTELLIGENCE, INTERNAL |

### Models

| Symbol | Description |
|--------|-------------|
| `ChannelBinding` | Binding of a channel to a room |
| `ChannelCapabilities` | Declared capabilities of a channel |
| `ChannelOutput` | Output of a channel delivery |
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
| `get_current_voice_session` | Get the current voice session from context |

### Tools, Callbacks & Human Input

| Symbol | Description |
|--------|-------------|
| `Tool` | Base class for tool definitions |
| `ToolHandler` | Tool handler type for realtime voice |
| `ToolPolicy` | Per-agent allow/deny rules for tool access |
| `RoleOverride` | Per-role tool policy override |
| `ToolCallCallback` | Callback type for tool call events |
| `ToolCallEvent` | Tool call event model |
| `ToolCallContent` | Content for TOOL_CALL_START and TOOL_CALL_END events |
| `AIGenerationEvent` | Payload for BEFORE_AI_GENERATION hooks, before AI provider invocation |
| `AIResponseEvent` | Payload for ON_AI_RESPONSE hooks, after AI generation completes |
| `BeforeGenerationCallback` | Async callback type receiving AIGenerationEvent |
| `AfterResponseCallback` | Async callback type receiving AIResponseEvent |
| `HumanInputHandler` | Manages pending human input requests |
| `HumanInputToolHandler` | ToolHandler wrapper that blocks on human input for specified tools |
| `PendingInput` | A pending human input request |
| `PendingInputEvent` | Event fired through ON_USER_INPUT_REQUIRED hooks |
| `PendingInputStatus` | Status of a pending human input request |

### Delivery

| Symbol | Description |
|--------|-------------|
| `DeliveryStrategy` | ABC controlling when and how content is delivered to a channel |
| `Immediate` | Deliver now; may interrupt ongoing TTS playback |
| `Queued` | Add to queue, deliver at the next idle window |
| `WaitForIdle` | Wait for TTS/speech to finish, then send |
| `DeliveryBackend` | ABC for persistent delivery queue backends |
| `DeliveryItem` | Serializable delivery request — the unit of work in the queue |
| `DeliveryItemStatus` | Lifecycle status of a delivery item |
| `InMemoryDeliveryBackend` | Asyncio-queue delivery backend (single process, no persistence) |
| `DeliveryResult` | Result of delivering a message |
| `DeliveryStatus` | Delivery status from provider webhook |

### Storage & Locking

| Symbol | Description |
|--------|-------------|
| `ConversationStore` | ABC for persistent room/event/binding/participant storage |
| `InMemoryStore` | Dict-based in-memory store for development and testing |
| `SQLiteStore` | Embedded stdlib SQLite store for single-process persistence |
| `SQLiteSchemaError` | Unsupported or unsafe SQLite schema migration error |
| `RoomLockManager` | ABC for per-room locking |
| `InMemoryLockManager` | In-process per-room asyncio locks with LRU eviction |
| `EventFilter` | Filter criteria for querying room events |
| `PersistencePolicy` | Controls which event types are persisted to the store |

### Orchestration

| Symbol | Description |
|--------|-------------|
| `Loop` | Producer/reviewer cycle strategy |
| `Orchestration` | ABC for orchestration strategies |
| `Pipeline` | Linear agent chain strategy |
| `Supervisor` | Supervisor delegates to workers strategy |
| `Swarm` | Bidirectional handoff strategy |
| `ConversationPhase` | Built-in conversation phases (StrEnum) |
| `ConversationState` | Tracks conversation progress within a room |
| `ConversationRouter` | Routes events to the appropriate agent |
| `ConversationPipeline` | Generates routing rules for sequential agent workflows |
| `PipelineStage` | A stage in a ConversationPipeline |
| `RoutingRule` | Routing rule mapping conditions to an agent |
| `RoutingConditions` | Conditions for a routing rule to match |
| `get_conversation_state` | Extract typed ConversationState from room metadata |
| `set_conversation_state` | Return a room copy with updated conversation state |
| `HandoffHandler` | Processes handoff tool calls |
| `HandoffRequest` | Parsed from an agent's handoff tool call arguments |
| `HandoffResult` | Result returned to the calling agent after a handoff |
| `HANDOFF_TOOL` | AITool definition for transferring a conversation to another agent |
| `setup_handoff` | Wires handoff into an AIChannel's tool chain |

### Memory, Skills & Sandbox

| Symbol | Description |
|--------|-------------|
| `MemoryProvider` | ABC for pluggable memory backends feeding AI context construction |
| `Skill` | Full skill definition including instructions body |
| `SkillMetadata` | Lightweight metadata parsed from SKILL.md frontmatter |
| `SkillRegistry` | Discovers, loads, and manages Agent Skills |
| `ScriptExecutor` | ABC for executing skill scripts with integrator-defined policy |
| `SandboxExecutor` | ABC for executing commands in a sandboxed environment |
| `SandboxResult` | Result of executing a sandbox command |

### Errors

| Symbol | Description |
|--------|-------------|
| `RoomKitError` | Base exception |
| `RoomNotFoundError` | Room does not exist |
| `RoomClosedError` | Room's status refuses new events (RFC §5.1) |
| `RoomNotAttachedError` | Channel acted on a room it is no longer attached to |
| `ChannelNotFoundError` | Channel not attached to room |
| `ChannelNotRegisteredError` | Channel not registered with framework |
| `ParticipantNotFoundError` | Participant not found in room |
| `ParticipantNotAdmittedError` | Participant barred from what was asked for them |
| `IdentityNotFoundError` | Identity not found |
| `SourceAlreadyAttachedError` | Source already attached |
| `SourceNotFoundError` | No source attached |
| `VoiceBackendNotConfiguredError` | Voice backend not configured |
| `VoiceNotConfiguredError` | Voice (STT/TTS) not configured |
| `ConferenceAlreadyAttachedError` | Second conference channel attached to a room |
| `ConferenceCapabilityError` | Conference operation needs a capability the backend lacks |
| `ConferenceCloseError` | Conference channel did not close all of its resources |

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
    store=None,                    # ConversationStore (default: InMemoryStore)
    identity_resolver=None,        # IdentityResolver for identifying inbound senders
    identity_channel_types=None,   # Restrict identity resolution to these ChannelTypes (None = all)
    inbound_router=None,           # InboundRoomRouter (default: DefaultInboundRoomRouter)
    lock_manager=None,             # RoomLockManager (default: InMemoryLockManager)
    realtime=None,                 # RealtimeBackend for ephemeral events (default: InMemoryRealtime)
    max_chain_depth=5,             # Max reentry chain depth — AI-to-AI loop prevention
    identity_timeout=10.0,         # Identity resolution timeout (seconds)
    process_timeout=30.0,          # Locked inbound processing timeout (seconds)
    stt=None,                      # STTProvider for transcription
    tts=None,                      # TTSProvider for synthesis
    voice=None,                    # VoiceBackend for real-time audio transport
    task_runner=None,              # TaskRunner for delegated background tasks (default: InMemoryTaskRunner)
    delivery_strategy=None,        # DeliveryStrategy | str — proactive delivery of task results
    delivery_backend=None,         # DeliveryBackend — persistent queue for deliver() (None = in-process)
    status_bus=None,               # StatusBus for multi-agent coordination (default: in-memory)
    telemetry=None,                # TelemetryProvider or TelemetryConfig (default: no-op)
    inbound_rate_limit=None,       # RateLimit applied to inbound messages, keyed per channel_id
    orchestration=None,            # Default Orchestration strategy for create_room()
    persistence_policy=None,       # PersistencePolicy — which event types are persisted (None = all)
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
    get_gemini_tts_provider, get_gemini_tts_config,
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
