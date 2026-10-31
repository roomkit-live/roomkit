# API Reference

RoomKit exports **348 symbols** from `roomkit`. Providers and voice types import from subpackages.

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
| `acp_event_text` | The text an `ACPChannel` gives its agent for an event (a `RichContent` as its `plain_text`), for a host building its own ACP prompt |
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
| `HostChannel` | The host application's own voice in a room: a `HOST` transport, never delivered to, that `kit.post()` speaks from (RFC §10.5) |
| `HTTPChannel` | HTTP webhook transport channel factory |
| `MessengerChannel` | Facebook Messenger transport channel factory |
| `RCSChannel` | RCS transport channel factory |
| `RealtimeAudioVideoChannel` | Realtime speech-to-speech with video |
| `RealtimeVoiceChannel` | Speech-to-speech AI channel |
| `SMSChannel` | SMS transport channel factory |
| `TeamsChannel` | Microsoft Teams transport channel factory |
| `TelegramChannel` | Telegram Bot transport channel factory |
| `TransportChannel` | Generic transport channel wrapper; takes its type's addressing unless given `addressing=` |
| `ChannelAddressing` | A channel's addressing declaration: `addressing`, `recipient_key`, `chat_key`, `replies_to_sender` (RFC §5.7) |
| `Addressing` | How a channel's addresses are written: PHONE_NUMBER (E.164), EMAIL_ADDRESS, CHAT, AS_WRITTEN |
| `addressing_of(channel_type)` | A type's default addressing (RFC Appendix A) |
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
| `ChannelType` | An open name (RFC 6.2); the well-known ones are members: SMS, MMS, RCS, EMAIL, WHATSAPP, WHATSAPP_PERSONAL, WEBSOCKET, AI, VOICE, REALTIME_VOICE, REALTIME_AUDIO_VIDEO, PUSH, MESSENGER, TELEGRAM, TEAMS, DISCORD, BUZZ, WEBHOOK, VIDEO, AUDIO_VIDEO, CONFERENCE, CLI, HOST, SYSTEM; `ChannelType("slack")` names another |
| `EventStatus` | PENDING, DELIVERED, READ, FAILED, BLOCKED |
| `EventType` | 28 values (MESSAGE, SYSTEM, EDIT, DELETE, TOOL_CALL_START, DTMF, etc.) |
| `HookExecution` | SYNC or ASYNC |
| `HookTrigger` | 85 hook triggers — full list in hooks.md |
| `RoomStatus` | ACTIVE, PAUSED, CLOSED, ARCHIVED |
| `Visibility` | Scope keywords for an event's `visibility` field: ALL, NONE, TRANSPORT, INTELLIGENCE, INTERNAL (the mark of a record the framework wrote: `attach_channel`, `set_visibility`, `send_event` and `InboundMessage` refuse it, an agent's answer carrying it is stored NONE) |

### Models

| Symbol | Description |
|--------|-------------|
| `ChannelBinding` | Binding of a channel to a room |
| `ChannelCapabilities` | Declared capabilities of a channel |
| `ChannelOutput` | Output of a channel delivery; its `metadata_updates` are written on the room once the delivery set has run, a `roomkit:` key or a key of the room's private state dropped with a warning |
| `EventSource` | Source attribution for an event |
| `FrameworkEvent` | Lightweight framework lifecycle event |
| `HookResult` | Result from sync hooks: `.allow()`, `.block(reason)`, `.modify(event)` |
| `InjectedEvent` | Event injected by a hook |
| `InboundMessage` | Incoming message from a provider |
| `InboundResult` | Inbound outcome; `response_metadata` contains the final root intelligence response record |
| `Participant` | Participant data model |
| `ProviderResult` | Result from a provider operation |
| `Room` | Room data model |
| `RoomContext` | Context passed to hooks (room, bindings, participants, events) |
| `RoomEvent` | Core event stored in the timeline |
| `RoomTimers` | Timer configuration for room inactivity: `inactive_after_seconds` (from `last_activity_at`, or `created_at` before any), `closed_after_seconds` (from `Room.paused_at`) |
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
| `EmptyEventDescriber` | `AIChannel(describe_empty_event=...)`: an event whose content extracts to nothing, in; the text the transcript reads for it, or `None` to omit it, out |
| `MediaResolver` | `RoomKit(media_resolver=...)`: a referenced `MediaContent` (`<scheme>://…`) and its room id, in; the media's bytes, or `None` when this room may not show it, out (RFC §6.4) |
| `AttachedMedia` | `RoomKit(attached_media=...)`: an event in; the `MediaContent`s its application keeps beside its content out, the same ones in the same order each time, which every AI channel reads after the event's own content (RFC §6.4) |
| `ContinuationPolicy` | `AIChannel(continuation=...)`: the text of a round the model ended itself without a call, in; the instruction to go on, or `None`, out (RFC §6.4) |
| `ToolRoundEvent` | Payload for AFTER_TOOL_ROUND hooks, between two rounds of a tool loop: the round's calls and results, the turn's toolset, `withdraw()` and `add_message()` |
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
| `Immediate` | Deliver now; on voice, after the playback under way |
| `Queued` | Add to queue, deliver at the next idle window |
| `WaitForIdle` | Wait for TTS/speech to finish, then send |
| `DeliveryBackend` | ABC for persistent delivery queue backends |
| `DeliveryItem` | Serializable delivery request — the unit of work in the queue |
| `DeliveryItemStatus` | Lifecycle status of a delivery item |
| `InMemoryDeliveryBackend` | Asyncio-queue delivery backend (single process, no persistence) |
| `TurnBackend` | ABC for what carries an intelligence channel's turns between the room owner and an executor (RFC §10.6) |
| `InMemoryTurnBackend` | Default turn backend: in this process, every message serialized but a request's events, which reach the executor as they are |
| `RedisTurnBackend` | Turn backend across processes (`pip install roomkit[redis]`): one per process on a shared Redis and store; an executor declared with `hosts={...}` takes turns from every process, beating from a thread; a killed executor fails its turn with `executor_lost` |
| `TurnExecutor` | Builds a channel and runs its turns; the kit owns one reading its registered channels. One a host builds (`TurnExecutor(build_channel, hosts={...}, store=, media_resolver=, attached_media=, telemetry=, error_classes=)`, `hosts` the set of channel ids it runs, or a predicate for its own process only; with an `OpenTelemetryProvider` as `telemetry=`, a turn from another process continues the owner's trace) builds each channel once, wires it as the kit wires the ones it registers (its hooks fire, what the channel publishes during a turn crosses as `ephemeral` / `framework_event` frames), and closes what it built; `executor.room_hook(name, trigger)(fn)` defines a room hook by name |
| `ChannelDeclaration` | What the room's owner knows of a channel without its object: `channel_id`, `channel_type`, `category`, `capabilities`, `identity`, `thinks_while_listening`, `recent_events_window`; `channel.declaration()` gives a live channel's |
| `AgentIdentity` | An agent's identity in its declaration: `name`, `role`, `description`, `scope`, `greeting`, `language`, `auto_greet` |
| `DeclaredChannel` | The channel the kit holds for a registered declaration: bound and routed to, never run in the kit's process |
| `TurnClient` | The room owner's side: runs a turn, reads it back as the channel's `ChannelOutput` |
| `OwnerTools` | ABC for the tools a room owner serves its turns: what a request lists, how a call's handler runs |
| `OwnerHost` | ABC for what a room owner runs for its turns' `TurnHost` (`TurnClient(host=)`): `call(request, verb, arguments)`, and `call_between_turns(message)`, refused by default |
| `OwnerEvents` | ABC for where a room owner publishes what its turns published on their executors (`TurnClient(events=)`): `publish(channel, event)`, `emit(...)`, and `reaches_room(room_id, channel_id)`, whose default `False` publishes no turn ephemeral (RFC §8.4) |
| `OwnerTool` | One tool a turn's request lists, served by the room owner (`tool`, `waits_by_design`) |
| `TurnHost` | What a host's tool or hook on a turn's executor asks of the room owner: `send_event`, `delegate`, `wait_task`, `post_status`, `update_room_metadata`, `add_observation` |
| `current_turn_host()` | The `TurnHost` of the turn the current code runs in; after the turn, in work it began, the host for `send_event`, `post_status`, `update_room_metadata`, `add_observation` (`host.open` false); `None` elsewhere |
| `RoomCall` / `RoomPublication` | What an executor sends its room owner after a turn ended (RFC §10.6.3): a host verb, answered by a `HostResult`, and a publication; accepted from the executor of the room's last turn only |
| `RoomOwner` | ABC for what reaches a room owner outside a turn (`room_call`, `room_publication`); `TurnBackend.attend(owner)` has executors reach it |
| `current_turn_id()` | The id of the turn the current code runs in, which `kit.steer(turn_id=...)` addresses; `None` outside a turn |
| `PreviousTurn` | A request's (or a channel call's) `previous_turn`: the room's last turn or channel call of the channel (`turn_id`, `executor_id`), `None` after a detach; an executor drops a room's state another left (RFC §10.6.2) |
| `RoomAttached` / `RoomDetached` | Channel control to the executors that may run the channel's turns (the kit's own alone for a channel it holds live, else every executor declaring it): it calls `on_room_attached` / `on_room_detached` and logs what they raise (RFC §10.6.3) |
| `ChannelCall` / `ChannelResult` | A channel's per-room verb, sent to the executor holding the room's state, and its answer (`result`, `error`, the changed `room_state`) |
| `CommittedEvent` | Channel control to the executor holding the room's state, else one declaring the channel, without moving the holder: it calls `on_room_event(room_id, event)` (a discussion's memory hand-off, RFC §10.6.3) |
| `ChannelControl` | The union of `RoomAttached`, `RoomDetached`, `CommittedEvent` and `ChannelCall` that `TurnBackend.control()` carries |
| `TurnLinkError` | A turn the backend could not carry to its end, its `reason`: `no_executor` (at once when no executor takes the channel, or once the kit's own executor closed for a channel it holds live), `executor_lost`, `frame_lost`, `frame_invalid`, `backend_closed` (a turn backend closed; the executor is not lost) |
| `RemoteTurnError` | An error raised across the turn backend of a class the reading side does not rebuild (outside RoomKit, Python's built-ins and its `error_classes`): keeps the class name (`kind`), its ancestors (`bases`) and its JSON attributes, and is also an instance of the nearest ancestor that side rebuilds (a `ProviderError` subclass still reads as one) |
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
| `RoomFilter` | The statuses and the organization a routing lookup reads |
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
| `save_conversation_state` | Persist a conversation state, writing its metadata key alone |
| `HandoffHandler` | Processes handoff tool calls |
| `HandoffRequest` | Parsed from an agent's handoff tool call arguments |
| `HandoffResult` | Result returned to the calling agent after a handoff |
| `HANDOFF_TOOL` | AITool definition for transferring a conversation to another agent |
| `setup_handoff` | Wires handoff into an AIChannel's tool chain; takes the agent or the id of one the kit holds |

### Memory, Skills & Sandbox

| Symbol | Description |
|--------|-------------|
| `MemoryProvider` | ABC for pluggable memory backends feeding AI context construction |
| `TurnFootprint` | What a turn takes of the window besides its history, measured by the AI channel before its memory read (`input_tokens`, `reply_tokens`); read with `current_turn_footprint()` from `roomkit.memory` (RFC §20) |
| `Skill` | Full skill definition including instructions body |
| `SkillMetadata` | Lightweight metadata parsed from SKILL.md frontmatter |
| `SkillRegistry` | Discovers, loads, and manages Agent Skills; tracks three visibility states — available, unlisted (`mark_unlisted`: activatable but out of the prompt manifest), unavailable (`mark_unavailable`: listed with a reason, not activatable); `add(skill)` registers one built in memory, `copy(names?, marks=True)` a subset with its paths and marks |
| `RunSkillScriptTool` | `run_skill_script` as a `Tool` a realtime channel serves for skills another agent holds, through the one script handler |
| `ScriptExecutor` | ABC for executing skill scripts with integrator-defined policy |
| `SandboxExecutor` | ABC for executing commands in a sandboxed environment |
| `SandboxResult` | Result of executing a sandbox command |

### Errors

| Symbol | Description |
|--------|-------------|
| `RoomKitError` | Base exception |
| `KitClosedError` | A call made once `kit.close()` started, but a read (answered while the store is open) or a teardown call (returns doing nothing); a read on a store the kit closed; a call it cut before its commit; also the code (`kit_closed`) of a send it cut (RFC §12.10.4) |
| `ShutdownCancelledError` | The kit's shutdown was cancelled before it ended, to a caller nobody cancelled (RFC §12.10.4) |
| `RoomNotFoundError` | Room does not exist |
| `RoomAlreadyExistsError` | A room created under an id the store holds (RFC §14.3) |
| `RoomClosedError` | Room's status refuses new events (RFC §5.1) |
| `ReentrantCallError` | A message routed back into the room whose lock the caller holds, or from an off-lock hook's body or a parse; nothing written (RFC §9.5.1) |
| `ReentryLoopCapError` | A direct `update_event()`, `delete_event()` or `regenerate_response()` an ASYNC hook's body made past the cascade's hook calls; nothing written (RFC §9.5.1, §10.3) |
| `TransitionRefusedError` | A lifecycle transition from a status it does not leave, such as `close_room()` on an ARCHIVED room; nothing written (RFC §5.1) |
| `ChangeRefusedError` | `send_event()` of an edit or a delete step 8a refused; `reason` says why (RFC §10.3) |
| `EventNotFoundError` | A store's `commit_event()` of a change whose target the room no longer holds (RFC §14.3) |
| `IdempotencyKeyTakenError` | A store's `commit_event()` of an event under a key its source channel holds in the room for another event; nothing written (RFC §13.4, §14.3) |
| `RoomNotAttachedError` | Channel acted on a room it is no longer attached to |
| `ChannelNotFoundError` | Channel not attached to room |
| `ChannelNotRegisteredError` | Channel not registered with framework (`send_event()` from it, whether or not the room still binds it) |
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
    process_timeout=30.0,          # One deadline from a call's start to its decision (seconds)
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
    turn_backend=None,             # TurnBackend for intelligence turns (default: InMemoryTurnBackend)
    error_classes=(),              # Application exception classes (with subclasses) rebuilt as themselves across the turn backend
)
```

## Key RoomKit Methods

### Room Lifecycle

| Method | Description |
|--------|-------------|
| `create_room(room_id?, metadata?, orchestration?)` | Create a room; a key of *metadata* only the framework writes is dropped with a warning, the rest written: any `roomkit:` key (a detach's revocation, what the room keeps for a channel) and a key of the room's private state (`author_register`, `_speak_queue_results`) |
| `get_room(room_id)` | Get room by ID |
| `close_room(room_id)` | Close a room; a CLOSED room is returned as stored, an ARCHIVED one raises `TransitionRefusedError` |
| `archive_room(room_id)` | Archive a room, the terminal status; an ARCHIVED room is returned as stored |
| `update_room_metadata(room_id, metadata)` | Update room metadata; a key only the framework writes is dropped with a warning, the rest written: any `roomkit:` key (a detach's revocation `roomkit:detached_channels`, what the room keeps for a channel `roomkit:channel_state:`, written by `channel_call` alone) and a key of the room's private state (`author_register`, `_speak_queue_results`). A read-modify-write (`{**room.metadata, ...}`) writes and leaves those keys as they are; a turn's `TurnHost.update_room_metadata` drops them the same way |
| `check_room_timers(room_id)` | Apply one room's due timer transitions, each at the instant it fell due (RFC §5.1) |
| `check_all_timers()` | Check every ACTIVE and PAUSED room; returns those that transitioned |

### Channel Operations

| Method | Description |
|--------|-------------|
| `register_channel(channel)` | Register a channel, or a `ChannelDeclaration`: an intelligence channel held by its declaration alone, its turns run on an executor that builds it (RFC §10.6.2) |
| `attach_channel(room_id, channel_id, category?, access?, ...)` | Attach channel to room |
| `detach_channel(room_id, channel_id)` | Detach channel from room |
| `mute(room_id, channel_id)` | Mute a channel |
| `unmute(room_id, channel_id)` | Unmute a channel |
| `set_access(room_id, channel_id, access)` | Set channel access level |

### Voice/Video

| Method | Description |
|--------|-------------|
| `join(room_id, channel_id, *, session=None, participant_id=None, ..., organization_id=None)` | Join voice/video session; the room read with its scope (another organization's is not found, no session joins it) |
| `leave(session)` | Leave voice/video session |
| `transcribe(audio)` | Speech-to-text |
| `synthesize(text, voice?)` | Text-to-speech |

### Inbound Pipeline

| Method | Description |
|--------|-------------|
| `process_inbound(message, room_id?, defer_delivery=False, organization_id=None)` | Process an inbound message; deferred results are backfilled by `DeliveryHandle.wait()` |
| `cancel_delivery(key, *, reason="caller_cancelled", organization_id=None)` | Stop the turns of the deliveries in flight of the messages sent with `delivery_key=key`, from the start of `process_inbound`, and a message held for verification until its release is answered (of the organization's rooms when given); the messages are still delivered; returns how many it cancelled |
| `regenerate_response(room_id, *, trigger_id=None, organization_id=None)` | Re-run the intelligence channel on the newest accepted transport message; `InboundResult(blocked=True, reason="room_closed")` on a closed or archived room; with `trigger_id`, `reason="trigger_moved"` when the selection under the lock is no longer that event; from an ASYNC hook's body, one of the cascade's hook calls (`ReentryLoopCapError` past them) |
| `regenerate_target(room_id, *, organization_id=None)` | The event `regenerate_response` would replay, or `None`; raises `RoomClosedError` when the room refuses events |

### Hooks

| Method | Description |
|--------|-------------|
| `hook(trigger, execution?, priority?, ...)` | Decorator to register a hook |
| `on(event_type)` | Decorator for framework events |
| `identity_hook(trigger, ...)` | Decorator for identity hooks |
| `on_delivery_status(fn)` | Decorator for delivery status |
| `await add_room_hook(room_id, trigger, execution, fn, ..., organization_id=None)` | Add room-scoped hook on an owner trigger |
| `await enable_room_hook(room_id, name, *, organization_id=None)` / `await disable_room_hook(room_id, name, *, organization_id=None)` | Run (or stop running) for the room's turns the room hook *name* the executors define on a trigger a turn fires; only an executor defining every enabled name takes the room's turns (RFC §10.6.6) |
| `await remove_room_hook(room_id, name, *, organization_id=None)` | Remove room-scoped hook |

### Realtime

| Method | Description |
|--------|-------------|
| `publish_typing(room_id, user_id, is_typing?, *, organization_id=None)` | Typing indicator |
| `publish_presence(room_id, user_id, status, *, organization_id=None)` | Presence update |
| `publish_reaction(room_id, user_id, target_event_id, emoji, *, organization_id=None)` | Reaction |
| `publish_read_receipt(room_id, user_id, event_id, *, organization_id=None)` | Read receipt |
| `subscribe_room(room_id, callback, *, organization_id=None)` | Subscribe to ephemeral events; another organization's room is not found, nothing subscribed |
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
| `delegate(room_id, agent_id, task, ..., organization_id=None)` | Delegate to background agent |
| `setup_background(agent, worker?, hand_back?)` | Give an agent (or the id of one the kit holds) the `work_in_background` tool: its long work runs as a task (RFC §23.4) |
| `setup_task_tools(agent)` | Give an agent (or the id of one the kit holds) `task_status` and `cancel_task`, served by the room's owner: the room's own tasks only (RFC §23.4) |
| `await start_task(room_id, agent_id, task, hand_back?, organization_id?)` | Start an agent's background task from the host, for any agent of the room |
| `send_greeting(room_id, channel_id?, greeting?, ..., organization_id=None)` | Send greeting |
| `send_event(room_id, channel_id, content, ...)` | Send event directly: `INBOUND` from a transport, `OUTBOUND` from an intelligence channel; from the room's own work (a hook, a parse, a delivery, a turn) it draws on that work's reentry budget (RFC §10.5, §9.5.1). A channel the kit does not register raises `ChannelNotRegisteredError` before anything is read, one the room still binds included; one the room does not bind, `ChannelNotFoundError` |
| `await post(room_id, content, *, participant_id?, addressed_to=(), metadata?, channel_id?, organization_id?, idempotency_key?, chain_depth=0)` | Write to the room's people in the host's own name, from a `HostChannel` attached on a room's first post (refused for a room it was detached from, or closed); solicits no agent unless addressed; speaks as a participant the host channel reaches (`ParticipantNotFoundError` for any other id); a key the channel holds returns the original, sending nothing again (RFC §10.5) |
| `await steer(directive, channel_id, *, room_id=None, turn_id=None, organization_id=None)` | Steer the running turns of an intelligence channel the kit started, wherever they run; answers the loops reached (RFC §21.3, §10.6.8) |
| `await channel_call(channel_id, room_id, verb, *, organization_id=None, **arguments)` | Run a per-room verb of a channel attached to the room, on the executor holding the room's state (an ACP agent's `set_config_option`, `close_session`, `session_config`, `config_options`); what the room keeps for the channel is stored in its metadata, written by this call alone (RFC §10.6.3, §6.4) |
| `await answer_input(request_id, value)` / `await reject_input(request_id, reason)` | Answer or reject a turn's human-input request, wherever its tool runs; an early answer is held 60 s, a second one returns `False` (RFC §10.6.5) |
| `commit_event(room_id, event, organization_id?)` | Commit a record no member receives, outside the pipeline: stored as given, its `delivery_results` included, its thread parent rooted as every event's, its index counted delivered, no hook, no broadcast; never a replay, a key its channel holds raises `IdempotencyKeyTakenError` (RFC §10.5) |
| `await start_room_recording(room_id, recorders, *, organization_id=None)` | Start recorders on an existing room, all or nothing, each announced (ON_RECORDING_STARTED) before any media |
| `await room_recordings(room_id, *, organization_id=None)` | The handles of the room's running recordings; the room read with its scope (another organization's, or a missing room recording nothing, is not found) |
| `await add_room_recording_track(room_id, track, *, organization_id=None)` | Declare a track to the room's recordings; returns a `RoomRecordingFeed` (`feed(data, timestamp_ms)`, `close()`), or `None` when the room records nothing; the room read with its scope |
| `await stop_room_recording(room_id, *, organization_id=None)` | Stop the room's recordings, each result announced (ON_RECORDING_STOPPED) |
| `get_timeline(room_id, offset?, limit?)` | Query event timeline |
| `close()` | Close the kit (RFC §12.10.4): every new call but a read or a teardown call, synchronous or not, raises `KitClosedError` at once, queued deliveries, lock waiters and model turns are cut at once, the running work gets a 5 s grace then is cut (1 s more, twice, for work ignoring its cut, then abandoned and logged); every call it cut returns `cancellation_reason="kit_closed"` with what it delivered, its event when its commit landed; one shutdown shared by every caller |

## Provider Subpackage Imports

### AI Providers

```python
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig
from roomkit.providers.openai import OpenAIAIProvider, OpenAIConfig
from roomkit.providers.gemini import GeminiAIProvider, GeminiConfig
from roomkit.providers.mistral import MistralAIProvider, MistralConfig
from roomkit.providers.ai import (
    AIContext,
    AIProvider,
    AIResponse,
    AITool,
    AIToolCall,
    MockAIProvider,
)
```

### SMS Providers

```python
from roomkit.providers.twilio import TwilioConfig, TwilioSMSProvider
from roomkit.providers.telnyx import TelnyxConfig, TelnyxSMSProvider
from roomkit.providers.sinch import SinchConfig, SinchSMSProvider
from roomkit.providers.sms import MockSMSProvider
```

### Voice Providers and Backends

```python
# Each module loads, with the vendor SDK it needs, on first use of its name.
from roomkit.voice.stt import DeepgramConfig, DeepgramSTTProvider, SherpaOnnxSTTProvider
from roomkit.voice.tts import (
    ElevenLabsConfig,
    ElevenLabsTTSProvider,
    GeminiTTSConfig,
    GeminiTTSProvider,
    SherpaOnnxTTSProvider,
)
from roomkit.voice.backends import (
    FastRTCVoiceBackend,
    LocalAudioBackend,
    RTPVoiceBackend,
    SIPVoiceBackend,
)
from roomkit.providers.gemini import GeminiLiveProvider
from roomkit.providers.openai import OpenAIRealtimeProvider
from roomkit.providers.xai import XAIRealtimeProvider
from roomkit.voice.realtime import WebSocketRealtimeTransport
from roomkit.voice import RNNoiseDenoiserProvider, SpeexAECProvider
```

### Voice Mocks

```python
from roomkit.voice.backends import MockVoiceBackend
from roomkit.voice.stt import MockSTTProvider
from roomkit.voice.tts import MockTTSProvider
from roomkit.voice.realtime import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.testing import ScenarioVoiceBackend, VoiceTrace, PCMAudio, read_wav, write_wav
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
from roomkit.voice import (
    AudioFrame,
    InterruptionConfig,
    InterruptionStrategy,
    STTLanguageLock,
)
```

### Orchestration

```python
from roomkit.orchestration import (
    ConversationPipeline,
    ConversationRouter,
    ConversationState,
    HandoffHandler,
    HandoffMemoryProvider,
    PipelineStage,
    RoutingRule,
    get_conversation_state,
)
```

### Storage

```python
from roomkit.store import ConversationStore, InMemoryStore, PostgresStore
```

### Content Types

```python
from roomkit.models import (
    AudioContent,
    CompositeContent,
    DeleteContent,
    EditContent,
    LocationContent,
    MediaContent,
    RichContent,
    SystemContent,
    TemplateContent,
    TextContent,
    VideoContent,
)
```
