# Architecture

## Hub-and-Spoke Model

RoomKit uses a hub-and-spoke architecture. The `RoomKit` class is the central hub. Channels are the spokes. Messages flow in through channels, get processed by the hub, then broadcast out to all channels attached to the room.

```
                    +------------------+
                    |     RoomKit      |
                    |  (orchestrator)  |
                    +--------+---------+
                             |
         +-------------------+-------------------+
         |         |         |         |         |
     +---+---+ +---+---+ +--+---+ +---+---+ +---+---+
     |  SMS  | | Email | | Voice| |  AI   | |  WS   |
     +-------+ +-------+ +------+ +-------+ +-------+
```

## Inbound Pipeline

Every inbound message follows an immutable pipeline order:

```
1. InboundRoomRouter.route()        # Find target room by channel binding
2. Channel.handle_inbound()         # Parse external format -> RoomEvent
3. IdentityResolver.resolve()       # Map sender_id -> participant
4. Identity hooks                   # ON_IDENTITY_AMBIGUOUS / ON_IDENTITY_UNKNOWN
5. Room lock acquired               # Per-room atomic processing
6. Idempotency check                # Deduplicate by provider_message_id
7. BEFORE_BROADCAST hooks           # Sync: can block or modify the event
8. Store event + update counters    # Persist to ConversationStore
9. EventRouter.broadcast()          # Deliver to all attached channels
   -> Content transcoding           # Adapt content per channel capabilities
   -> Rate limiting                 # TokenBucketRateLimiter per binding
   -> Retry with backoff            # RetryPolicy per binding
10. AFTER_BROADCAST hooks           # Async: fire-and-forget side effects
11. Room lock released
```

This order is defined in the RFC (Section 10.1) and must not be reordered.

## Channel Categories

Channels have two categories:

- **TRANSPORT** — Push messages to users (SMS, Email, WebSocket, Voice). Default category.
- **INTELLIGENCE** — Generate responses (AI, agents). Receives broadcasts, responds through the inbound pipeline.

## Channel-to-AI Message Flow (Reentry Loop)

When you send a message from a transport channel, it flows to the AI and back automatically:

```
1. Transport channel (SMS/WS/Voice) → kit.process_inbound(InboundMessage)
2. Inbound pipeline: route → parse → identity → hooks → store
3. EventRouter.broadcast() → delivers event to ALL attached channels
   ├── Transport channels: note delivery (no response)
   └── AI channel: calls LLM provider → generates response
       └── Returns ChannelOutput(response_events=[RoomEvent])
4. REENTRY LOOP: AI response re-enters as new inbound event
   ├── BEFORE_BROADCAST hooks run again (ConversationRouter stamps routing)
   ├── Event stored in timeline
   ├── Broadcast to all channels again
   │   ├── Transport channels: DELIVER the AI response to users
   │   └── Other AI channels: see the response (may generate follow-up)
   └── If follow-up AI responses exist → loop again (chain depth checked)
5. Chain depth limit (default max=5) → stops AI-to-AI infinite loops
6. AFTER_BROADCAST hooks fire (async side effects)
```

**Key insight**: The "pipeline" is a reentry loop — not a linear chain. AI responses go back through the same BEFORE_BROADCAST hooks, content transcoding, and broadcast cycle as user messages.

### Minimal Channel→AI Example

```python
from roomkit import RoomKit, AIChannel, WebSocketChannel, ChannelCategory, InboundMessage, TextContent
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig

kit = RoomKit()

# Transport channel (user-facing)
ws = WebSocketChannel("ws-user")
ws.register_connection("conn-1", on_recv)
kit.register_channel(ws)

# Intelligence channel (AI)
ai = AIChannel("ai", provider=AnthropicAIProvider(AnthropicConfig(
    api_key="sk-ant-...", model="claude-sonnet-4-20250514",
)))
kit.register_channel(ai)

# Room wires them together
await kit.create_room(room_id="chat")
await kit.attach_channel("chat", "ws-user")  # TRANSPORT (default)
await kit.attach_channel("chat", "ai", category=ChannelCategory.INTELLIGENCE)

# User sends message → AI responds → response delivered to WebSocket
await kit.process_inbound(
    InboundMessage(channel_id="ws-user", sender_id="user", content=TextContent(body="Hi!"))
)
```

### Pipeline vs Pipeline

RoomKit uses "Pipeline" in two contexts:

| Term | What It Is | Where |
|------|-----------|-------|
| **Inbound processing pipeline** | The 11-step message processing flow (route → parse → identity → hooks → store → broadcast) | `core/mixins/inbound_locked.py` |
| **Pipeline orchestration strategy** | A linear agent chain (triage → handler → resolver) for multi-agent handoffs | `from roomkit import Pipeline` |

The inbound pipeline processes every message. The Pipeline strategy controls which agent handles each turn.

## Pluggable Components

Every core component follows the ABC + default pattern:

| Component | ABC | Default | Purpose |
|-----------|-----|---------|---------|
| `ConversationStore` | `store/base.py` | `InMemoryStore` | Room, event, participant persistence |
| `RoomLockManager` | `core/locks.py` | `InMemoryLockManager` | Per-room atomic processing |
| `RealtimeBackend` | `realtime/base.py` | `InMemoryRealtime` | Ephemeral events (typing, presence) |
| `IdentityResolver` | `identity/base.py` | `None` (disabled) | Sender -> participant mapping |
| `InboundRoomRouter` | `core/inbound_router.py` | `DefaultInboundRoomRouter` | Route messages to rooms |

Replace any component at construction:

```python
from roomkit import RoomKit
from roomkit.store.postgres import PostgresStore

kit = RoomKit(store=PostgresStore("postgresql://..."))
```

## Room Lifecycle

```
ACTIVE -> PAUSED -> CLOSED -> ARCHIVED
           ^          |
           +----------+  (can close from paused)
```

- **ACTIVE** — Accepting messages, all channels active.
- **PAUSED** — Messages queued, channels paused. Auto-transition via `inactive_after_seconds`.
- **CLOSED** — No new messages. Auto-transition via `closed_after_seconds`.
- **ARCHIVED** — Terminal state, read-only.

## Event Model

Every message becomes a `RoomEvent` with:

- **content** — One of 11 content types (TextContent, RichContent, MediaContent, AudioContent, VideoContent, LocationContent, CompositeContent, TemplateContent, EditContent, DeleteContent, SystemContent)
- **source** — Who sent it (channel_id, participant_id, direction)
- **index** — Sequential, monotonically increasing per room
- **metadata** — Arbitrary key-value data

## Voice Architecture

The voice subsystem has three layers:

1. **VoiceBackend** — Pure audio transport (mic, SIP, RTP, WebRTC). No speech detection.
2. **AudioPipeline** — Processes audio frames: Resampler -> Recorder -> AEC -> AGC -> Denoiser -> VAD -> Diarization.
3. **VoiceChannel** — Wires backend -> pipeline -> STT/TTS, handles interruption and turn detection.

```
Inbound:   Backend -> [Resampler] -> [Recorder] -> [AEC] -> [AGC] -> [Denoiser] -> VAD -> [Diarization] + [DTMF]
Outbound:  TTS -> [PostProcessors] -> [Recorder] -> AEC.feed_reference -> [Resampler] -> Backend
```

## Realtime Voice Architecture

Speech-to-speech AI (Gemini Live, OpenAI Realtime) bypasses STT/TTS entirely:

- **RealtimeVoiceProvider** — Handles the AI model connection (WebSocket to Gemini/OpenAI).
- **RealtimeAudioTransport** — Handles browser/client audio (WebSocket or WebRTC).
- **RealtimeVoiceChannel** — Bridges transport <-> provider with tool calling support.
