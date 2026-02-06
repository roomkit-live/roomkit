# RoomKit

> Pure async Python library for multi-channel conversations with rooms, hooks, and pluggable backends.

## Quick Reference

```bash
# Install dependencies
uv sync --extra dev

# Run all checks (lint + typecheck + security + test)
make all

# Run specific checks
uv run ruff check src/roomkit/         # Lint check
uv run ruff check src/roomkit/ --fix   # Lint fix
uv run ruff format src/ tests/         # Format code
uv run mypy src/roomkit/               # Type check (enforced in CI)
uv run bandit -r src/ -c pyproject.toml # Security scan (enforced in CI)

# Run tests
uv run pytest tests/ -q                # All tests
uv run pytest tests/test_framework.py -v  # Specific test file
uv run pytest --cov=roomkit --cov-report=term-missing  # With coverage
```

## Project Structure

```
src/roomkit/
├── __init__.py              # Public API exports — ALL public classes exported here
├── _version.py              # Version string (auto-managed)
├── ai_docs.py               # AI documentation helpers (llms.txt, AGENTS.md)
├── core/
│   ├── framework.py         # RoomKit class — central orchestrator
│   ├── _inbound.py          # Inbound message processing pipeline
│   ├── _room_lifecycle.py   # Room CRUD, timers, participant resolution
│   ├── _channel_ops.py      # Channel attach/detach/mute/access/visibility
│   ├── _helpers.py          # Shared helper methods
│   ├── hooks.py             # HookEngine, HookRegistration
│   ├── event_router.py      # Broadcast routing to channels
│   ├── inbound_router.py    # InboundRoomRouter ABC, DefaultInboundRoomRouter
│   ├── locks.py             # RoomLockManager ABC, InMemoryLockManager
│   ├── circuit_breaker.py   # CircuitBreaker (closed/open/half-open)
│   ├── rate_limiter.py      # TokenBucketRateLimiter
│   ├── retry.py             # retry_with_backoff() with RetryPolicy
│   ├── transcoder.py        # DefaultContentTranscoder
│   └── router.py            # ContentTranscoder ABC
├── channels/
│   ├── __init__.py          # Factory functions: SMSChannel(), EmailChannel(), etc.
│   ├── base.py              # Channel ABC
│   ├── transport.py         # TransportChannel (generic transport wrapper)
│   ├── ai.py                # AIChannel (intelligence layer)
│   ├── voice.py             # VoiceChannel (real-time audio with STT/TTS)
│   ├── realtime_voice.py    # RealtimeVoiceChannel (speech-to-speech AI)
│   └── websocket.py         # WebSocketChannel (bidirectional real-time)
├── providers/
│   ├── ai/                  # AIProvider ABC, AIContext, AIResponse, MockAIProvider
│   ├── anthropic/           # AnthropicAIProvider, AnthropicConfig
│   ├── openai/              # OpenAIAIProvider, OpenAIConfig
│   ├── gemini/              # GeminiAIProvider, GeminiConfig
│   ├── vllm/                # VLLMConfig, create_vllm_provider (local AI)
│   ├── pydantic_ai/         # PydanticAI integration
│   ├── sms/                 # SMSProvider ABC, MockSMSProvider, phone utils
│   ├── twilio/              # TwilioSMSProvider, TwilioRCSProvider
│   ├── telnyx/              # TelnyxSMSProvider, TelnyxRCSProvider
│   ├── sinch/               # SinchSMSProvider
│   ├── voicemeup/           # VoiceMeUpSMSProvider
│   ├── rcs/                 # RCSProvider ABC, MockRCSProvider
│   ├── email/               # EmailProvider ABC, MockEmailProvider
│   ├── elasticemail/        # ElasticEmailProvider
│   ├── sendgrid/            # SendGridEmailProvider
│   ├── messenger/           # FacebookMessengerProvider, MockMessengerProvider
│   ├── teams/               # BotFrameworkTeamsProvider, MockTeamsProvider
│   ├── whatsapp/            # WhatsAppProvider ABC, WhatsAppPersonalProvider
│   └── http/                # WebhookHTTPProvider, MockHTTPProvider
├── models/
│   ├── enums.py             # All enumerations (ChannelType, EventType, HookTrigger, etc.)
│   ├── event.py             # RoomEvent, content types (Text, Rich, Media, Audio, etc.)
│   ├── room.py              # Room, RoomTimers
│   ├── participant.py       # Participant
│   ├── identity.py          # Identity, IdentityResult, IdentityHookResult
│   ├── delivery.py          # InboundMessage, InboundResult, DeliveryStatus
│   ├── channel.py           # ChannelBinding, ChannelCapabilities, RateLimit, RetryPolicy
│   ├── hook.py              # HookResult, InjectedEvent
│   ├── context.py           # RoomContext (room, bindings, participants, recent_events)
│   ├── task.py              # Task, Observation
│   └── framework_event.py   # FrameworkEvent (observability)
├── store/
│   ├── base.py              # ConversationStore ABC
│   ├── memory.py            # InMemoryStore (default)
│   └── postgres.py          # PostgresStore (asyncpg, production)
├── realtime/
│   ├── base.py              # RealtimeBackend ABC, EphemeralEvent, EphemeralEventType
│   └── memory.py            # InMemoryRealtime (default)
├── sources/
│   ├── base.py              # SourceProvider ABC, SourceStatus, SourceHealth
│   ├── websocket.py         # WebSocketSource
│   ├── sse.py               # SSESource (Server-Sent Events)
│   └── neonize.py           # WhatsAppPersonalSourceProvider
├── voice/
│   ├── __init__.py          # Voice subsystem exports
│   ├── base.py              # AudioChunk, VoiceSession, VoiceCapability, callbacks
│   ├── audio_frame.py       # AudioFrame dataclass (inbound audio)
│   ├── events.py            # BargeInEvent, VADSilenceEvent, SpeakerChangeEvent, etc.
│   ├── stt/                 # Speech-to-text: DeepgramSTT, SherpaOnnxSTT, MockSTT
│   ├── tts/                 # Text-to-speech: ElevenLabsTTS, SherpaOnnxTTS, MockTTS
│   ├── pipeline/            # Audio processing pipeline (VAD, denoiser, diarization)
│   │   ├── engine.py        # AudioPipeline orchestrator
│   │   ├── config.py        # AudioPipelineConfig
│   │   ├── vad_provider.py  # VADProvider ABC, VADEvent, VADConfig
│   │   ├── denoiser_provider.py  # DenoiserProvider ABC
│   │   ├── diarization_provider.py  # DiarizationProvider ABC, DiarizationResult
│   │   ├── postprocessor.py # AudioPostProcessor ABC (deferred)
│   │   └── mock.py          # MockVADProvider, MockDenoiserProvider, MockDiarizationProvider
│   ├── realtime/            # Speech-to-speech: GeminiLive, OpenAIRealtime, Mock
│   └── backends/            # Audio transport: FastRTCVoiceBackend, MockVoiceBackend
└── identity/
    ├── base.py              # IdentityResolver ABC
    └── mock.py              # MockIdentityResolver

tests/
├── conftest.py              # Shared fixtures
├── test_framework.py        # Core RoomKit tests, SimpleChannel fixture
├── test_framework_events.py # Framework event observability
├── test_framework_queries.py # Timeline/query tests
├── test_hooks.py            # Hook system tests
├── test_identity_pipeline.py # Identity resolution tests
├── test_realtime.py         # Ephemeral events (typing, presence, reactions)
├── test_router.py           # Event routing and transcoding
├── test_circuit_breaker.py  # Circuit breaker tests
├── test_resilience.py       # Retry and rate limiting tests
├── test_observability.py    # Task/observation tests
├── test_sources_*.py        # Source provider tests
├── test_voice*.py           # Voice subsystem tests
├── test_channels/           # Channel-specific tests
├── test_integration/        # Integration tests
└── test_providers/          # Provider-specific tests

examples/                    # 32 runnable examples (uv run python examples/<name>.py)
```

## Architecture Patterns

### ABC + Default Implementation

RoomKit uses abstract base classes with in-memory defaults:

```python
# ABC in base.py
class ConversationStore(ABC):
    @abstractmethod
    async def create_room(self, room: Room) -> Room: ...

# Default in memory.py
class InMemoryStore(ConversationStore):
    async def create_room(self, room: Room) -> Room:
        self._rooms[room.id] = room
        return room

# Usage - default is automatic
kit = RoomKit()  # Uses InMemoryStore
kit = RoomKit(store=PostgresStore(...))  # Custom backend
```

This pattern applies to: `ConversationStore`, `RoomLockManager`, `RealtimeBackend`, `IdentityResolver`, `InboundRoomRouter`.

### Channel Implementation

Channels are created via factory functions in `roomkit.channels`:

```python
from roomkit.channels import SMSChannel
from roomkit.providers.twilio.sms import TwilioSMSProvider
from roomkit.providers.twilio.config import TwilioConfig

# Factory function creates a TransportChannel with the right provider
sms = SMSChannel("sms-main", provider=TwilioSMSProvider(TwilioConfig(...)))
kit.register_channel(sms)
```

For custom channels, extend `Channel` or `TransportChannel`:

```python
from roomkit.channels.base import Channel

class CustomChannel(Channel):
    channel_type = ChannelType.WEBHOOK

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        # Send to external system
        return ChannelOutput.empty()
```

### Provider Implementation

```python
from roomkit.providers.sms.base import SMSProvider

class TwilioSMSProvider(SMSProvider):
    def __init__(self, config: TwilioConfig) -> None:
        self._config = config

    async def send(
        self, event: RoomEvent, to: str, from_: str | None = None
    ) -> ProviderResult:
        # Extract content from event and send via API
        return ProviderResult(success=True, provider_message_id="SM123")

    async def close(self) -> None:
        await self._client.aclose()
```

### Hook Implementation

```python
from roomkit import RoomKit, HookTrigger, HookExecution, HookResult

kit = RoomKit()

# Sync hook — can block/modify events (BEFORE_BROADCAST)
@kit.hook(HookTrigger.BEFORE_BROADCAST)
async def content_filter(event: RoomEvent, ctx: RoomContext) -> HookResult:
    if "spam" in event.content.body.lower():
        return HookResult.block("Spam detected")
    return HookResult.allow()

# Async hook — fire-and-forget side effects (AFTER_BROADCAST)
@kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC)
async def log_event(event: RoomEvent, ctx: RoomContext) -> None:
    await analytics.track("message", {"room": event.room_id})

# Hook with filters — only run for specific channels/directions
@kit.hook(
    HookTrigger.AFTER_BROADCAST,
    execution=HookExecution.ASYNC,
    channel_types={ChannelType.SMS},
    directions={ChannelDirection.INBOUND},
    priority=10,  # Lower runs first
)
async def sms_audit(event: RoomEvent, ctx: RoomContext) -> None:
    ...
```

### Audio Pipeline (Voice)

The audio pipeline sits between the voice backend and STT, processing raw audio frames through pluggable stages: **denoiser -> VAD -> diarization**.

```python
from roomkit import VoiceChannel
from roomkit.voice.pipeline import AudioPipelineConfig, VADConfig

# All stages are optional — configure what you need
# Traditional voice: VAD drives speech detection + STT
pipeline = AudioPipelineConfig(
    vad=my_vad_provider,
    denoiser=my_denoiser,
    diarization=my_diarizer,
    vad_config=VADConfig(silence_threshold_ms=500),
)

# Realtime voice: denoiser + diarization only (provider handles VAD)
# preprocess = AudioPipelineConfig(denoiser=my_denoiser, diarization=my_diarizer)

voice = VoiceChannel("voice", stt=stt, tts=tts, backend=backend, pipeline=pipeline)
```

**Provider ABCs** (implement these to add a new provider):

- `VADProvider` — `process(frame: AudioFrame) -> VADEvent | None`, `reset()`, `close()`
- `DenoiserProvider` — `process(frame: AudioFrame) -> AudioFrame`, `close()`
- `DiarizationProvider` — `process(frame: AudioFrame) -> DiarizationResult | None`, `reset()`, `close()`
- `AudioPostProcessor` — `process(frame: AudioFrame) -> AudioFrame`, `close()` (deferred)

**AudioFrame** (`voice/audio_frame.py`) replaces `AudioChunk` for inbound audio:

```python
@dataclass
class AudioFrame:
    data: bytes
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2        # bytes per sample (2 = 16-bit PCM)
    timestamp_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

Pipeline stages annotate `frame.metadata` as they process (e.g., `denoiser`, `vad`, `diarization` keys).

**Mock providers** (`voice/pipeline/mock.py`) accept pre-configured event sequences for testing:

```python
from roomkit.voice.pipeline import MockVADProvider, VADEvent, VADEventType

vad = MockVADProvider(events=[
    VADEvent(type=VADEventType.SPEECH_START),
    None,  # no event for this frame
    VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"speech"),
])
```

### Realtime Ephemeral Events

Typing, presence, and reactions are ephemeral — not stored in history:

```python
# Publish typing indicator
await kit.publish_typing("room-1", "alice", is_typing=True)

# Publish presence
await kit.publish_presence("room-1", "alice", "online")  # online/away/offline

# Publish reaction
await kit.publish_reaction("room-1", "alice", target_event_id="evt-123", emoji="thumbsup")

# Publish read receipt
await kit.publish_read_receipt("room-1", "alice", event_id="evt-123")

# Subscribe to ephemeral events for a room
sub_id = await kit.subscribe_room("room-1", my_callback)
await kit.unsubscribe_room(sub_id)
```

### Sources (Event-Driven Providers)

Sources maintain persistent connections and push events into the framework:

```python
from roomkit.sources.neonize import NeonizeSource

source = NeonizeSource(session_path="~/.roomkit/wa.db")
await kit.attach_source(
    "whatsapp-personal", source,
    auto_restart=True,            # Auto-restart on failure
    max_restart_attempts=5,       # Give up after 5 failures
    max_concurrent_emits=20,      # Backpressure control
)

# Check health
health = await kit.source_health("whatsapp-personal")
sources = kit.list_sources()  # {channel_id: SourceStatus}

# Detach
await kit.detach_source("whatsapp-personal")
```

### Resilience Patterns

```python
from roomkit import RateLimit, RetryPolicy
from roomkit.core.circuit_breaker import CircuitBreaker
from roomkit.core.retry import retry_with_backoff

# Rate limit on channel binding
await kit.attach_channel("room-1", "sms-out",
    rate_limit=RateLimit(max_per_second=1.0, max_per_minute=30.0),
    retry_policy=RetryPolicy(max_retries=3, base_delay_seconds=1.0),
)

# Circuit breaker for provider fault isolation
cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
if cb.allow_request():
    try:
        result = await provider.send(...)
        cb.record_success()
    except Exception:
        cb.record_failure()  # Opens after 5 consecutive failures

# Retry with exponential backoff
policy = RetryPolicy(max_retries=3, base_delay_seconds=1.0, max_delay_seconds=60.0)
result = await retry_with_backoff(flaky_function, policy)
```

### Room Timers

```python
from roomkit import RoomTimers

# Create room with auto-pause after 5 min, auto-close after 1 hour
room = await kit.create_room(room_id="support-123")
# Set timers on the room model (timers are evaluated via check_room_timers)

# Check timers for one room
room = await kit.check_room_timers("support-123")

# Batch check all rooms (call periodically, e.g. every 60s)
transitioned = await kit.check_all_timers()
```

### Delivery Status Tracking

```python
from roomkit import DeliveryStatus

@kit.on_delivery_status
async def track_delivery(status: DeliveryStatus) -> None:
    if status.status == "failed":
        logger.error("Message %s failed: %s", status.message_id, status.error_message)

# Process status webhooks from providers
await kit.process_delivery_status(status)
```

### Framework Events (Observability)

```python
# Listen for framework-level events (not message events)
@kit.on("room_created")
async def on_room_created(event):
    logger.info("Room created: %s", event.data["room_id"])

@kit.on("source_error")
async def on_source_error(event):
    logger.error("Source failed: %s", event.data["error"])

# Event types: room_created, room_closed, room_paused,
# room_channel_attached, room_channel_detached,
# channel_connected, channel_disconnected,
# voice_connected, voice_disconnected,
# source_attached, source_detached, source_error, source_exhausted
```

## Code Style

### Required in All Files

```python
from __future__ import annotations  # Always first import
```

### Models Use Pydantic

```python
from pydantic import BaseModel, Field

class Room(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    status: RoomStatus = RoomStatus.ACTIVE
    metadata: dict[str, Any] = Field(default_factory=dict)
```

### Async-First

All I/O operations must be async:

```python
# Correct
async def send(self, event: RoomEvent, to: str) -> ProviderResult:
    response = await self._client.post(...)
    return ProviderResult(success=True, provider_message_id=response["id"])

# Wrong — blocks event loop
def send(self, event: RoomEvent, to: str) -> ProviderResult:
    response = requests.post(...)  # Never use sync HTTP
```

### Type Hints Required

```python
# All public methods must have type hints
async def process_inbound(self, message: InboundMessage) -> InboundResult:
    ...

# Use | for unions (Python 3.12+)
def get_room(self, room_id: str) -> Room | None:
    ...
```

### Imports

```python
# Standard library
from __future__ import annotations
import asyncio
from typing import TYPE_CHECKING, Any

# Third party (if needed)
import httpx

# Local imports — absolute from roomkit
from roomkit.models.room import Room
from roomkit.models.enums import RoomStatus

# TYPE_CHECKING imports for circular dependencies
if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
```

## Testing Patterns

### Test Class Structure

```python
class TestFeatureName:
    async def test_specific_behavior(self) -> None:
        """Docstring describing what is tested."""
        kit = RoomKit()
        # ... test code
        assert result.expected == actual
```

### Use SimpleChannel for Tests

```python
from tests.test_framework import SimpleChannel

async def test_something() -> None:
    kit = RoomKit()
    ch = SimpleChannel("test-ch")
    kit.register_channel(ch)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "test-ch")
```

### Async Test Methods

All test methods that use async code must be `async def`:

```python
# Correct
async def test_create_room(self) -> None:
    kit = RoomKit()
    room = await kit.create_room()
    assert room.status == RoomStatus.ACTIVE
```

## Common Tasks

### Adding a New Provider

1. Create config in `providers/<name>/config.py`
2. Create provider in `providers/<name>/<type>.py`
3. Create webhook parser if needed: `parse_<name>_webhook()`
4. Export from `__init__.py`
5. Add to main `roomkit/__init__.py` exports
6. Add tests in `tests/test_<name>_provider.py`

### Adding a New Channel Type

1. Add enum value to `ChannelType` in `models/enums.py`
2. Create channel class extending `TransportChannel` or `Channel`
3. Export from `channels/__init__.py`
4. Add to main `roomkit/__init__.py` exports

### Adding a New Pipeline Provider

1. Create provider in `voice/pipeline/<name>.py` implementing the ABC (`VADProvider`, `DenoiserProvider`, or `DiarizationProvider`)
2. Implement `name` property, `process()` method, and optionally `reset()` / `close()`
3. Export from `voice/pipeline/__init__.py`
4. Add tests in `tests/test_audio_pipeline.py`

### Adding a New Hook Trigger

1. Add enum value to `HookTrigger` in `models/enums.py`
2. Add firing logic in appropriate mixin (`_inbound.py`, `_room_lifecycle.py`, etc.)
3. Add tests

## Boundaries

### Always Do

- Run `make all` before committing (lint + typecheck + security + test)
- Add tests for new features
- Export new public classes from `roomkit/__init__.py`
- Follow existing ABC patterns for new pluggable backends
- Use `model_copy(update={...})` for Pydantic model updates (immutable pattern)

### Ask First

- Adding new dependencies to `pyproject.toml`
- Changing public API signatures
- Modifying hook trigger behavior
- Changes to the inbound processing pipeline

### Never Do

- Modify `_version.py` manually
- Use synchronous I/O (requests, open()) in async methods
- Add `print()` statements — use `logging.getLogger("roomkit.xxx")`
- Break backward compatibility of public API
- Commit without running tests
- Add secrets or credentials to code

## Key Concepts

### Room Lifecycle

```
ACTIVE → PAUSED → CLOSED → ARCHIVED
         ↑          ↓
         └──────────┘ (can close from paused)
```

Timers: `inactive_after_seconds` (ACTIVE→PAUSED), `closed_after_seconds` (→CLOSED).

### Channel Access

```
READ_WRITE  — receives and sends messages (default)
READ_ONLY   — receives messages only
WRITE_ONLY  — sends messages only
NONE        — temporarily disabled
```

Channels can also be muted/unmuted per binding: `kit.mute("room", "channel")`.

### Event Flow

```
Inbound Message
    → InboundRoomRouter.route()           # Find target room
    → Channel.handle_inbound()            # Parse external → RoomEvent
    → IdentityResolver.resolve()          # Identify sender
    → Identity hooks                      # ON_IDENTITY_AMBIGUOUS/UNKNOWN
    → Room lock acquired
    → Idempotency check
    → BEFORE_BROADCAST hooks              # Sync: can block/modify
    → Store event + update room counters
    → EventRouter.broadcast()             # Deliver to all channels
        → Content transcoding             # Adapt content per channel capabilities
        → Rate limiting                   # TokenBucketRateLimiter
        → Retry with backoff              # RetryPolicy per binding
    → AFTER_BROADCAST hooks               # Async: logging, analytics, side effects
    → Framework event emitted             # Observability
```

### Identity Resolution

```python
# IdentityResult statuses and their hooks:
IDENTIFIED      → No hook, participant_id stamped on event
AMBIGUOUS       → ON_IDENTITY_AMBIGUOUS hook
PENDING         → ON_IDENTITY_AMBIGUOUS hook
UNKNOWN         → ON_IDENTITY_UNKNOWN hook
REJECTED        → ON_IDENTITY_UNKNOWN hook

# Hook can return:
IdentityHookResult.resolved(identity)  # Identity found
IdentityHookResult.pending(candidates) # Wait for manual resolution
IdentityHookResult.challenge(inject)   # Ask sender to identify
IdentityHookResult.reject(reason)      # Block the message
```

### Content Types

```python
TextContent         # Plain text with optional language
RichContent         # HTML/Markdown with buttons, cards, quick replies
MediaContent        # Image/video/document with MIME type
AudioContent        # Audio file with optional transcript
VideoContent        # Video with optional thumbnail
LocationContent     # Latitude/longitude with label/address
CompositeContent    # Multi-part (text + media + location in one event)
TemplateContent     # Pre-approved templates (WhatsApp Business)
EditContent         # Edit a previously sent message
DeleteContent       # Delete a previously sent message
SystemContent       # System notifications with code & data
```

### Hook Triggers

```
Event Pipeline:      BEFORE_BROADCAST, AFTER_BROADCAST
Channel Lifecycle:   ON_CHANNEL_ATTACHED, ON_CHANNEL_DETACHED, ON_CHANNEL_MUTED, ON_CHANNEL_UNMUTED
Room Lifecycle:      ON_ROOM_CREATED, ON_ROOM_PAUSED, ON_ROOM_CLOSED
Identity:            ON_IDENTITY_AMBIGUOUS, ON_IDENTITY_UNKNOWN, ON_PARTICIPANT_IDENTIFIED
Delivery:            ON_DELIVERY_STATUS
Side Effects:        ON_TASK_CREATED, ON_ERROR
Voice:               ON_SPEECH_START, ON_SPEECH_END, ON_TRANSCRIPTION, BEFORE_TTS, AFTER_TTS
Voice Pipeline:      ON_VAD_SILENCE, ON_VAD_AUDIO_LEVEL, ON_SPEAKER_CHANGE, ON_BARGE_IN, ON_TTS_CANCELLED
Realtime Voice:      ON_REALTIME_TOOL_CALL, ON_REALTIME_TEXT_INJECTED
Observability:       ON_OBSERVATION
```
