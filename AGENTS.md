# RoomKit

> Pure async Python library for multi-channel conversations with rooms, hooks, and pluggable backends.

## Related Repositories

The RoomKit ecosystem spans three repos (sibling directories at `../`):

| Repo | Path | Purpose |
|------|------|---------|
| **roomkit** | `.` (this repo) | Library source, tests, examples |
| **roomkit-docs** | `../roomkit-docs/` | MkDocs documentation site (42 pages: features, architecture, guides, API reference) |
| **roomkit-specs** | `../roomkit-specs/` | Normative RFC specification (`roomkit-rfc.md`) |

### RFC Conformance

The RFC at `../roomkit-specs/roomkit-rfc.md` is the **normative specification**. All implementations must conform to it. Key invariants enforced by the RFC:

- **Inbound pipeline order** (Section 10.1): route → handle_inbound → identity → lock → idempotency → index → BEFORE_BROADCAST → store → broadcast → AFTER_BROADCAST → unlock. MUST NOT reorder.
- **Permission model** (Section 7): access (READ_WRITE/READ_ONLY/WRITE_ONLY/NONE), muting (suppresses responses, not side effects), visibility filtering.
- **Event indexing**: sequential, atomic, monotonically increasing per room, starting at 0.
- **Chain depth**: default max=5. MUST be enforced to prevent AI-to-AI infinite loops.
- **Voice pipeline** (Section 12): stage ordering, AEC double-feeding prevention, DTMF parallel execution before AEC/AGC/denoiser, recording consent, turn detection latency <200ms.
- **Side effects always collected**: tasks and observations are captured regardless of mute/access. "Muting silences the voice, not the brain."
- **4 conformance levels**: Level 0 (Core, REQUIRED), Level 1 (Transport, RECOMMENDED), Level 2 (Rich, OPTIONAL), Level 3 (Voice, OPTIONAL).

If a proposed change conflicts with the RFC, update the RFC first (in `roomkit-specs`), then update the code.

### Documentation

The docs at `../roomkit-docs/` are a MkDocs Material site. When adding features, update the corresponding docs page. Key structure:
- `docs/features.md` — comprehensive feature reference
- `docs/architecture.md` — architecture overview
- `docs/guides/` — practical guides (resampler, sherpa-onnx, smart-turn, WAV recorder, RTP, SIP)
- `docs/api/` — 27 API reference pages (generated via mkdocstrings)

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
│   ├── _voice_hooks.py      # Voice-specific hook implementations
│   ├── _voice_stt.py        # Speech-to-text orchestration
│   ├── _voice_tts.py        # Text-to-speech orchestration
│   ├── _voice_turn.py       # Turn-taking logic for voice channels
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
│   ├── trace.py             # ProtocolTrace (SIP/RTP debugging)
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
│   ├── stt/                 # Speech-to-text: DeepgramSTT, SherpaOnnxSTT, GradiumSTT, MockSTT
│   ├── tts/                 # Text-to-speech: ElevenLabsTTS, SherpaOnnxTTS, QwenTTS, GradiumTTS, MockTTS
│   ├── interruption.py      # InterruptionHandler, InterruptionConfig, 4 strategies
│   ├── pipeline/            # Audio processing pipeline (10 provider ABCs)
│   │   ├── engine.py        # AudioPipeline orchestrator (inbound + outbound)
│   │   ├── config.py        # AudioPipelineConfig
│   │   ├── debug_taps.py    # Debug tapping for audio inspection
│   │   ├── resampler/       # ResamplerProvider ABC, LinearResampler, SincResampler, Mock
│   │   ├── vad/             # VADProvider ABC, VADEvent, VADConfig, MockVADProvider
│   │   ├── denoiser/        # DenoiserProvider ABC, RNNoiseDenoiserProvider, Mock
│   │   ├── diarization/     # DiarizationProvider ABC, DiarizationResult, Mock
│   │   ├── aec/             # AECProvider ABC, SpeexAECProvider, Mock
│   │   ├── agc/             # AGCProvider ABC, AGCConfig, Mock
│   │   ├── dtmf/            # DTMFDetector ABC, DTMFEvent, Mock
│   │   ├── recorder/        # AudioRecorder ABC, RecordingConfig, Mock
│   │   ├── turn/            # TurnDetector ABC, TurnContext, TurnDecision, Mock
│   │   ├── backchannel/     # BackchannelDetector ABC, Mock
│   │   └── postprocessor/   # AudioPostProcessor ABC
│   ├── realtime/            # Speech-to-speech: GeminiLive, OpenAIRealtime, Mock
│   └── backends/            # Audio transport: FastRTC, RTP, SIP, Local, MockVoiceBackend
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

The audio pipeline sits between the voice backend and STT, processing audio through pluggable inbound and outbound chains:

```
Inbound:   Backend → [Resampler] → [Recorder] → [DTMF] → [AEC] → [AGC] → [Denoiser] → VAD → [Diarization]
Outbound:  TTS → [PostProcessors] → [Recorder] → AEC.feed_reference → [Resampler] → Backend
```

AEC and AGC stages are automatically skipped when the backend declares `NATIVE_AEC` / `NATIVE_AGC` capabilities via `VoiceCapability` flags.

```python
from roomkit import VoiceChannel
from roomkit.voice.pipeline import AudioPipelineConfig, VADConfig
from roomkit.voice.interruption import InterruptionConfig, InterruptionStrategy

# All stages are optional — configure what you need
pipeline = AudioPipelineConfig(
    resampler=my_resampler,
    vad=my_vad_provider,
    denoiser=my_denoiser,
    diarization=my_diarizer,
    aec=my_aec,
    agc=my_agc,
    dtmf=my_dtmf_detector,
    recorder=my_recorder,
    recording_config=my_recording_config,
    turn_detector=my_turn_detector,
    vad_config=VADConfig(silence_threshold_ms=500),
)

voice = VoiceChannel(
    "voice", stt=stt, tts=tts, backend=backend, pipeline=pipeline,
    interruption=InterruptionConfig(strategy=InterruptionStrategy.CONFIRMED, min_speech_ms=300),
)
```

**Provider ABCs** (implement these to add a new provider):

- `ResamplerProvider` — `resample(frame, target_rate, target_channels, target_width) -> AudioFrame`, `close()`
- `VADProvider` — `process(frame) -> VADEvent | None`, `reset()`, `close()`
- `DenoiserProvider` — `process(frame) -> AudioFrame`, `reset()`, `close()`
- `DiarizationProvider` — `process(frame) -> DiarizationResult | None`, `reset()`, `close()`
- `AudioPostProcessor` — `process(frame) -> AudioFrame`, `reset()`, `close()`
- `AGCProvider` — `process(frame) -> AudioFrame`, `reset()`, `close()`
- `AECProvider` — `process(frame) -> AudioFrame`, `feed_reference(frame)`, `reset()`, `close()`
- `DTMFDetector` — `process(frame) -> DTMFEvent | None`, `reset()`, `close()`
- `AudioRecorder` — `start(session, config) -> RecordingHandle`, `stop(handle) -> RecordingResult`, `tap_inbound/outbound(handle, frame)`
- `TurnDetector` — `evaluate(context: TurnContext) -> TurnDecision`
- `BackchannelDetector` — `classify(context: BackchannelContext) -> BackchannelDecision`

**Capability flags** (`voice/base.py`):
- `VoiceCapability.NATIVE_AEC` — backend has built-in echo cancellation; AEC stage is skipped
- `VoiceCapability.NATIVE_AGC` — backend has built-in gain control; AGC stage is skipped
- `VoiceCapability.DTMF_INBAND` — backend sends DTMF as in-band audio tones
- `VoiceCapability.DTMF_SIGNALING` — backend sends DTMF via out-of-band signaling

**InterruptionHandler** (`voice/interruption.py`):
Four strategies for handling user speech during TTS playback:
- `IMMEDIATE` — interrupt on any speech (matches legacy `enable_barge_in=True`)
- `CONFIRMED` — wait for `min_speech_ms` of sustained speech (default)
- `SEMANTIC` — use `BackchannelDetector` to ignore acknowledgements ("uh-huh", "yeah")
- `DISABLED` — ignore speech during playback (matches legacy `enable_barge_in=False`)

Legacy `enable_barge_in`/`barge_in_threshold_ms` params are automatically mapped to the appropriate strategy.

**Turn detection** (`voice/pipeline/turn/base.py`):
`TurnDetector` integrates post-STT in `VoiceChannel._process_speech_end()`. Transcription fragments accumulate in `_pending_turns` until the detector returns `is_complete=True`, at which point the combined text is routed. `ON_TURN_COMPLETE` and `ON_TURN_INCOMPLETE` hooks fire accordingly.

**AudioFrame** (`voice/audio_frame.py`) is the inbound audio unit:

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

Pipeline stages annotate `frame.metadata` as they process (e.g., `denoiser`, `vad`, `aec`, `agc`, `diarization`, `dtmf` keys).

**Mock providers** (each in its provider's `mock.py`, e.g. `voice/pipeline/vad/mock.py`) — 9 mocks for testing: `MockVADProvider`, `MockDenoiserProvider`, `MockDiarizationProvider`, `MockAGCProvider`, `MockAECProvider`, `MockDTMFDetector`, `MockAudioRecorder`, `MockTurnDetector`, `MockBackchannelDetector`. All accept pre-configured event sequences:

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
# voice_session_started, voice_session_ended,
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

1. Create provider in the appropriate subdirectory, e.g. `voice/pipeline/vad/silero.py`, implementing the corresponding ABC (`VADProvider`, `DenoiserProvider`, `DiarizationProvider`, `AGCProvider`, `AECProvider`, `DTMFDetector`, `AudioRecorder`, `AudioPostProcessor`, `TurnDetector`, or `BackchannelDetector`)
2. Implement `name` property, `process()` (or `evaluate()` for TurnDetector/BackchannelDetector), and `reset()` / `close()`
3. Add a corresponding mock to the provider's `mock.py` (e.g. `voice/pipeline/vad/mock.py`)
4. Export from the subdirectory's `__init__.py` and from `voice/pipeline/__init__.py`
5. Add tests in `tests/test_audio_pipeline.py`

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
                     ON_DTMF, ON_TURN_COMPLETE, ON_TURN_INCOMPLETE, ON_BACKCHANNEL,
                     ON_RECORDING_STARTED, ON_RECORDING_STOPPED
Audio Levels:        ON_INPUT_AUDIO_LEVEL, ON_OUTPUT_AUDIO_LEVEL
Realtime Voice:      ON_REALTIME_TOOL_CALL, ON_REALTIME_TEXT_INJECTED
Observability:       ON_OBSERVATION
```
