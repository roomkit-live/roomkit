# RoomKit

> Pure async Python library for multi-channel conversations with rooms, hooks, and pluggable backends.

## Quick Reference

```bash
# Install dependencies
uv sync

# Run all tests
pytest tests/ -q

# Run specific test file
pytest tests/test_framework.py -v

# Lint check
ruff check src/roomkit/

# Lint fix
ruff check src/roomkit/ --fix

# Type check (optional, not enforced in CI)
mypy src/roomkit/
```

## Project Structure

```
src/roomkit/
├── __init__.py          # Public API exports - ALL public classes exported here
├── _version.py          # Version string (auto-managed)
├── core/
│   ├── framework.py     # RoomKit class - central orchestrator
│   ├── _inbound.py      # Inbound message processing pipeline
│   ├── _room_lifecycle.py  # Room CRUD operations
│   ├── _channel_ops.py  # Channel attach/detach/mute operations
│   ├── _helpers.py      # Shared helper methods
│   ├── hooks.py         # HookEngine, HookRegistration
│   ├── locks.py         # RoomLockManager ABC, InMemoryLockManager
│   ├── event_router.py  # Broadcast routing to channels
│   └── inbound_router.py # InboundRoomRouter ABC
├── channels/
│   ├── base.py          # Channel ABC
│   ├── transport.py     # TransportChannel (SMS, Email, etc.)
│   ├── ai.py            # AIChannel (intelligence layer)
│   ├── voice.py         # VoiceChannel (real-time audio)
│   ├── realtime_voice.py # RealtimeVoiceChannel (speech-to-speech AI)
│   └── websocket.py     # WebSocketChannel
├── providers/
│   ├── ai/base.py       # AIProvider ABC, AIContext, AIResponse
│   ├── sms/base.py      # SMSProvider ABC
│   ├── twilio/sms.py    # TwilioSMSProvider
│   ├── telnyx/sms.py    # TelnyxSMSProvider
│   ├── vllm/            # VLLMConfig + create_vllm_provider
│   └── ...              # Other provider implementations
├── models/
│   ├── room.py          # Room, RoomTimers
│   ├── event.py         # RoomEvent, content types
│   ├── participant.py   # Participant
│   ├── identity.py      # Identity, IdentityResult, IdentityHookResult
│   ├── delivery.py      # InboundMessage, InboundResult, DeliveryResult
│   ├── channel.py       # ChannelBinding, ChannelCapabilities
│   ├── hook.py          # HookResult, InjectedEvent
│   └── enums.py         # All enumerations
├── store/
│   ├── base.py          # ConversationStore ABC
│   └── memory.py        # InMemoryStore
├── realtime/
│   ├── base.py          # RealtimeBackend ABC, EphemeralEvent
│   └── memory.py        # InMemoryRealtime
├── voice/
│   ├── __init__.py      # Voice subsystem exports
│   ├── base.py          # Shared types (AudioChunk, VoiceSession, callbacks)
│   ├── events.py        # Voice-specific events
│   ├── stt/             # Speech-to-text providers
│   │   ├── base.py      # STTProvider ABC
│   │   ├── mock.py      # MockSTTProvider
│   │   ├── deepgram.py  # DeepgramSTTProvider
│   │   └── sherpa_onnx.py # SherpaOnnxSTTProvider
│   ├── tts/             # Text-to-speech providers
│   │   ├── base.py      # TTSProvider ABC
│   │   ├── mock.py      # MockTTSProvider
│   │   ├── elevenlabs.py # ElevenLabsTTSProvider
│   │   └── sherpa_onnx.py # SherpaOnnxTTSProvider
│   ├── realtime/        # Realtime voice (speech-to-speech)
│   │   ├── base.py      # RealtimeSession, RealtimeSessionState
│   │   ├── provider.py  # RealtimeVoiceProvider ABC
│   │   ├── transport.py # RealtimeAudioTransport ABC
│   │   ├── events.py    # Transcription, speech, tool call events
│   │   ├── ws_transport.py # WebSocketRealtimeTransport
│   │   └── mock.py      # MockRealtimeProvider, MockRealtimeTransport
│   └── backends/        # Voice transport backends
│       ├── base.py      # VoiceBackend ABC
│       ├── mock.py      # MockVoiceBackend, MockVoiceCall
│       └── fastrtc.py   # FastRTCVoiceBackend (WebRTC)
└── identity/
    ├── base.py          # IdentityResolver ABC
    └── mock.py          # MockIdentityResolver

tests/
├── conftest.py          # Shared fixtures
├── test_framework.py    # Core RoomKit tests, SimpleChannel fixture
├── test_hooks.py        # Hook system tests
├── test_identity_pipeline.py  # Identity resolution tests
└── ...
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

```python
from roomkit.channels.transport import TransportChannel
from roomkit.models.enums import ChannelType

class SMSChannel(TransportChannel):
    channel_type = ChannelType.SMS

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        phone = binding.metadata.get("phone_number")
        text = self.extract_text(event)
        result = await self._provider.send(to=phone, body=text)
        return ChannelOutput(provider_result=result)
```

### Provider Implementation

```python
from roomkit.providers.sms.base import SMSProvider

class TwilioSMSProvider(SMSProvider):
    def __init__(self, config: TwilioConfig) -> None:
        self._config = config
        self._client = httpx.AsyncClient(...)

    async def send(self, to: str, body: str, media_urls: list[str] | None = None) -> ProviderResult:
        response = await self._client.post(...)
        return ProviderResult(message_id=response["sid"], raw=response)

    async def close(self) -> None:
        await self._client.aclose()
```

### Hook Implementation

```python
from roomkit import RoomKit, HookTrigger, HookExecution, HookResult

kit = RoomKit()

# Sync hook - can block/modify events
@kit.hook(HookTrigger.BEFORE_BROADCAST)
async def content_filter(event: RoomEvent, ctx: RoomContext) -> HookResult:
    if "spam" in event.content.body.lower():
        return HookResult.block("Spam detected")
    return HookResult.allow()

# Async hook - fire-and-forget side effects
@kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC)
async def log_event(event: RoomEvent, ctx: RoomContext) -> None:
    await analytics.track("message", {"room": event.room_id})
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
async def send(self, to: str, body: str) -> ProviderResult:
    response = await self._client.post(...)
    return ProviderResult(...)

# Wrong - blocks event loop
def send(self, to: str, body: str) -> ProviderResult:
    response = requests.post(...)  # Never use sync HTTP
```

### Type Hints Required

```python
# All public methods must have type hints
async def process_inbound(self, message: InboundMessage) -> InboundResult:
    ...

# Use | for unions (Python 3.10+)
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

# Local imports - absolute from roomkit
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

# Wrong - won't work
def test_create_room(self) -> None:
    kit = RoomKit()
    room = await kit.create_room()  # SyntaxError
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

### Adding a New Hook Trigger

1. Add enum value to `HookTrigger` in `models/enums.py`
2. Add firing logic in appropriate mixin (`_inbound.py`, `_room_lifecycle.py`, etc.)
3. Add tests

## Boundaries

### Always Do

- Run `ruff check src/roomkit/` before committing
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
- Add `print()` statements - use `logging.getLogger("roomkit.xxx")`
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

### Event Flow

```
Inbound Message
    → InboundRoomRouter.route()
    → Channel.handle_inbound()
    → IdentityResolver.resolve()
    → Identity hooks (ON_IDENTITY_AMBIGUOUS/UNKNOWN)
    → Room lock acquired
    → Idempotency check
    → BEFORE_BROADCAST hooks (sync, can block/modify)
    → Store event
    → EventRouter.broadcast() to all channels
    → AFTER_BROADCAST hooks (async, fire-and-forget)
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
