# CLAUDE.md - RoomKit Framework Guide

RoomKit is a pure async Python library for multi-channel conversations with rooms, hooks, and pluggable backends. It provides a unified abstraction for orchestrating messages across SMS, RCS, Email, WhatsApp, Messenger, Teams, Telegram, Voice, WebSocket, HTTP, and AI channels.

## Related Repositories

The RoomKit ecosystem spans three repos under `../` (relative to this repo root):

| Repo | Path | Purpose |
|------|------|---------|
| **roomkit** | `.` (this repo) | Library source, tests, examples |
| **roomkit-docs** | `../roomkit-docs/` | MkDocs documentation site (`docs/` with 42 pages, guides, API reference) |
| **roomkit-specs** | `../roomkit-specs/` | Normative RFC specification (`roomkit-rfc.md`) |

### RFC Conformance (IMPORTANT)

The RFC at `../roomkit-specs/roomkit-rfc.md` is the **normative specification** for RoomKit. All implementation decisions must conform to it. When in doubt about behavior, consult the RFC — it defines:

- **Exact inbound/broadcast pipeline ordering** (Section 10) — MUST NOT be reordered
- **Permission enforcement rules** (Section 7) — access, mute, visibility semantics
- **Event indexing invariants** — sequential, atomic, monotonically increasing per room
- **Chain depth limits** — default max=5, prevents AI-to-AI infinite loops
- **Voice pipeline stage ordering and constraints** (Section 12) — AEC/DTMF/recording rules
- **27 hook triggers** with sync/async execution semantics (Section 9)
- **4 conformance levels**: Level 0 (Core, REQUIRED), Level 1 (Transport, RECOMMENDED), Level 2 (Rich, OPTIONAL), Level 3 (Voice, OPTIONAL)

Before implementing new features or changing core behavior, read the relevant RFC section. If a proposed change conflicts with the RFC, the RFC must be updated first (in `roomkit-specs`), then the code follows.

### Documentation Site

The docs at `../roomkit-docs/` are built with MkDocs Material. Key sections:
- `docs/features.md` — comprehensive feature reference
- `docs/architecture.md` — high-level architecture overview
- `docs/guides/` — practical guides (resampler, sherpa-onnx, smart-turn, WAV recorder, RTP, SIP)
- `docs/api/` — 27 API reference pages (auto-generated from docstrings)

When adding new features, update the corresponding docs page in `roomkit-docs`. When adding new API surface, ensure `mkdocstrings` can generate its reference.

## Build & Test Commands

```bash
uv sync --extra dev              # Install dependencies
make all                         # Lint + typecheck + security + test (run before committing)
uv run pytest                    # Run tests
uv run pytest tests/test_X.py -v # Run specific test file
uv run ruff check src/ tests/    # Lint
uv run ruff check --fix          # Lint with auto-fix
uv run ruff format src/ tests/   # Format
uv run mypy src/roomkit/         # Type check (strict mode)
uv run bandit -r src/ -c pyproject.toml  # Security scan
uv run pytest --cov=roomkit --cov-report=term-missing  # Coverage (80% minimum)
make release VERSION=x.y.z      # Release workflow
```

## Core Architecture

### Multi-Channel Room Model

The framework centers on **Rooms** — containers that hold participants and channel bindings. Messages flow through channels into rooms and are broadcast to all attached channels.

```
Inbound Message
  → InboundRoomRouter.route()       # Find target room
  → Channel.handle_inbound()        # Parse → RoomEvent
  → IdentityResolver.resolve()      # Identify sender
  → BEFORE_BROADCAST hooks          # Can block/modify
  → Store event
  → EventRouter.broadcast()         # Deliver to all channels
    → Content transcoding           # Adapt per channel capabilities
    → Rate limiting + retry
  → AFTER_BROADCAST hooks           # Async side effects
```

### Key Components

- **RoomKit** (`core/framework.py`) — central orchestrator, registers channels, manages rooms
- **Channel** (`channels/base.py`) — ABC for all channel types; `TransportChannel` wraps a provider
- **VoiceChannel** (`channels/voice.py`) — real-time audio with STT/TTS/pipeline
- **RealtimeVoiceChannel** (`channels/realtime_voice.py`) — speech-to-speech AI (Gemini Live, OpenAI Realtime)
- **AIChannel** (`channels/ai.py`) — intelligence layer for AI responses
- **HookEngine** (`core/hooks.py`) — event filtering/modification via hook triggers
- **AudioPipeline** (`voice/pipeline/engine.py`) — inbound + outbound audio processing

### Voice Pipeline

Audio processing between voice backend and STT, with pluggable stages:

```
Inbound:   Backend → [Resampler] → [Recorder] → [AEC] → [AGC] → [Denoiser] → VAD → [Diarization] + [DTMF]
Outbound:  TTS → [PostProcessors] → [Recorder] → AEC.feed_reference → [Resampler] → Backend
```

- **VoiceBackend** is a pure transport — no VAD/speech detection
- AEC/AGC stages auto-skip when backend has `NATIVE_AEC`/`NATIVE_AGC` capabilities
- **InterruptionHandler** (`voice/interruption.py`): 4 strategies — IMMEDIATE, CONFIRMED, SEMANTIC, DISABLED
- **AudioFrame** for inbound audio, **AudioChunk** for outbound TTS
- 10 pipeline provider ABCs, each in own subdirectory under `voice/pipeline/`

### Providers

All in `providers/` with the pattern: ABC in `base.py`, implementations alongside:

- **AI**: Anthropic, OpenAI, Gemini, Mistral, vLLM
- **SMS**: Twilio, Sinch, Telnyx, VoiceMeUp
- **Email**: ElasticEmail, SendGrid
- **Chat**: Telegram, Teams (Bot Framework), Facebook Messenger
- **RCS**: Twilio, Telnyx
- **WhatsApp**: Cloud API, Personal (neonize)

### Storage & Realtime

- **ConversationStore** ABC — `InMemoryStore` (default), `PostgresStore` (production)
- **RealtimeBackend** ABC — ephemeral events (typing, presence, reactions)

## Project Structure

```
src/roomkit/
├── core/           # Framework, hooks, event routing, inbound pipeline
├── channels/       # Channel ABC + implementations (Voice, AI, WebSocket, etc.)
├── providers/      # 21 subdirs — AI, SMS, Email, Chat provider integrations
├── models/         # Pydantic models — enums, events, rooms, participants
├── voice/          # Voice subsystem
│   ├── backends/   # Audio transports (FastRTC, RTP, SIP, Local, Mock)
│   ├── stt/        # Speech-to-text (Deepgram, SherpaOnnx, Gradium, Mock)
│   ├── tts/        # Text-to-speech (ElevenLabs, SherpaOnnx, Qwen, Gradium, Mock)
│   ├── pipeline/   # 10 audio processing stages (VAD, AEC, AGC, Denoiser, etc.)
│   └── realtime/   # Speech-to-speech (OpenAI Realtime, Gemini Live)
├── store/          # Conversation persistence (Memory, Postgres)
├── realtime/       # Ephemeral events (typing, presence, reactions)
├── sources/        # Event-driven providers (WebSocket, SSE, Neonize)
└── identity/       # User identification resolution
tests/              # 100+ test files, conftest.py with shared fixtures
examples/           # 30+ runnable examples
```

## Important Patterns

### ABC + Default Implementation

Every pluggable backend follows: ABC in `base.py`, default in-memory impl, mock for testing.

```python
# ABC
class ConversationStore(ABC):
    @abstractmethod
    async def create_room(self, room: Room) -> Room: ...

# Default
class InMemoryStore(ConversationStore): ...

# Usage
kit = RoomKit()                          # InMemoryStore by default
kit = RoomKit(store=PostgresStore(...))   # Custom backend
```

Applies to: `ConversationStore`, `RoomLockManager`, `RealtimeBackend`, `IdentityResolver`, `InboundRoomRouter`.

### Channel Factory Functions

Channels created via PascalCase factories in `roomkit.channels`:

```python
from roomkit.channels import SMSChannel
sms = SMSChannel("sms-main", provider=TwilioSMSProvider(TwilioConfig(...)))
kit.register_channel(sms)
```

### Hook System

```python
@kit.hook(HookTrigger.BEFORE_BROADCAST)          # Sync: can block/modify
async def filter(event, ctx) -> HookResult: ...

@kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC)  # Fire-and-forget
async def log(event, ctx) -> None: ...
```

### Pipeline Provider Pattern

Each of the 10 pipeline stages lives in `voice/pipeline/<stage>/`:
- `base.py` — ABC with `process()`, `reset()`, `close()`
- `mock.py` — Mock with pre-configured event sequences
- Implementation files alongside (e.g., `vad/silero.py`, `aec/speex.py`)

## Development Standards

### Code Style

- `from __future__ import annotations` — always first import
- Python 3.12+ — use `X | None` unions, not `Optional[X]`
- Type hints required on all public methods
- Ruff: 99-char line length, `E/F/I/N/UP/B/SIM` rules
- Models use Pydantic `BaseModel`
- Async-first — never use synchronous I/O in async methods
- Logging via `logging.getLogger("roomkit.xxx")` — no `print()`
- All public classes exported from `roomkit/__init__.py`

### Testing

- Framework: pytest with `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed
- Use `SimpleChannel` fixture from `tests/test_framework.py` for core tests
- Use mock providers (`MockVADProvider`, `MockSTTProvider`, etc.) with event sequences
- Voice pipeline tests: `backend.simulate_audio_received()` to trigger processing
- Run `uv run pytest` (not `python -m pytest`)

### Adding New Components

Every new feature, provider, or channel type **must** include:

1. **Tests** — unit tests + integration tests where applicable
2. **Documentation** — update or add a guide in `../roomkit-docs/docs/guides/`, add a section to `../roomkit-docs/docs/features.md`, and update `../roomkit-docs/mkdocs.yml` nav if adding a new guide page
3. **Example** — add a runnable example in `examples/` demonstrating the feature end-to-end

**New provider**: config in `providers/<name>/config.py`, implementation in `providers/<name>/<type>.py`, export from `__init__.py` and `roomkit/__init__.py`, add tests, add example in `examples/`, add to docs.

**New pipeline stage**: implement ABC in `voice/pipeline/<stage>/`, add mock, export from subdirectory and `voice/pipeline/__init__.py`, add tests, add guide in `../roomkit-docs/docs/guides/`, add example.

**New channel type**: add to `ChannelType` enum, extend `TransportChannel` or `Channel`, export from `channels/__init__.py`, add tests, add example, update `features.md`.

## Boundaries

### Always

- Run `make all` before committing
- Add tests for new features
- Add documentation and a runnable example for new features/providers
- Export new public classes from `roomkit/__init__.py`
- Use `model_copy(update={...})` for Pydantic model updates

### Ask First

- Adding dependencies to `pyproject.toml`
- Changing public API signatures
- Modifying hook trigger behavior
- Changes to the inbound processing pipeline

### Never

- Modify `_version.py` manually (managed by release script)
- Use synchronous I/O in async methods
- Add `print()` — use `logging.getLogger()`
- Break backward compatibility of public API
- Commit without running tests
- Add secrets or credentials to code

## Reference

For detailed API patterns, content types, hook triggers, and code examples, see [AGENTS.md](AGENTS.md).
