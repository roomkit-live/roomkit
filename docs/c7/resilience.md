# Resilience

RoomKit provides built-in resilience patterns for production deployments: rate limiting, circuit breakers, retry with backoff, chain depth limits, and room lifecycle timers.

## Rate Limiting

Apply rate limits per channel binding:

```python
from roomkit import RoomKit
from roomkit.models import RateLimit

kit = RoomKit()

await kit.attach_channel("room-1", "sms-out",
    rate_limit=RateLimit(max_per_second=1.0, max_per_minute=30.0),
)
```

Framework-level inbound rate limiting:

```python
from roomkit import RoomKit
from roomkit.models import RateLimit

kit = RoomKit(inbound_rate_limit=RateLimit(max_per_second=10.0))
```

## Circuit Breaker

RoomKit keeps one circuit breaker per transport channel. Five consecutive
failed deliveries (each after its binding's retries) open it: deliveries to
that channel are then skipped, without calling the provider, for 60 seconds;
after that one probe delivery goes through, and a success closes the breaker.
A delivery refused for having no recipient does not count. Each transition is
a framework event:

```python
@kit.on("circuit_breaker_opened")
async def breaker_opened(event):
    logger.warning("Deliveries to %s suspended", event.data["channel_id"])

@kit.on("circuit_breaker_closed")
async def breaker_closed(event):
    logger.info("Deliveries to %s resumed", event.data["channel_id"])
```

States: CLOSED (normal) -> OPEN (failing, fast-reject) -> HALF_OPEN (testing recovery).

## Retry with Backoff

A binding's `RetryPolicy` retries a failed delivery with exponential backoff
(`base_delay_seconds × exponential_base^attempt`, capped at
`max_delay_seconds`). An error that declares itself not retryable, or one
outside `retryable_errors` when the policy names some, fails on its first
attempt:

```python
from roomkit.models import RetryPolicy

await kit.attach_channel("room-1", "sms-out",
    retry_policy=RetryPolicy(
        max_retries=3,
        base_delay_seconds=1.0,
        max_delay_seconds=60.0,
    ),
)
```

## Connect vs Read Timeout

Every httpx-backed provider config in `roomkit.providers` carries `timeout` (the read/write/pool budget, sized for the slowest response) and `connect_timeout` (default 5 s, the TCP connect alone, TLS handshake included). Each adapter hands its client or SDK a `Timeout(timeout, connect=connect_timeout)` (httpx's, or the SDK's own where it runs on httpx2: the `openai` SDK and Anthropic's) rather than a bare float, so a host that no longer accepts connections is given up on in seconds rather than after the read budget: with one float for both, the kernel exhausts its SYN retries first, about two minutes per attempt, whatever the value.

```python
from roomkit.providers.polargrid import PolarGridConfig

config = PolarGridConfig(
    api_key="pg_...",
    timeout=240.0,  # read/write/pool: sized for the slowest generation
    connect_timeout=5.0,  # TCP connect alone (the default)
)
```

Applies to the AI providers (OpenAI and its derivatives, Azure, Anthropic, Ollama, vLLM, PolarGrid), the image providers, and the SMS, RCS, email, Telegram, Messenger and webhook transports. `AnamConfig.timeout` has no counterpart: it bounds the SDK's connect call itself, not an HTTP client. Raise `connect_timeout` on a link or a TLS-terminating proxy slow enough to need more.

Outside `roomkit.providers`, the same pair sits on `GrokTTSConfig`, `OpenAIVisionConfig`, `GeminiVisionConfig`, `GeminiTTSConfig` and `GeminiSTTConfig`, and is a keyword on `WebSocketAvatarProvider` and `SSESource`. Two read differently: `SSESource.timeout` bounds write and pool only (its read side is left unbounded so the stream survives idle periods between events; `connect_timeout` is what gives up on a dead host), and the Gemini providers (TTS, STT, vision) hand the SDK their own httpx client through `HttpOptions.httpx_async_client` rather than a per-request timeout, because google-genai flattens a per-request `httpx.Timeout` to its largest value (that client also keeps every SDK call on httpx when aiohttp is installed). The STT provider's Files API calls go through the SDK's classic path, which takes one value in milliseconds, so they carry the flat `timeout` and no connect split.

## Chain Depth Limit

Prevents infinite AI-to-AI loops. Default max depth is 5.

```python
kit = RoomKit(max_chain_depth=3)  # Stricter limit
```

When an AI response triggers another AI response, chain depth increments. Processing stops when the limit is reached.

What a turn sends into a room itself (`send_event()` or `process_inbound()`
from its channel, a tool handler or a `TurnHost` verb) is at least as deep as
the turn's response, so an agent that injects through another channel at each
turn stops at the limit too. Besides the depth, each cascade (an inbound
message or a direct injection) has a reentry budget of `10 ×
max_chain_depth` passes: its responses, its hooks' injected events, and what
the room's own work sends into the room (a hook, a parse, a delivery, a
turn). Past it, an answer or an injected event is stored `BLOCKED` with
`reentry_loop_cap`, and a message the room's work sends is refused
`reentry_loop_cap`, so a transport whose `deliver()` sends into its own room
stops too.

A speech-to-speech model (a `RealtimeVoiceChannel`, a conference's realtime
configuration) is held the same way. What its tool call's gate, handler and
hooks send into a room is at least as deep as the model's answer that issued
the call, and a text whose answer would reach the limit enters its session
silently, whatever brings it (a broadcast, `deliver()`, a handoff greeting,
`inject_text()`, whose text is at least as deep as the work it is called
from): the model takes it into its context and answers nothing. An `ON_TOOL_CALL` hook that posts at each call the model then
hears ends at the limit, and so does a hook on the model's words
(`ON_TRANSCRIPTION`), which runs at the depth of the answer it hears, and
one on its delegations (`ON_REALTIME_DELEGATION`), at the depth of the answer
that delegated (`examples/realtime_tool_observer_depth.py`).

A voice agent built from speech to text, an agent and text to speech (a
`VoiceChannel`, a conference's voice lane) is held the same way: the hooks on
what it says (`BEFORE_TTS`, on each sentence of a streamed reply too, and
`AFTER_TTS`) run at the depth of the answer they speak, so a hook that asks
the agent something after each of its answers ends at the limit
(`examples/hook_observer_depth.py`).

## Processing Timeout

`process_timeout` (seconds, default 30) is one deadline from the call's start
to its decision (RFC §13.6): routing, with the identity it reads, the router
and step 2's waits for the room lock, then the context build, the parse, the
off-lock checks and the locked gates. Past it, `process_inbound()` returns
`blocked=True, reason="process_timeout"` and `send_event()` raises
`ProcessTimeoutError`, nothing of the event written; the `process_timeout`
framework event names no room when the deadline passed while the message was
routed. A room step 2 creates is completed whatever the deadline, and step 2's
attach or record of a routed sender in a room that exists is bounded on its
wait for the lock only. The commit and everything after it (delivery,
answers, side effects) carry no deadline.

## Room Lifecycle Timers

Auto-transition rooms based on inactivity (RFC §5.1):

```python
from roomkit import RoomKit, RoomTimers

kit = RoomKit()

room = await kit.create_room(
    room_id="session-1",
    timers=RoomTimers(
        inactive_after_seconds=300,  # ACTIVE -> PAUSED after 5 min without activity
        closed_after_seconds=3600,  # PAUSED -> CLOSED 1 h after paused_at
    ),
)

# A room nobody writes to transitions when its timers are checked
room = await kit.check_room_timers("session-1")

# Batch check all rooms (call periodically)
transitioned = await kit.check_all_timers()
```

- **Activity** is the events the pipeline stores DELIVERED: each moves
  `timers.last_activity_at` in its commit. A record stored BLOCKED, a
  lifecycle record (an attach, a mute, a timer's transition), a
  `commit_event()` record and an instruction are no activity: a mute does not
  restart the inactivity timer. Inactivity counts from `last_activity_at`, or
  from `created_at` while the room has had none.
- **Pause and close.** An ACTIVE room pauses once `inactive_after_seconds`
  passed without activity, `paused_at` recording when; a PAUSED room closes
  once `closed_after_seconds` passed from `paused_at`. An ACTIVE room closes by
  timer only once it paused: to close N seconds after the last activity, set
  `inactive_after_seconds=N, closed_after_seconds=0`.
- **Resume.** A delivered event resumes a PAUSED room in its commit (ACTIVE,
  `paused_at` null), with no hook and no framework event.
- **When.** Every operation that reads the room's status under its lock (a
  message, `send_event()`, an agent's answer, an attach, a detach or a binding
  change, `commit_event()`, `update_event()`, `delete_event()`, `close_room()`,
  `archive_room()`) applies the transitions due at its instant first, so a
  message after a due close is refused as if the timer had run on time. A
  transition takes the instant it fell due, however late it is applied: a
  check long after both pauses the room at the first instant and closes it at
  the second, in one write. Each emits `room_paused` / `room_closed` (`reason =
  timer`) once stored, before the operation's own refusal (`room_closed` comes
  before the message's `room_refused_event`); its `ON_ROOM_PAUSED` /
  `ON_ROOM_CLOSED` hook fires once the room lock is released, its event's
  `SystemContent.data` the `threshold` and the instant (`at`).

## Delivery Status Tracking

Track whether messages were delivered to providers:

```python
from roomkit import DeliveryStatus

@kit.on_delivery_status
async def track(status: DeliveryStatus) -> None:
    if status.status == "failed":
        logger.error("Delivery failed: %s — %s", status.message_id, status.error_message)
    elif status.status == "delivered":
        logger.info("Delivered: %s", status.message_id)

# Process status webhooks from providers
await kit.process_delivery_status(status)
```

## Production Setup Example

```python
from roomkit import RoomKit
from roomkit.models import RateLimit, RetryPolicy
from roomkit.store import PostgresStore

kit = RoomKit(
    store=PostgresStore("postgresql://user:pass@localhost/roomkit"),
    max_chain_depth=5,
    inbound_rate_limit=RateLimit(max_per_second=50.0),
    process_timeout=30.0,
)

# Per-channel resilience
await kit.attach_channel("room", "sms-out",
    rate_limit=RateLimit(max_per_second=1.0, max_per_minute=30.0),
    retry_policy=RetryPolicy(max_retries=3, base_delay_seconds=1.0),
)
```

## Framework Events for Monitoring

```python
@kit.on("source_error")
async def on_error(event):
    logger.error("Source error: %s", event.data["error"])

@kit.on("voice_session_ended")
async def on_voice_end(event):
    logger.info("Voice session ended: %s", event.data["session_id"])
```
