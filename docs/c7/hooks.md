# Hooks

Hooks intercept events at specific points in the processing pipeline. They can block messages, modify content, trigger side effects, or observe the conversation.

## Hook Basics

```python
from roomkit import RoomKit, HookTrigger, HookExecution, HookResult, RoomEvent, RoomContext

kit = RoomKit()

# Sync hook: runs BEFORE broadcast, can block or modify
@kit.hook(HookTrigger.BEFORE_BROADCAST)
async def content_filter(event: RoomEvent, ctx: RoomContext) -> HookResult:
    if "spam" in event.content.body.lower():
        return HookResult.block("Spam detected")
    return HookResult.allow()

# Async hook: runs AFTER broadcast, fire-and-forget
@kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC)
async def log_event(event: RoomEvent, ctx: RoomContext) -> None:
    await analytics.track("message", {"room": event.room_id})
```

## HookResult

Sync hooks (BEFORE_BROADCAST) must return a `HookResult`:

```python
from roomkit import HookResult, TextContent

# Allow the event to proceed
HookResult.allow()

# Block the event with a reason
HookResult.block("Contains prohibited content")

# Modify the event before broadcast
modified = event.model_copy(update={"content": TextContent(body="[REDACTED]")})
HookResult.modify(modified)
```

## Hook Priority

Lower priority numbers run first. Default is 0.

```python
# Runs first (priority=0)
@kit.hook(HookTrigger.BEFORE_BROADCAST, name="profanity_filter", priority=0)
async def profanity_filter(event: RoomEvent, ctx: RoomContext) -> HookResult:
    blocked_words = {"badword", "spam", "scam"}
    if isinstance(event.content, TextContent):
        words = set(event.content.body.lower().split())
        if words & blocked_words:
            return HookResult.block(f"Blocked: {words & blocked_words}")
    return HookResult.allow()

# Runs second (priority=1)
@kit.hook(HookTrigger.BEFORE_BROADCAST, name="pii_redactor", priority=1)
async def pii_redactor(event: RoomEvent, ctx: RoomContext) -> HookResult:
    import re
    if isinstance(event.content, TextContent):
        redacted = re.sub(
            r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[REDACTED]", event.content.body
        )
        if redacted != event.content.body:
            modified = event.model_copy(update={"content": TextContent(body=redacted)})
            return HookResult.modify(modified)
    return HookResult.allow()
```

## Hook Filters

Filter hooks by channel type, channel ID, or direction:

```python
from roomkit.models.enums import ChannelType, ChannelDirection

@kit.hook(
    HookTrigger.AFTER_BROADCAST,
    execution=HookExecution.ASYNC,
    channel_types={ChannelType.SMS},
    directions={ChannelDirection.INBOUND},
    priority=10,
)
async def sms_audit(event: RoomEvent, ctx: RoomContext) -> None:
    await audit_log.record(event)
```

## Room-Scoped Hooks

Add hooks to specific rooms instead of globally:

```python
from roomkit import HookExecution

await kit.add_room_hook(
    room_id="vip-room",
    trigger=HookTrigger.BEFORE_BROADCAST,
    execution=HookExecution.SYNC,
    fn=my_hook_function,
    name="vip_filter",
)

# Remove later
await kit.remove_room_hook("vip-room", "vip_filter")
```

## Complete Hook Trigger Reference

### Event Pipeline

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `BEFORE_BROADCAST` | SYNC | `(event, ctx) -> HookResult` | Before event is stored and broadcast. Can block/modify. |
| `AFTER_BROADCAST` | ASYNC | `(event, ctx) -> None` | After event is broadcast. Fire-and-forget side effects. |

### Channel Lifecycle

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_CHANNEL_ATTACHED` | ASYNC | `(event, ctx) -> None` | Channel was attached to a room |
| `ON_CHANNEL_DETACHED` | ASYNC | `(event, ctx) -> None` | Channel was detached from a room |
| `ON_CHANNEL_MUTED` | ASYNC | `(event, ctx) -> None` | Channel was muted in a room |
| `ON_CHANNEL_UNMUTED` | ASYNC | `(event, ctx) -> None` | Channel was unmuted in a room |

### Room Lifecycle

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_ROOM_CREATED` | ASYNC | `(event, ctx) -> None` | Room was created |
| `ON_ROOM_PAUSED` | ASYNC | `(event, ctx) -> None` | Room was paused |
| `ON_ROOM_CLOSED` | ASYNC | `(event, ctx) -> None` | Room was closed |

### Identity

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_IDENTITY_AMBIGUOUS` | SYNC | `(event, ctx) -> IdentityHookResult` | Multiple identity matches found |
| `ON_IDENTITY_UNKNOWN` | SYNC | `(event, ctx) -> IdentityHookResult` | No identity match found |
| `ON_PARTICIPANT_IDENTIFIED` | ASYNC | `(event, ctx) -> None` | Participant was successfully identified |

### Delivery

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_DELIVERY_STATUS` | ASYNC | `(status) -> None` | Delivery status update from provider |

### Side Effects

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_TASK_CREATED` | ASYNC | `(event, ctx) -> None` | AI extracted a task from conversation |
| `ON_ERROR` | ASYNC | `(event, ctx) -> None` | Error occurred during processing |

### Voice

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_SPEECH_START` | ASYNC | `(event, ctx) -> None` | VAD detected speech start |
| `ON_SPEECH_END` | ASYNC | `(event, ctx) -> None` | VAD detected speech end |
| `ON_TRANSCRIPTION` | ASYNC | `(event, ctx) -> None` | STT produced a transcription |
| `BEFORE_TTS` | SYNC | `(event, ctx) -> HookResult` | Before text is sent to TTS. Can block/modify. |
| `AFTER_TTS` | ASYNC | `(event, ctx) -> None` | After TTS audio is generated |

### Voice Pipeline

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_VAD_SILENCE` | ASYNC | `(event, ctx) -> None` | VAD detected silence |
| `ON_VAD_AUDIO_LEVEL` | ASYNC | `(event, ctx) -> None` | Audio level update from VAD |
| `ON_SPEAKER_CHANGE` | ASYNC | `(event, ctx) -> None` | Diarization detected speaker change |
| `ON_BARGE_IN` | ASYNC | `(event, ctx) -> None` | User interrupted TTS playback |
| `ON_TTS_CANCELLED` | ASYNC | `(event, ctx) -> None` | TTS playback was cancelled |
| `ON_DTMF` | ASYNC | `(event, ctx) -> None` | DTMF tone detected |
| `ON_TURN_COMPLETE` | ASYNC | `(event, ctx) -> None` | Turn detector says turn is complete |
| `ON_TURN_INCOMPLETE` | ASYNC | `(event, ctx) -> None` | Turn detector says turn is incomplete |
| `ON_BACKCHANNEL` | ASYNC | `(event, ctx) -> None` | Backchannel detected (uh-huh, yeah) |
| `ON_RECORDING_STARTED` | ASYNC | `(event, ctx) -> None` | Audio recording started |
| `ON_RECORDING_STOPPED` | ASYNC | `(event, ctx) -> None` | Audio recording stopped |
| `ON_INPUT_AUDIO_LEVEL` | ASYNC | `(event, ctx) -> None` | Input audio level update |
| `ON_OUTPUT_AUDIO_LEVEL` | ASYNC | `(event, ctx) -> None` | Output audio level update |

### Tool Execution

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_TOOL_CALL` | ASYNC | `(event, ctx) -> None` | AI invoked a tool (fires from AIChannel and RealtimeVoiceChannel) |

### Realtime Voice

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_REALTIME_TEXT_INJECTED` | ASYNC | `(event, ctx) -> None` | Text was injected into realtime session |

### AI Response

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_AI_RESPONSE` | ASYNC | `(AIResponseEvent, ctx) -> None` | AI generation completed. Carries response content, usage, latency, tool call count. |

### Planning

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_PLAN_UPDATED` | ASYNC | `(event, ctx) -> None` | AI updated its task plan via `_plan_tasks` tool (requires `enable_planning=True`) |

### Feedback

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_FEEDBACK` | ASYNC | `(Observation, ctx) -> None` | User submitted quality feedback via `kit.submit_feedback()` |

## Framework Events

Framework events are lightweight lifecycle notifications (not message events):

```python
@kit.on("room_created")
async def on_room_created(event):
    print(f"Room created: {event.data['room_id']}")

@kit.on("voice_session_started")
async def on_voice(event):
    print(f"Voice session: {event.data['session_id']}")
```

Available framework event types: `room_created`, `room_closed`, `room_paused`, `room_channel_attached`, `room_channel_detached`, `channel_connected`, `channel_disconnected`, `voice_session_started`, `voice_session_ended`, `source_attached`, `source_detached`, `source_error`, `source_exhausted`.
