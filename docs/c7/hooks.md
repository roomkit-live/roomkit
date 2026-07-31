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

### What happens when a hook fails

A hook that does not produce a usable result — it raises, it exceeds its
timeout, it returns something that is not a `HookResult`, or it returns a
`modify` whose payload is the wrong type for the trigger — is treated as
**allow**, and the error is logged. A broken hook must not be able to take a
room down, so the event goes through.

**Two triggers invert that**, because their payload is content a hook may
exist to withhold:

| Trigger | On hook failure |
|---|---|
| `BEFORE_TTS` | **Blocked** — the text is not synthesised |
| `ON_TRANSCRIPTION` | **Blocked** — the transcript is not published |
| every other sync trigger | Allowed, error logged |

All four failure modes block on those two, not just exceptions: a rule that
covered exceptions but allowed timeouts would leak through the timeout.

The consequence for `BEFORE_BROADCAST` is worth being explicit about, since
it is the trigger most often used for moderation: **a moderation hook that
crashes lets the content through.** That is deliberate (RFC §9.3), not an
oversight. Two levers exist if you need more, and their exact scope matters:

- The `hook_error` framework event is emitted by the **inbound pipeline's**
  `BEFORE_BROADCAST` pass only. Other sync-hook passes (`ON_TRANSCRIPTION`,
  `BEFORE_TTS`, the re-entry path) and every async trigger report failures
  in the log, not through `hook_error` — do not build monitoring for those
  on this event.
- To make additional triggers fail closed, extend the set the engine reads
  through its instance — RoomKit builds its own `HookEngine`, so there is
  no constructor to subclass into:

  ```python
  kit.hook_engine.FAIL_CLOSED_TRIGGERS = kit.hook_engine.FAIL_CLOSED_TRIGGERS | {
      HookTrigger.BEFORE_BROADCAST,
  }
  ```

  Weigh it first: fail-closed moderation means an outage in your hook is an
  outage of the room.

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

All 76 `HookTrigger` enum values (`src/roomkit/models/enums.py`). **Execution** is how
the engine invokes the trigger: SYNC triggers run through the sequential,
priority-ordered sync pipeline (can block/modify); ASYNC triggers run
concurrently, fire-and-forget, errors logged. Registration mode is forgiving in
both directions: ASYNC-registered hooks on a SYNC trigger fire as observers
after the sync pass; SYNC-registered hooks (the default) on an ASYNC trigger
fire like any other observer. Triggers marked *Reserved* exist in the enum but
are not fired by any built-in code.

### Event Pipeline

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `BEFORE_BROADCAST` | SYNC | `(event, ctx) -> HookResult` | Before event is stored and broadcast. Can block/modify. |
| `AFTER_BROADCAST` | ASYNC | `(event, ctx) -> None` | After event is broadcast. Fire-and-forget side effects. |

### Event Mutation (edit/delete)

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_EVENT_UPDATED` | ASYNC | `(event, ctx) -> None` | A persisted event's stored state changed — inbound `EditContent` or `kit.update_event()`. Payload is the updated event (`metadata.edited=True` on the edit path). |
| `ON_EVENT_DELETED` | ASYNC | `(event, ctx) -> None` | A persisted event was deleted — inbound `DeleteContent` (soft, `metadata.deleted=True`) or `kit.delete_event()` (hard; payload is the pre-delete snapshot). |

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

### Membership

Synthetic system events (`SystemContent`, `visibility=INTERNAL`), fired by the
member management API.

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_PARTICIPANT_JOINED` | ASYNC | `(event, ctx) -> None` | Member added via `kit.add_member()`. `content.data`: `participant_id`, `identity_id`. |
| `ON_PARTICIPANT_LEFT` | ASYNC | `(event, ctx) -> None` | Member removed via `kit.remove_member()` — soft status flip to LEFT/BANNED. `content.data`: `participant_id`, `status`. |
| `ON_PARTICIPANT_UPDATED` | ASYNC | `(event, ctx) -> None` | Member renamed via `kit.rename_member()` (display name only; identity never changes). |

### Delivery

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_DELIVERY_STATUS` | ASYNC | `(DeliveryStatus, ctx) -> None` | Provider delivery receipt dispatched by `kit.process_delivery_status()` / `kit.process_webhook()`. The `@kit.on_delivery_status` decorator registers a `(status)`-only callback on this trigger. |
| `BEFORE_DELIVER` | ASYNC | `(event, ctx) -> None` | Before a proactive `kit.deliver()` strategy executes (in-process and worker paths). Payload is a synthetic INTERNAL system-source event describing the delivery. Observational — invoked through the async pipeline, cannot block (RFC §9 lists it SYNC; code invokes it ASYNC). |
| `AFTER_DELIVER` | ASYNC | `(event, ctx) -> None` | After the delivery strategy completes. Same payload shape with status DELIVERED/FAILED and `metadata.error` set on failure. |

### Side Effects

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_TASK_CREATED` | ASYNC | `(event, ctx) -> None` | A `Task` returned in `HookResult.tasks` was persisted. Payload event has type `TASK_CREATED` and `metadata.task_id`/`task_title`. Fires once per persisted task. |
| `ON_ERROR` | ASYNC | `(event, ctx) -> None` | Error during processing (every provider/inference failure path funnels here). Payload event carries `metadata.error`, `error_type`, `error_category`. |

### Session Lifecycle

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_SESSION_STARTED` | ASYNC | `(SessionStartedEvent, ctx) -> None` | A session began: voice session bound, realtime session opened, conference bot connected (`event.session` = the session/bot), or first inbound on an auto-created text room. On the inbound path, internal (`_`-prefixed) hooks are awaited (greeting gate ordering) and user hooks fire in the background. Auto-greeting is an internal hook on this trigger. |

### Voice

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_SPEECH_START` | ASYNC | `(VoiceSession, ctx) -> None` | VAD detected speech start (VoiceChannel, RealtimeVoiceChannel). Conference lanes fire it per track with a synthetic system event. |
| `ON_SPEECH_END` | ASYNC | `(VoiceSession, ctx) -> None` | VAD detected speech end. Same per-lane conference behavior. |
| `ON_TRANSCRIPTION` | SYNC | `(event, ctx) -> HookResult` | STT produced a final transcript. Can block/modify; **fails closed** — a hook that raises/times out blocks publication. Payload: `TranscriptionEvent` (VoiceChannel), `RealtimeTranscriptionEvent` (RealtimeVoiceChannel), `ConferenceTranscription` (ConferenceChannel). Modify by returning the same type (voice/realtime also accept a plain `str`) with the new text. |
| `ON_PARTIAL_TRANSCRIPTION` | ASYNC | `(PartialTranscriptionEvent, ctx) -> None` | Streaming interim STT result. Hot path — skipped entirely when no hooks are registered. |
| `BEFORE_TTS` | SYNC | `(text: str, ctx) -> HookResult` | Before text is synthesized. Payload is the text `str`; modify with a `str`. Can block; **fails closed**. |
| `AFTER_TTS` | ASYNC | `(text: str, ctx) -> None` | After TTS audio was sent. Payload is the final synthesized text. |

### Voice Pipeline

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_VAD_SILENCE` | ASYNC | `(VADSilenceEvent, ctx) -> None` | VAD detected silence |
| `ON_VAD_AUDIO_LEVEL` | ASYNC | `(VADAudioLevelEvent, ctx) -> None` | Audio level update from VAD (high-frequency; telemetry-suppressed) |
| `ON_SPEAKER_CHANGE` | ASYNC | `(SpeakerChangeEvent, ctx) -> None` | Diarization detected speaker change |
| `ON_BARGE_IN` | ASYNC | `(BargeInEvent, ctx) -> None` | User interrupted TTS playback; carries `interrupted_text`, `audio_position_ms` |
| `ON_TTS_CANCELLED` | ASYNC | `(TTSCancelledEvent, ctx) -> None` | TTS playback was cancelled (barge-in or explicit interrupt) |
| `ON_DTMF` | ASYNC | `(DTMFDetectedEvent, ctx) -> None` | DTMF tone detected |
| `ON_TURN_COMPLETE` | ASYNC | `(TurnCompleteEvent, ctx) -> None` | Turn detector says turn is complete; carries combined text + confidence |
| `ON_TURN_INCOMPLETE` | ASYNC | `(TurnIncompleteEvent, ctx) -> None` | Turn detector says turn is incomplete |
| `ON_BACKCHANNEL` | ASYNC | `(BackchannelEvent, ctx) -> None` | Backchannel detected (uh-huh, yeah) |
| `ON_RECORDING_STARTED` | ASYNC | `(RecordingStartedEvent, ctx) -> None` | Audio recording started (voice session or conference track) |
| `ON_RECORDING_STOPPED` | ASYNC | `(RecordingStoppedEvent, ctx) -> None` | Audio recording stopped, result available |
| `ON_INPUT_AUDIO_LEVEL` | ASYNC | `(AudioLevelEvent, ctx) -> None` | Inbound audio level (throttled; telemetry-suppressed) |
| `ON_OUTPUT_AUDIO_LEVEL` | ASYNC | `(AudioLevelEvent, ctx) -> None` | Outbound audio level (throttled; telemetry-suppressed) |
| `BEFORE_BRIDGE_AUDIO` | SYNC | `(BridgeAudioEvent, ctx) -> HookResult` | Before an audio frame is forwarded across an audio bridge. Can block/modify the frame. Only invoked when hooks are registered — otherwise frames bypass the event loop for latency. |

### Tool Execution

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `BEFORE_TOOL_USE` | SYNC | `(ToolCallEvent, ctx) -> HookResult` | Before a tool executes (AIChannel, realtime channels, external/ACP tools). Block denies the call. Fails closed: if room context cannot be built, the call is denied. |
| `ON_TOOL_CALL` | SYNC | `(ToolCallEvent, ctx) -> HookResult` | A tool call executed (AIChannel, RealtimeVoiceChannel, skills, external/ACP tools). Block returns an error result to the model; `HookResult(metadata={"result": ...})` supplies or overrides the tool result (`event.result is None` means the hook must provide it). RFC §9 lists this ASYNC; code invokes it through the sync pipeline. |
| `ON_USER_INPUT_REQUIRED` | SYNC | `(PendingInputEvent, ctx) -> HookResult` | A `HumanInputHandler`-backed tool paused awaiting human input. Sync so the notification (e.g. WebSocket push) lands before `wait()` starts blocking; block auto-rejects the pending request. |

### Realtime Voice

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_REALTIME_TEXT_INJECTED` | ASYNC | `(event, ctx) -> None` | Text was injected into realtime session |

### AI Generation

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `BEFORE_AI_GENERATION` | SYNC | `(AIGenerationEvent, ctx) -> HookResult` | Before the AI provider is invoked. `event.ai_context` (messages, system prompt, tools) may be mutated in place; block skips generation. |
| `ON_AI_THINKING` | — | — | Reserved — defined in the enum, not fired by any built-in code (RFC §9 marks it Implemented). |
| `ON_AI_RESPONSE` | ASYNC | `(AIResponseEvent, ctx) -> None` | AI generation completed. Carries response content, usage, latency, tool call count. |

### Protocol Observability

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_PROTOCOL_TRACE` | ASYNC | `(ProtocolTrace, ctx) -> None` | Transport-level protocol trace (SIP, RTP, …) forwarded from a channel. Traces arriving before the room exists are buffered and replayed on attach. |

### Orchestration (multi-agent)

Handoff triggers fire from the `HandoffCoordinator` with a synthetic INTERNAL
system event (AI-channel source; `metadata`: `from_agent`, `to_agent`,
`accepted`, `new_phase`).

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_HANDOFF` | ASYNC | `(event, ctx) -> None` | Agent handoff accepted and executed. |
| `ON_HANDOFF_REJECTED` | ASYNC | `(event, ctx) -> None` | Agent handoff was rejected (`metadata.accepted=False`). |
| `ON_PHASE_TRANSITION` | ASYNC | `(event, ctx) -> None` | Fired alongside `ON_HANDOFF` after an accepted handoff, carrying the new phase. |
| `ON_STATUS_POSTED` | — | — | Reserved — StatusBus posts emit the `status_posted` framework event instead; no hook fires (RFC §9 marks it Implemented). |

### Delegation (background tasks)

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_TASK_DELEGATED` | ASYNC | `(event, ctx) -> None` | `kit.delegate()` dispatched a task to a child room. INTERNAL event, type `TASK_DELEGATED`; `metadata`: `task_id`, `child_room_id`, `agent_id`. |
| `ON_TASK_COMPLETED` | ASYNC | `(event, ctx) -> None` | A delegated task finished. Fires in the parent room; type `TASK_COMPLETED`, body = output or error. |

### Video

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `BEFORE_BRIDGE_VIDEO` | SYNC | `(BridgeVideoEvent, ctx) -> HookResult` | Before a video frame is forwarded across a bridge. Can block/modify the frame. Only invoked when hooks are registered (fast path bypasses the event loop). |
| `ON_VIDEO_SESSION_STARTED` | ASYNC | `(SessionStartedEvent, ctx) -> None` | Video path live (VideoChannel, AVChannel, RealtimeAVChannel). `event.session` is the Video/VoiceSession. |
| `ON_VIDEO_SESSION_ENDED` | ASYNC | `(SessionStartedEvent, ctx) -> None` | Video session ended (same payload shape). |
| `ON_VIDEO_TRACK_ADDED` | — | — | Reserved — defined in the enum, not fired by built-in channels. |
| `ON_VIDEO_TRACK_REMOVED` | — | — | Reserved — defined in the enum, not fired by built-in channels. |
| `ON_VISION_RESULT` | SYNC | `(VisionEvent, ctx) -> HookResult` | A VisionProvider analyzed a frame. Block discards the result; modify rewrites the description before event injection and AI context update. RFC §9 lists this ASYNC; code invokes it through the sync pipeline. |
| `ON_SCREEN_SHARE_STARTED` | ASYNC | `(event, ctx) -> None` | ConferenceChannel: a SCREEN_SHARE track was published. `content.data`: `track_id`, `participant_id`. |
| `ON_SCREEN_SHARE_STOPPED` | ASYNC | `(event, ctx) -> None` | ConferenceChannel: a SCREEN_SHARE track was unpublished. |
| `ON_VIDEO_DETECTION` | ASYNC | `(VideoDetectionEvent, ctx) -> None` | Video pipeline filter emitted a detection event (object, face, …). |

### Conference (SFU)

All fired by `ConferenceChannel` as synthetic system events
(`SystemContent` with a `data` dict including `channel_id`); the channel's own
bot never triggers them.

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_CONFERENCE_PARTICIPANT_JOINED` | ASYNC | `(event, ctx) -> None` | Participant joined the conference media session. `data`: `participant_id`. |
| `ON_CONFERENCE_PARTICIPANT_LEFT` | ASYNC | `(event, ctx) -> None` | Participant left the conference media session. |
| `ON_CONFERENCE_TRACK_PUBLISHED` | ASYNC | `(event, ctx) -> None` | Participant published a track. `data`: `track_id`, `participant_id`, `kind` (audio/video/screen_share). |
| `ON_CONFERENCE_TRACK_UNPUBLISHED` | ASYNC | `(event, ctx) -> None` | A track was unpublished (same `data` shape). |
| `ON_CONFERENCE_TRACK_MUTED` | ASYNC | `(event, ctx) -> None` | Publisher muted a track — "camera off" is usually a muted VIDEO track, not an unpublish. |
| `ON_CONFERENCE_TRACK_UNMUTED` | ASYNC | `(event, ctx) -> None` | Publisher unmuted a track. |
| `ON_ACTIVE_SPEAKER_CHANGED` | ASYNC | `(event, ctx) -> None` | SFU reported a dominant-speaker change. `data`: `participant_id`. |
| `ON_CONNECTION_QUALITY_CHANGED` | ASYNC | `(event, ctx) -> None` | SFU reported a participant's connection quality. `data`: `participant_id`, `quality`. |

### Planning

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_PLAN_UPDATED` | — | — | Reserved — not fired by any built-in code. The `plan_tasks` tool (`enable_planning=True`) publishes an ephemeral realtime event `{"type": "plan_updated"}` instead (RFC §9 marks the hook Implemented). |

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
