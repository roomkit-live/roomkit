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

**Hooks of a turn.** The triggers a turn fires (`BEFORE_AI_GENERATION`,
`ON_AI_THINKING`, `ON_AI_RESPONSE`, `BEFORE_TOOL_USE`, `AFTER_TOOL_ROUND`,
`ON_TOOL_CALL`, `ON_USER_INPUT_REQUIRED`, `ON_SPEAK_DECISION`, `ON_THOUGHT`,
`ON_PLAN_UPDATED`) fire on the turn executor that runs the turn (RFC §10.6.6).
Registered on the kit, such a hook is the kit's own executor's: it fires for
the agents the kit runs, and for realtime voice and conference tool calls.

```python
@kit.hook(HookTrigger.BEFORE_TOOL_USE, name="tool_gate")
async def tool_gate(event, ctx) -> HookResult:
    return HookResult.allow()
```

The channel filters read the turn's agent: `channel_ids={"support"}` keeps a
turn hook to that agent's turns, and `channel_types` to agents of a type when
the trigger's payload carries it (a tool call, an input request).

An agent the kit holds by its declaration alone
(`kit.register_channel(ChannelDeclaration(...))`) runs its turns on another
executor, where a hook on the kit never fires. So the kit refuses, with a
`ValueError` naming the executor-side call, a turn hook such an agent could
fire (`channel_ids` can leave it out), and such a declaration while such a
hook is registered; a room hook on a turn trigger is refused while the kit
holds any declared agent. Register those hooks on the executor that runs the
agent, in one function you run on every executor:

```python
def register_turn_hooks(executor):
    @executor.hook(HookTrigger.BEFORE_TOOL_USE, name="tool_gate")
    async def tool_gate(event, ctx) -> HookResult:
        return HookResult.allow()

register_turn_hooks(host_executor)
```

`kit.hooks_for(trigger)` answers the engine a trigger's hooks register on, for
code that registers on triggers of both sides (`roomkit.TURN_TRIGGERS` lists
the turn's). An executor you build fires them too: it wires each channel its
factory builds, as the kit wires the ones it registers, and gives its hooks a
room context read from the store you give it:

```python
executor = TurnExecutor(build_agent, store=store)  # its hooks read the shared store
register_turn_hooks(executor)
```

`await kit.add_room_hook(room_id, trigger, ...)` on one of these triggers is the
kit's own executor's hook for that room. For agents other executors run, a
room hook is defined by name on the executors and enabled per room on the
kit, which names a room's hooks in each turn's request; only an executor that
defines every name takes the room's turns, so define it on each executor
that runs the room's agents, the kit's own included when the room holds an
agent the kit runs:

```python
async def audit(event, ctx) -> None: ...

for executor in (kit.turn_executor, host_executor):
    executor.room_hook("audit", HookTrigger.ON_TOOL_CALL, HookExecution.ASYNC)(audit)

await kit.enable_room_hook("vip-room", "audit")
await kit.disable_room_hook("vip-room", "audit")
```

On an executor a name carries one hook per trigger: `room_hook()` called again
with a name and a trigger it carries replaces the first. The name a room
enables joins the room's hooks, where a name is unique: `enable_room_hook()`
raises `ValueError` for a name a room hook the kit registered for that room
holds, and `add_room_hook()` for a name the room enables.

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

### Injected events

A `BEFORE_BROADCAST` hook can inject events whatever it decides about the
event it reads: a notice to a moderator when it blocks a message, say.

```python
from roomkit import ChannelType, EventSource, HookResult, InjectedEvent, RoomEvent, TextContent

@kit.hook(HookTrigger.BEFORE_BROADCAST)
async def moderate(event, ctx):
    if event.source.channel_id == "moderation":
        return HookResult.allow()  # its own notice runs this hook too
    if not looks_like_spam(event):
        return HookResult.allow()
    notice = RoomEvent(
        room_id=event.room_id,
        source=EventSource(channel_id="moderation", channel_type=ChannelType.SYSTEM),
        content=TextContent(body="A message was held for review."),
    )
    return HookResult.block(
        "spam", injected=[InjectedEvent(event=notice, target_channel_ids=["ws-moderator"])]
    )
```

Each injected event is committed after the event that caused it, as a pass
of its own (RFC §9.4):

- It commits once its cause's locked section has released the room lock, in a
  section of its own, so another event may commit between them (a message
  waiting for the lock, an agent's answer to the cause). It still commits
  before `process_inbound()` or `send_event()` returns, in the order the hooks
  ran.
- It runs the `BEFORE_BROADCAST` hooks, the one that injected it included, so
  a hook that blocks or injects on every event must let its own injections
  through. A hook may block it (stored `BLOCKED`), and what its hooks inject
  is committed after it the same way.
- It counts against the reentry budget (`10 × max_chain_depth` per inbound
  message or direct injection, shared with what the room's own work sends
  into the room, below). Past it, it is stored `BLOCKED` with
  `reentry_loop_cap` and its hooks do not run.
- It keeps its cause's chain depth and carries no idempotency key. Its
  visibility names the targets the room binds (`none` when it has none), and
  a hook may narrow it, never widen it.
- It takes no source check: it is the hook's, not a channel's.
- It is delivered to its transport targets only, transcoded like any
  delivery, never to its own source. An agent among the targets is not
  called: it reads the event in its history at its next turn. A realtime
  voice channel, or a conference with a realtime model, among the targets
  takes its text into the model's context silently and asks no answer. To
  have an agent answer, send the event with `kit.send_event()`.
- Its delivery set fires `AFTER_BROADCAST` like any other event's, an empty
  one when it is stored only.
- An injected `INSTRUCTION` is dropped with a warning before any pass: no
  hook runs on it, nothing of it is stored or delivered, and the rest of the
  result stands. Only the application issues instructions; a hook that wants
  an agent directed calls `kit.send_event(..., event_type=EventType.INSTRUCTION,
  addressed_to=[agent_id])`.

An identity challenge (`IdentityHookResult.challenge(inject=...)`) is
committed the same way. No other trigger commits injected events: a SYNC hook
on `BEFORE_DELIVER`, `ON_VERIFICATION_REQUIRED` or any other trigger that
returns some has them dropped, with a warning naming the hook and the trigger.

### Calling into the room from a hook

A hook may call `kit.send_event()` or `kit.process_inbound()` into the room of
the event it runs for. The call is part of that event's work (RFC §9.5.1):

- From a locked hook (or any code under the room lock), it is **nested**: it
  takes no admission ticket, runs every `BEFORE_BROADCAST` hook under the lock
  and commits ahead of the event being processed. What its own hooks inject
  joins the queue of the section holding the lock, committed once that
  section has released it, after the event being processed.
- A task a locked hook starts acts for the hook while the hook runs, awaited
  or not: its call is nested too. Once the hook has returned, the task holds
  nothing of the lock, though the event's pass still holds it for its next
  hook: its calls are new arrivals, which wait for the lock and commit after
  the event. A later hook of the same pass that awaits that task waits with
  it, until its own timeout (which cancels the task it awaited, unless it
  shields it). Await work in the hook that starts it, or leave it to run
  after the event.
- From an off-lock hook's body (`needs_lock=False`), or a `hook_error`
  handler reporting that check's failure, it is **adopted beside** the event:
  no ticket, every hook under the lock, its own section for its commit,
  still ahead of the event. A task the body starts stands beside the event
  only while the body runs: once the body has returned, while the check's
  next hook runs, its calls are new arrivals, which take a ticket and commit
  after the event.
- Either way it answers at its commit (a `process_inbound()` result carries
  no `delivery_results`). Made under the lock, its delivery waits in the
  room's lane behind the work that made it, and `result.delivery.wait()`
  there returns at once, unwaited. Made in an event's admission work (an
  off-lock hook's body, a `hook_error` handler, a channel's
  `handle_inbound()`), its delivery runs at its turn in the lane, during that
  work and ahead of the event, so a `wait()` from the work returns with its
  `delivery_results`; one whose delivery calls into the room and needs the
  event's admission ticket (a transport's `deliver()` that sends into it)
  ends at the event's `process_timeout`, the event refused. It draws on the
  event's reentry budget, one pass as it starts:
  a hook that injects on every event stops at the cap. Past it, `send_event()`
  stores the event `BLOCKED` with `reentry_loop_cap`, no hook run, and returns
  that record; `process_inbound()` naming the room returns `blocked=True,
  reason="reentry_loop_cap"` with no event, before anything else: its channel
  is not looked up, nothing is attached. A call acting for an organization
  has the room read with it first, so another organization's room raises
  `RoomNotFoundError` and draws nothing.
- A message routed back into the room (`process_inbound()` with no
  `room_id`) raises `ReentrantCallError` before anything is written: its room
  is known only once routed, so it cannot be adopted. Name the room instead.

Once a hook ran under the lock, the room is read again before the event
commits (RFC §10.1 step 9a), and the event is planned with the room as its
hooks left it: a message a hook deleted or redacted (`kit.delete_event()`,
`kit.update_event()`) is not handed to an agent as it was, and a channel a
hook detached receives nothing, its record stored or not. The event takes the
room's next index: what the hooks committed ahead of it took the one they were
shown, so an instruction, never stored, reports the index after it, allowed or
blocked. A room a hook closed
(`kit.close_room()`) refuses the event as a closed room does: `blocked=True,
reason="room_closed"` (`send_event()` raises `RoomClosedError`), nothing of it
stored, what its hooks returned dropped, and `room_refused_event` emitted. An
agent's response and an injected event whose hook closed the room are refused
the same way (`operation = reentry`).

A channel's `handle_inbound()` stands in its message's admission order the
same way: what it sends into its room is adopted beside the message, on its
budget, and a message it routes back into that room raises
`ReentrantCallError`. A call from a delivery set's work (a transport's
`deliver()`, an agent's turn, a tool handler), or from the body of an ASYNC
hook the cascade fires once an event's delivery set has run (`AFTER_BROADCAST`,
a change's `ON_EVENT_UPDATED` or `ON_EVENT_DELETED`) or of an ASYNC hook
observing a SYNC trigger the room's work fires (an ASYNC hook on
`BEFORE_BROADCAST`, `BEFORE_DELIVER`, `BEFORE_TTS`...; not one that fires per
tool call, below), is a new arrival with its
own ticket and `process_timeout`, but it draws on the budget of the cascade
that delivers while that work runs (a hook's body on the cascade's hook
calls, a counter of the same size kept apart from its passes), and a call
made while a turn or a tool call runs (a realtime or a conference call
included), by it or a task it spawned, into whatever room, is at least as
deep as the turn's response or the call's, so a transport, an agent or an
ASYNC hook that sends into its room every time it is reached stops at the cap
or at `max_chain_depth`. That depth holds where the owner runs the turn's
code (an in-memory turn calling the kit it holds, a tool handler, a
`TurnHost` verb);
with a Redis turn backend a turn reaches the room through its `TurnHost`. A
call of such work that creates a room, routed or named (the router answering
none or naming a room the store lacks, `process_inbound(..., room_id=...)`
naming one), is that work's own and draws on the same budget, charged once
before the room is created, so a delivery that opens a room each time stops at
the cap. A call into another existing room has a budget of its own,
and work a turn, a tool call, a delivery or an ASYNC hook's body began starts
fresh once that work is over: depth 0 and a budget of its own.

```python
@kit.hook(HookTrigger.BEFORE_BROADCAST)
async def acknowledge(event, ctx):
    if event.source.channel_id == "receipts":
        return HookResult.allow()  # its own receipt runs this hook too
    # Nested: committed ahead of the event, within this event's budget.
    await kit.send_event(event.room_id, "receipts", TextContent(body="Received."))
    return HookResult.allow()
```

A cascade counts the calls of its ASYNC hooks' bodies, the observers of its
SYNC triggers included, apart from its own passes (its answers, chained
streams and injected events), on a counter of the
same size, `10 × max_chain_depth` (50 by default), shared by all the cascade's
hooks together; and a call a hook's body makes is at least one deeper than the
event that fired the hook, as a call made within a turn is at least as deep as
its response:

- A hook that posts after every event, its own posts included, stops at the
  cap: each post draws one hook call, and past the cap it is stored `BLOCKED`
  with `reentry_loop_cap`.
- A call refused for its organization (`organization_id=` naming another
  organization's room) raises `RoomNotFoundError` before it draws anything:
  `process_inbound()` naming the room as much as `post()`, `send_event()`,
  `deliver()` or a regeneration.
- A hook that asks an agent again after each of its answers (a judge) ends at
  `max_chain_depth`, or at a discussion's `max_depth`: each post, message,
  `deliver()` (a queued item carries the depth) or regeneration is one deeper
  than the answer, about `max_chain_depth / 2` answers in all (3 at 6).
- A hook on a turn's trigger (`BEFORE_AI_GENERATION`, `ON_AI_RESPONSE`, a
  tool call's), on a speech-to-speech model's words or delegations, or on
  what a voice agent says (`BEFORE_TTS`, `AFTER_TTS`), observes no event of a
  depth of its own: its body sends at least as deep as the turn, the tool
  call or the answer was when the trigger fired, for as long as the body
  runs, once that work is over too. A reminder it posts at each turn ends at
  `max_chain_depth` (`examples/hook_observer_depth.py`).
- A notice posted once per event draws one hook call per event: it costs the
  agents no answer, and their answers block no notice. Only past 50 calls of
  all the cascade's hooks together (a streamed turn of more rows than that, a
  notice per row) is a notice blocked.
- A direct `kit.update_event()`, `kit.delete_event()` or
  `kit.regenerate_response()` made from the body draws one hook call too, and
  past the cap raises `ReentryLoopCapError`, nothing written; the engine logs
  one it lets escape as a warning. The change's own hooks (`ON_EVENT_UPDATED`,
  `ON_EVENT_DELETED`, a regeneration's answers) run on the budget of the work
  it was made in, one of their own when it came from outside the room's work,
  so a hook that edits, deletes or regenerates on every change stops there.
  Bulk work from one body stops at the cap too (15 deletes against 10: the
  last 5 refused).
- A task the body starts and does not await begins fresh once the body has
  returned: at depth 0, whether the turn or the other hooks of the same
  trigger still run or not, and on a budget of its own once the room's
  delivery it was fired from is over (while it runs, a call into that room
  draws on the delivery's budget, as any of its work does). A quality gate
  that relaunches the agent from such a task relaunches it at depth 0 each
  time, bounded by its own attempts, or by the first delivery's reentry cap
  when every relaunch is sent while a turn still runs (about `10 ×
  max_chain_depth` relaunches) (`examples/hook_observer_depth.py`). A
  task the body awaits, or gathers, stays on the budget and the depth while
  the body runs.

Work that follows an agent's answer outside the answer's cascade, and bulk work
of more than 50 changes, runs in such a task. A post or a `send_event()` from
it passes the chain it continues, `chain_depth`, so that its round trips with
the agent end at `max_chain_depth`. `update_event()`, `delete_event()` and
`regenerate_response()` take no `chain_depth`: made from a task, nothing of
the framework bounds them, and the hook bounds the changes it makes there
itself:

```python
background: set[asyncio.Task] = set()

@kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC)
async def ask_again(event, ctx):
    if event.source.channel_id != "assistant":
        return
    # Fresh once this body returns: chain_depth keeps it in the answer's chain.
    task = asyncio.create_task(kit.post(
        event.room_id, "Check your answer.", addressed_to=["assistant"],
        chain_depth=event.chain_depth + 1,
    ))
    background.add(task)
    task.add_done_callback(background.discard)
```

The visibility `internal` marks a record the framework wrote, and only the
framework writes it: a `BEFORE_BROADCAST` modify that writes it, or that changes
it on a record that holds it (a voice channel's interruption record), has the
event's visibility put back. A modify may narrow an event's
`response_visibility`, never widen or clear it: what it writes is intersected
with the original, `None` keeps the original and `internal` reads as `none`.

### What happens when a hook fails

A hook that does not produce a usable result — it raises, it exceeds its
timeout, it returns something that is not a `HookResult`, it returns a
`modify` whose payload is the wrong type for the trigger, or its replacement
or an event it injects holds more than five composite levels (RFC §5.3) — is
treated as **allow**, and the error is logged. A broken hook must not be able to take a
room down, so the event goes through.

**Three triggers invert that**, because their payload is content a hook may
exist to withhold, or an action it may exist to prevent:

| Trigger | On hook failure |
|---|---|
| `BEFORE_TTS` | **Blocked** — the text is not synthesised |
| `ON_TRANSCRIPTION` | **Blocked** — the transcript is not published |
| `BEFORE_TOOL_USE` | **Refused** — the call does not run; the model reads `Tool 'x' denied by pre-execution hook.`, and the hook's error reaches ON_TOOL_CALL's observers on `error_detail` (an external tool handler decides and reports the call itself) |
| every other sync trigger | Allowed, error logged |

All four failure modes block on those three, not just exceptions: a rule that
covered exceptions but allowed timeouts would leak through the timeout.

The consequence for `BEFORE_BROADCAST` is worth being explicit about, since
it is the trigger most often used for moderation: **a moderation hook that
crashes lets the content through.** That is deliberate (RFC §9.3), not an
oversight. Two levers exist if you need more, and their exact scope matters:

- The `hook_error` framework event is emitted by the `BEFORE_BROADCAST`
  passes only: an inbound message or a direct injection, an agent's answer,
  a hook's injected event, a streamed row. Other sync triggers
  (`ON_TRANSCRIPTION`, `BEFORE_TTS`...) and every async trigger report
  failures in the log, not through `hook_error` — do not build monitoring
  for those on this event. An off-lock hook (`needs_lock=False`) that fails
  is reported as it fails, before the room lock, whatever the gates under
  the lock then decide (a duplicate, a room closed meanwhile).
- To make additional triggers fail closed, extend the set the engine reads
  through its instance — RoomKit builds its own `HookEngine`, so there is
  no constructor to subclass into:

  ```python
  kit.hook_engine.FAIL_CLOSED_TRIGGERS = kit.hook_engine.FAIL_CLOSED_TRIGGERS | {
      HookTrigger.BEFORE_BROADCAST,
  }
  ```

  A trigger a turn fires (`BEFORE_TOOL_USE`, `ON_TOOL_CALL`...) set there
  goes to the kit's own executor; an executor you build has its own set,
  `executor.hooks.FAIL_CLOSED_TRIGGERS`.

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
from roomkit.models import ChannelType, EventDirection

@kit.hook(
    HookTrigger.AFTER_BROADCAST,
    execution=HookExecution.ASYNC,
    channel_types={ChannelType.SMS},
    directions={EventDirection.INBOUND},
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

`add_room_hook()`, `remove_room_hook()`, `enable_room_hook()` and
`disable_room_hook()` take `organization_id=` (RFC §17.2): the room is read
with it first, and another organization's room raises `RoomNotFoundError`
with nothing registered, removed, enabled or disabled. `kit.hook_engine`'s
own `add_room_hook()` / `remove_room_hook()` are the synchronous engine
calls: they read no room.

A trigger a turn fires takes no room hook by callable: define it by name on
the executors (`executor.room_hook(name, trigger)(fn)`) and enable the name
for the room (`await kit.enable_room_hook(room_id, name)`), as above.

## Complete Hook Trigger Reference

All 76 `HookTrigger` enum values (`src/roomkit/models/enums.py`). **Execution** is how
the engine invokes the trigger: SYNC triggers run through the sequential,
priority-ordered sync pipeline (can block/modify); ASYNC triggers run
concurrently, fire-and-forget, errors logged. An event's `AFTER_BROADCAST` hooks, a
change's `ON_EVENT_UPDATED` and `ON_EVENT_DELETED`, `ON_TASK_CREATED` and
`ON_ERROR` hold neither the call nor the room: `process_inbound` returns once
its delivery set and its answers are done, and the room's next message and an
agent's answer are delivered while they run; `event_processed` comes once the
event's `AFTER_BROADCAST` hooks have ended (RMK-1020). A room change starts its
hooks and returns without waiting for them: `attach_channel`, `detach_channel`,
`mute`, `unmute`, `close_room`, `check_room_timers`, the member calls and a direct
`update_event` or `delete_event`; `create_room` alone awaits its `ON_ROOM_CREATED`
hooks, so one that attaches the room's agent has done so before anything is sent
into the room. The triggers only this implementation has follow the same rule
(RMK-1049), but for two more it awaits: `ON_RECORDING_STARTED`, the consent point
no media reaches a recording before, and the triggers of a voice, realtime,
conference or video session's stream, observed in order with the stream's next
trigger. A framework event that reports the operation (`ai_response`, `feedback`,
`recording_started`, `recording_stopped`) comes once its hooks have ended.
`await kit.hook_engine.drain_observers()` waits for the hooks still
running. Registration mode is forgiving in
both directions: ASYNC-registered hooks on a SYNC trigger fire as observers
after the sync pass, in the background: the operation goes on without waiting
for them, so a hook whose work it needs first is registered SYNC (RMK-903).
The ASYNC hooks of `BEFORE_BROADCAST` start once the event's pass is over,
whatever the store, for every pass (a message, a send, a post, an agent's
answer, an injected event): the event stored and its delivery placed in the
room's lane. One the SYNC hooks blocked has no observer; one they allowed is
observed even when its pass then ends without storing it as sent (its source
may not write, `process_timeout` ran out, the commit failed) (RMK-1039), but
for a pass the kit's close cuts, which starts none (RMK-1046).
Fired from the room's work, such an observer is that cascade's work: what it
sends into the room draws one of the cascade's hook calls, as an
`AFTER_BROADCAST` body's does, so one that posts on every event it observes
stops at the cap (RMK-978). The observers of the triggers that fire per tool
call, per round or per input request (`BEFORE_TOOL_USE`, `ON_TOOL_CALL`,
`AFTER_TOOL_ROUND`, `ON_USER_INPUT_REQUIRED`) are not the cascade's work:
they run as any task the turn's work starts. A post they make while the
turn's delivery runs draws one of that cascade's passes, so one per tool
call stops at the cap (`10 × max_chain_depth`, in process); a direct change
(`update_event()`, a progress card) draws nothing; and what they send is at
least as deep as the turn was when the trigger fired, once the turn is over
too (an ACP turn carries no depth).
SYNC-registered hooks (the default) on an ASYNC trigger
fire like any other observer. Triggers marked *Reserved* exist in the enum but
are not fired by any built-in code.

### Event Pipeline

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `BEFORE_BROADCAST` | SYNC | `(event, ctx) -> HookResult` | Before event is stored and broadcast. Can block/modify. Its ASYNC observers start once the event's pass is over. |
| `AFTER_BROADCAST` | ASYNC | `(event, ctx) -> None` | After event is broadcast. Fire-and-forget side effects. |

### Event Mutation (edit/delete)

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_EVENT_UPDATED` | ASYNC | `(event, ctx) -> None` | A persisted event's stored state changed — an accepted EDIT, whichever pass committed it (an inbound message, an agent's response, a hook's injected event), or `kit.update_event()`. Payload is the updated event (`metadata.edited=True` on the edit path). |
| `ON_EVENT_DELETED` | ASYNC | `(event, ctx) -> None` | A persisted event was deleted — an accepted DELETE (soft, `metadata.deleted=True`, payload the marked target) or `kit.delete_event()` (hard: fires once per removed event, the root, its replies and the removed events' edits and deletes, each as it was stored). |

### Channel Lifecycle

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_CHANNEL_ATTACHED` | ASYNC | `(event, ctx) -> None` | Channel was attached to a room |
| `ON_CHANNEL_DETACHED` | ASYNC | `(event, ctx) -> None` | Channel was detached from a room |
| `ON_CHANNEL_MUTED` | ASYNC | `(event, ctx) -> None` | Channel was muted in a room: its binding's `muted` changed to true, and only then (muting a muted channel fires nothing) |
| `ON_CHANNEL_UNMUTED` | ASYNC | `(event, ctx) -> None` | Channel was unmuted in a room: its binding's `muted` changed to false, and only then |

### Room Lifecycle

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_ROOM_CREATED` | ASYNC | `(event, ctx) -> None` | Room was created |
| `ON_ROOM_PAUSED` | ASYNC | `(event, ctx) -> None` | Room was paused by its inactivity timer; fires once the room lock is released, `ctx.room` as the transition left it |
| `ON_ROOM_CLOSED` | ASYNC | `(event, ctx) -> None` | Room was closed, by `close_room()` or by its close timer (that one once the room lock is released); a close of a CLOSED room fires nothing |

### Identity

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_IDENTITY_AMBIGUOUS` | SYNC | `(event, ctx) -> IdentityHookResult` | Multiple identity matches found |
| `ON_IDENTITY_UNKNOWN` | SYNC | `(event, ctx) -> IdentityHookResult` | No identity match found |
| `ON_PARTICIPANT_IDENTIFIED` | ASYNC | `(event, ctx) -> None` | Participant was successfully identified |
| `ON_VERIFICATION_REQUIRED` | SYNC | `(event, ctx) -> HookResult` | An identified sender who is not verified wrote, under a `verification_policy` (RFC §11.7). `allow()` holds the message from the agents, `allow_unverified()` passes it, `block()` refuses it. |
| `ON_VERIFICATION_COMPLETED` | ASYNC | `(event, ctx) -> None` | A verification request was completed. A SYSTEM event whose data carries `identity_id`, `organization_id`, `request_id`, `channel_id`, `method`, `expires_at`. |
| `ON_VERIFICATION_ENDED` | ASYNC | `(event, ctx) -> None` | A verification or an open request ended: expired, revoked, cancelled or failed. Same data plus `reason` and `ended` (`"verification"` or `"request"`). |

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
| `BEFORE_DELIVER` | SYNC | `(event, ctx) -> HookResult` | Before a proactive `kit.deliver()` strategy executes (in-process and worker paths), a delegation's or a supervisor's background result included. Payload is a synthetic INTERNAL system-source event describing the delivery, of type `INSTRUCTION` for `deliver(instruction=True)`. `HookResult.block()` refuses the delivery (outcome `blocked`); a modified event's text replaces the content. A hook that raises does not block (fail-open, RFC §9.3). |
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
| `ON_TRANSCRIPTION` | SYNC | `(event, ctx) -> HookResult` | STT produced a final transcript. Can block/modify; **fails closed** — a hook that raises/times out blocks publication. Payload: `TranscriptionEvent` (VoiceChannel), `RealtimeTranscriptionEvent` (RealtimeVoiceChannel), `ConferenceTranscription` (ConferenceChannel). Modify by returning the same type with the new text; any other payload is an unusable result (a plain `str` included), which blocks since the trigger fails closed. |
| `ON_PARTIAL_TRANSCRIPTION` | ASYNC | `(PartialTranscriptionEvent, ctx) -> None` | Streaming interim STT result. Hot path — skipped entirely when no hooks are registered. |
| `BEFORE_TTS` | SYNC | `(text: str, ctx) -> HookResult` | Before text is synthesized. Payload is the text `str`; modify with a `str`. Can block; **fails closed**. On a streamed response it runs once per sentence: a block drops that sentence. |
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
| `ON_RECORDING_STARTED` | ASYNC | `(RecordingStartedEvent, ctx) -> None` | Recording started (voice session, conference track or room), before any media: the consent point |
| `ON_RECORDING_STOPPED` | ASYNC | `(RecordingStoppedEvent, ctx) -> None` | Recording stopped (voice session, conference track or room), result available; a room's `session` is `None` and `room_id` names it |
| `ON_INPUT_AUDIO_LEVEL` | ASYNC | `(AudioLevelEvent, ctx) -> None` | Inbound audio level (throttled; telemetry-suppressed) |
| `ON_OUTPUT_AUDIO_LEVEL` | ASYNC | `(AudioLevelEvent, ctx) -> None` | Outbound audio level (throttled; telemetry-suppressed) |
| `BEFORE_BRIDGE_AUDIO` | SYNC | `(BridgeAudioEvent, ctx) -> HookResult` | Before an audio frame is forwarded across an audio bridge. Can block/modify the frame. Only invoked when hooks are registered — otherwise frames bypass the event loop for latency. |

### Tool Execution

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `BEFORE_TOOL_USE` | SYNC | `(ToolCallEvent, ctx) -> HookResult` | Before a tool executes (AIChannel, realtime channels, conference, external/ACP tools). Block denies the call, and the model reads the block's reason. The call runs with the arguments the chain leaves: a MODIFY's or an in-place edit's, read by the next hook, or a `metadata["arguments"]` rewrite, which no later hook reads and which has the last word; validated again either way. **Fails closed**: a hook that raises, times out or returns something unusable denies the call; the model then reads `Tool 'x' denied by pre-execution hook.`, never the hook's error, which reaches ON_TOOL_CALL's observers on `error_detail` (an external tool handler decides and reports the call itself). A room context that cannot be built refuses the call too. |
| `ON_TOOL_CALL` | SYNC | `(ToolCallEvent, ctx) -> HookResult` | A tool call executed (AIChannel, RealtimeVoiceChannel, skills, external/ACP tools). In the AI channel's tool loop, Block returns an error result to the model and marks the call failed, its structured copy dropped; `HookResult(metadata={"result": ...})` or `HookResult.modify(event)` supplies or overrides the tool result (`event.result is None` means the hook must provide it; a served result replaced with `None` reads as JSON `null`, on every channel) and keeps the copy; SYNC hooks chain, each seeing the result the previous one left, and ASYNC observers see the final one as the model reads it (a replacement serialized), a blocked call included (`is_error=True`), while a report on a call whose outcome the model already read (an external tool's, one the provider ran itself) reaches them as it was read, whatever a SYNC hook returned; `event.structured_content` is the copy the tool-call event carries for UI surfaces (MCP `structuredContent`), which `metadata={"structured_content": ...}` replaces and `None` clears. RFC §9 lists this ASYNC; code invokes it through the sync pipeline. |
| `AFTER_TOOL_ROUND` | SYNC | `(ToolRoundEvent, ctx) -> HookResult` | Between two rounds of an AIChannel's tool loop, after each round the channel ran calls in. The event carries the round whole (`calls`, the channel's `results`, the provider's `answered`) and the names of the turn's toolset (`tools`). Act on it in place: `event.withdraw(*names)` takes tools out of the rest of the turn with every guarantee of a BEFORE_AI_GENERATION withdrawal, `event.add_message(text)` is what the next round reads after the results. A BLOCK stops the hooks after it and changes nothing of the round; a MODIFY's event is not read. Not fired by realtime sessions, a reasoning backend's agent or an ACP agent's own loop. |
| `ON_USER_INPUT_REQUIRED` | SYNC | `(PendingInputEvent, ctx) -> HookResult` | A `HumanInputHandler`-backed tool paused awaiting human input. Sync so the notification (e.g. WebSocket push) lands before `wait()` starts blocking; block auto-rejects the pending request. |

### Realtime Voice

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_REALTIME_TEXT_INJECTED` | ASYNC | `(event, ctx) -> None` | Text was injected into realtime session |
| `ON_REALTIME_DELEGATION` | ASYNC | `(RealtimeDelegationEvent, ctx) -> None` | A full-duplex model handed reasoning or tool use to a backend, hosted or integrator-side |

### AI Generation

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `BEFORE_AI_GENERATION` | SYNC | `(AIGenerationEvent, ctx) -> HookResult` | Before the AI provider is invoked. `event.ai_context` (messages, system prompt, tools) may be mutated in place; block skips generation. Its `tools` are the turn's toolset after the tool policy and skill gating, under Tool Search the whole catalogue: a tool it removes is declared at no round, named by no `find_tools`, and refused; a tool it adds is declared at every round, never deferred. |
| `ON_AI_THINKING` | ASYNC | `(ThinkingEvent, ctx) -> None` | A reasoning window opened (`THINKING_START`) on an AIChannel; fires whether or not a realtime backend is configured. The reasoning text itself streams on the ephemeral `THINKING_DELTA` / `THINKING_END` events, so `event.thinking` is empty here. |
| `ON_AI_RESPONSE` | ASYNC | `(AIResponseEvent, ctx) -> None` | AI generation completed. Carries the turn's text — `response_content`, the segments a tool call cut joined with a blank line, and `segments` one by one — plus usage, latency, tool call count, and `loop_end_reason` — which of the tool loop's rules ended the turn (`completed`, `max_rounds`, `timeout`, `budget_exceeded`, `cancelled`, `force_stopped`, `truncated`, `empty_response`, `unfinished`, `error`). Read that rather than inferring a cut-off from `tool_calls_count`, which counts the calls the turn ran. |

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
| `ON_STATUS_POSTED` | ASYNC | `(StatusEntry, ctx) -> None` | A `StatusBus` post whose `metadata` carries a `room_id` ran that room's hooks; a post with no room reaches no hook (the bus is global). Every post also emits the `status_posted` framework event. |

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

### Speaking Turns

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_SPEAK_DECISION` | SYNC | `(SpeakDecisionEvent, ctx) -> HookResult` | An AI channel's speak policy decided whether the agent speaks, offers or stays silent on an event: the decision with its reason and judgments, `duration_ms`, `asked_again` when it decided again once the agent thought, `thought` (what the agent had in mind, with a thinker), and `late` when that thought came back after the event was decided: no turn runs on a late decision, the hooks act on it. The channel acts on the decision the SYNC hooks leave: a MODIFY whose event carries another decision replaces it (only `event.decision` is read), a BLOCK leaves the agent silent (`blocked by <hook>`). Fail-open. ASYNC hooks observe the decision that stands. |
| `ON_THOUGHT` | ASYNC | `(ThoughtEvent, ctx) -> None` | An AI channel's thinker came back with a new thought in a room, or speaking emptied what the agent wanted to say: the thought, the one it replaces, `duration_ms` of the thinker call. |

A SYNC hook takes in code what the decision calls for, rather than leaving it
to the model: here an offer becomes a raised hand, and the agent stays silent
until someone gives it the floor (`examples/raise_hand_on_offer.py`).

```python
import dataclasses

from roomkit import HookResult, HookTrigger, SpeakDecision, SpeakDecisionEvent

@kit.hook(HookTrigger.ON_SPEAK_DECISION, name="raise_hand")
async def raise_hand(event: SpeakDecisionEvent, ctx) -> HookResult:
    mode = event.decision.mode
    if mode == "silent" or (mode == "speak" and not event.late):
        return HookResult.allow()
    # An offer, or a late decision (no turn runs on it): the hand goes up.
    await meeting.raise_hand()  # the application's own action
    silent = SpeakDecision("silent", reason="hand raised")
    return HookResult.modify(dataclasses.replace(event, decision=silent))
```

A SYNC hook on this trigger returns a `HookResult`; one that only watches the
decisions is registered `execution=HookExecution.ASYNC`.

### Discussion

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_DISPATCH_DECISION` | ASYNC | `(DispatchDecisionEvent, ctx) -> None` | A discussion's dispatch policy decided who takes a person's message that names no agent (RFC §19.7.5 rule 18): `event.candidates`, `event.decision` (`agents` in order, `reason`, `judgments`), `event.duration_ms`. The reason `fallback` means the policy failed or ran past `dispatch_timeout`, and every candidate was asked. |
| `ON_SPEAK_QUEUE` | ASYNC | `(SpeakQueueEvent, ctx) -> None` | A discussion's speak queue changed (RFC §19.7.5): `event.change` is a `SpeakQueueChange` (`queued`, `turn_given`, `turn_ended`, `instruction_dropped`, `listening`, `talking_again`, `waiting`, `over`), `event.channel_ids` the agents it concerns, `event.queue` the `SpeakQueue` as it now is. Announced in the order the changes happened. |

### Planning

| Trigger | Execution | Signature | Description |
|---------|-----------|-----------|-------------|
| `ON_PLAN_UPDATED` | ASYNC | `(PlanUpdatedEvent, ctx) -> None` | The `plan_tasks` tool (`enable_planning=True`) updated the room's task plan; `event.tasks` carries the new list. The same update also goes out as an ephemeral realtime event `{"type": "plan_updated"}` for live UIs. |

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

Available framework event types: `room_created`, `room_closed`, `room_paused` (both with `reason = timer` when a timer made the transition), `room_archived`, `room_refused_event`, `room_channel_attached`, `room_channel_detached`, `channel_connected`, `channel_disconnected`, `voice_session_started`, `voice_session_ended`, `source_attached`, `source_detached`, `source_error`, `source_exhausted`.
