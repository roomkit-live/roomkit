# Orchestration

RoomKit provides five declarative orchestration strategies for multi-agent workflows. Pass a strategy to `RoomKit(orchestration=...)` or `create_room(orchestration=...)` — agents, routing, handoff tools, and conversation state are wired automatically.

## Strategies

### Pipeline

Linear agent chain: triage -> handler -> resolver. Each agent can only hand off to the next in sequence.

```python
from roomkit import Agent, Pipeline, RoomKit, WebSocketChannel
from roomkit.providers.ai import MockAIProvider

triage = Agent(
    "triage",
    provider=MockAIProvider(responses=["Transferring you..."]),
    role="Triage agent",
    description="Routes requests to the right specialist",
    system_prompt="You triage incoming requests.",
)
handler = Agent(
    "handler",
    provider=MockAIProvider(responses=["Let me help with that."]),
    role="Request handler",
    description="Handles customer requests",
    system_prompt="You handle requests.",
)
resolver = Agent(
    "resolver",
    provider=MockAIProvider(responses=["All done!"]),
    role="Resolution specialist",
    description="Confirms resolution",
    system_prompt="You resolve and close requests.",
)

kit = RoomKit(orchestration=Pipeline(agents=[triage, handler, resolver]))
```

### Swarm

Every agent can hand off to every other agent. Bidirectional routing.

```python
from roomkit import Swarm

kit = RoomKit(orchestration=Swarm(agents=[billing, shipping, returns]))
```

### Supervisor

A supervisor agent delegates tasks to worker agents in child rooms:

```python
from roomkit import Supervisor

kit = RoomKit(orchestration=Supervisor(
    supervisor=manager_agent,
    workers=[researcher, writer, reviewer],
))
```

### Loop

Producer/reviewer cycle. The reviewer has an `approve_output` tool to break the loop.
A message the loop's tasks cannot carry (a photo, a text with a photo, an empty
text) is answered by the producer alone, which sees the media: the tasks carry
text only.

```python
from roomkit import Loop

kit = RoomKit(orchestration=Loop(
    agent=writer_agent,
    reviewer=editor_agent,
    max_iterations=3,
))
```

### Discussion

Several agents and people hold one conversation; one agent speaks at a time
(RFC §19.7.5). Any agent may address any other with `@handle`, a person
may address any agent, and who speaks next follows from who was addressed.
The strategy queues the agents each message asks for and gives their turns
one at a time; each turn reads the room as it is when it starts.

```python
from roomkit import Discussion

room = await kit.create_room(orchestration=Discussion(
    agents=[investigator, dev, sre],   # AI channels (Agents among them) and ACPChannel agents
    everyone=["investigator", "dev", "sre"],  # a message naming nobody asks these, in order
    people=["oncall"],                 # names agents address people by
))
```

- Each agent has a handle, the name `@` addresses it by: `handles={channel_id:
  handle}`, else its channel id when that can be a name, else its `name` kept
  to a name's characters (`Agent("agent:3f2a…", name="Jev")` is `@Jev`; an
  `agent:<uuid>` id cannot be a name). Letters, digits, `_ . -`, at most 32,
  not `all`; an agent with none, or two alike ignoring case, is refused at
  construction. Notes, transcript labels, the dispatch policy,
  `DiscussionConsole` and a `CLIChannel` (its banner and its labels, unless
  the host passes `agent_label`) show handles; the queue, `addressed_to` and
  hooks keep channel ids.
  `strategy.handles()` returns them by channel id.

- A person's message puts the agents it names at the front of the queue;
  naming nobody, it answers the agents that asked that person, else
  `everyone`. With `addressed_only=True` it asks only the agents it names.
- An agent's answer that names another queues it at the back once the turn
  has ended, one turn per agent however often it is named. `@all` names
  every agent.
- An agent with nothing to add answers `silent_token` (`"(silent)"`): the row
  is stored `BLOCKED` with `blocked_by="discussion_silent"` and never reaches
  a transport, streamed or not.
- Each turn's notes end with how the room works: by default, end the message
  with the handle of the agent that must act next. `rules="..."` gives the
  host's text instead, `{handle}` standing for another agent's `@handle` and
  `{silent}` for `silent_token`; it is stored with the discussion, so every
  process gives its turns with it.
- `max_depth` bounds how far a chain of agent turns reaches from a person's
  message (default: the kit's `max_chain_depth`); a stopped turn waits for
  the next person's message. `max_turns` and `done(room_id)` end it.
- The discussion waits for a person (`waiting`) when an agent asked one, an
  agent of the queue only listens, or the depth limit stopped a turn.
- `kit.listen_only(room_id, ids)` / `kit.talk_again(room_id, ids)`: an agent
  that only listens answers only a person who names it; setting it on the
  speaking agent cuts its turn as a `Cancel` does.
- `kit.speak_queue(room_id)` returns the `SpeakQueue` (`speaking`, `queue`,
  `listening`, `asked`, `waiting`, `over`); `ON_SPEAK_QUEUE` announces each
  change as a `SpeakQueueEvent`. The host calls take `organization_id`.
- Agents address people by the transcript's names (`@AliceMartin`, `@sms1`
  for a sender with no name, at most 32 characters); an ask is recorded
  against that exact person, never one who takes their name.
- An `INSTRUCTION` addressed to agents gives each a turn of its own at the
  front; `regenerate_response()` does for each agent that answered (else each
  the message asked), once per agent and message. Once the discussion is
  over, both are refused with `reason="discussion_over"`.
- `process_inbound()` returns no agent answer: the answers come in the turns.
- Each agent's memory provider is handed every message the agent may see
  (not its own) as it commits, as in a room with no discussion, so a memory
  that learns as messages arrive (an index, a summary) learns the whole
  conversation; the turn that answers a message does not hand it again.
- The room is the discussion's: install refuses another intelligence
  channel, a voice or realtime channel, an agent with a thinker, a router
  or another strategy, and refuses binding or installing them afterwards.
  `await kit.uninstall_strategy(room_id)` gives the room back to its
  `agent_response_policy` and forgets the discussion.
- The configuration (`_discussion`) and the queue (`_speak_queue`) are
  stored with the room and outlive a restart. Every process serving the room
  follows them, even one whose host did not install the discussion: no agent
  asked at broadcast, names read, turns queued, under the room lock (use
  `PostgresAdvisoryLockManager` across processes). One process that
  installed it gives the turns under a 15 s lease it renews; another takes
  over once it expires. Example: `examples/discussion_two_workers.py`.
  Text only in this version.
- `kit.install_strategy(room_id, Discussion(...))` on a room that holds a
  discussion, installed here or stored by another process, takes its place
  and keeps the queue, who listens and the turns given: how a restarted
  process takes the discussion back, and how the host changes the agents.
  An agent left out leaves the queue, its running turn is cut and it is
  detached when an install attached it (one the host bound itself is
  refused). Every process routes by the new configuration; one whose host
  installed the old one gives no turn until its host installs the new one.
  `kit.uninstall_strategy(room_id)` works from any process, one that only
  follows included, and detaches the agents the installs attached. Both may
  run while the host holds `kit.lock_manager.locked(room_id)`: uninstalling
  cuts the running turn without waiting for its end.
- `await kit.room_strategy_is_current(room_id, Discussion)` says whether the
  discussion this process installed is the one the room stores. It is false
  once another process installed another configuration over it (this
  process then gives no turn), when this process installed none, and once it
  was uninstalled; an uninstall made elsewhere is taken at once, and
  `kit.room_strategy(room_id)` then answers `None`. A host that finds it
  false installs its agents again only while its own state still wants the
  discussion: `if wanted and not await kit.room_strategy_is_current(room_id,
  Discussion): await kit.install_strategy(room_id, Discussion(agents))`. A
  kind not stored with the room (`Pipeline`, `Swarm`, ...) raises `TypeError`.

**Talking and working.** A turn talks; long work runs as a task, off the
floor (rule 19). An agent given the background tool starts it, says it is
on it and ends its turn; a question asked meanwhile is answered in its turn:

```python
kit.setup_background(comms)                       # comms does its own tasks
kit.setup_background(dev, worker="dev-code", hand_back="post")
kit.setup_task_tools(comms)                      # task_status + cancel_task, owner-served
task = await kit.start_task(room_id, "dev", "Go through every support ticket.")
await kit.cancel_task(task.id)
```

- `setup_task_tools(agent)` gives an AI channel `task_status` and
  `cancel_task`, served by the room's owner wherever its turns run (RFC
  §10.6.5); a `cancel_task` reaches only a task the kit delegated for the
  room of the call and posts on the status bus. A `task` line naming another
  room's task cancels nothing, and a task a strategy runs unposted
  (`post_status=False`) is the strategy's to follow. A task of the room the
  call may not cancel (an inline delegation, a strategy's unposted run) is
  answered `running`, with why; one that ended, with its status, even before
  the bus shows that end; another room's running task, or another kit's on a
  shared bus, is `unknown`; a line listed ended is answered as it is listed.
- The tools strategies and `setup_*` set up (handoff, `delegate_workers`,
  `delegate_to_<id>`, `submit_result`, `work_in_background`, `task_status`,
  `cancel_task`, `delegate_task`) are the kit's, kept per channel and room
  beside the channel's declaration, never on the agent (RFC §10.6.2); each
  turn's request lists them. An executor refuses a turn whose agent declares
  a tool of one of their names (`ToolNameCollisionError`).

- `setup_background(agent, worker=None, hand_back="relay")` gives an AI
  channel the `work_in_background` tool; the task runs on its worker (itself,
  another agent, an external agent) in a child room, handed the room's last
  12 messages (`delegate(brief=..., conversation=12)`).
- `start_task(room_id, agent_id, task, hand_back=None, organization_id=None)`
  starts the same task from the host, for any agent of the room, an
  `ACPChannel` included; another organization's room raises
  `RoomNotFoundError`, nothing started (RFC §17.2).
- `cancel_task(task_id, organization_id=None)` cancels a task the kit runs
  in the background; with `organization_id=`, only one running for a room
  of that organization: another organization's task answers `False`, as one
  the kit does not hold, and keeps running.
- `hand_back="relay"`: the agent is told the result and says it in a turn of
  its own, at the front of the queue. `"post"`: the result is posted as the
  agent's message, as it is, with no generation; it waits at the front of the
  queue for the turn under way to end, and its names ask no one.
- Tasks run in parallel; each turn's notes list the room's tasks; a worker's
  call of the tool is refused; a result that comes back once the discussion
  is over or gone is posted at once (relayed: refused once over).
  A posted result the discussion took and has not committed (queued,
  waiting for the floor, or being committed) is kept in the room's private
  state, apart from its metadata: `uninstall_strategy`, in any process,
  posts it through that process's hooks before it detaches the agents the
  installs attached, keyed by the result's id, so the process that held it
  posts nothing more. One a closing kit did not post, or whose post failed,
  waits for the next uninstall.
  Example: `examples/discussion_background.py`.

**External agents.** An `ACPChannel` (Claude Code, Codex) is an agent like
any other: its `name`, `role` and `description` present it, `instructions`
go once per session, each turn's prompt ends with the turn's notes, and
`listen_only` on it while it speaks cancels its prompt. Its tools are its
own. Example: `examples/discussion_acp.py`.

A **dispatch policy** decides who takes a person's message that names no
agent, in place of giving it to the agents that asked that person, else to
every agent of `everyone` (rule 18). In a room of specialists, most of those
turns say `(silent)`, the agent that should answer may come third, and an
agent that asked a question takes the next message even when it is a new
request for someone else:

```python
from roomkit import ClassifierDispatchPolicy, Discussion, JevClassifier

Discussion(
    agents=[investigator, sre, dev, comms],  # Agents: name and description tell who does what
    dispatch=ClassifierDispatchPolicy(JevClassifier(), threshold=0.5, max_agents=2),
)
```

- One decision per message, by the process holding the lease, off the room
  lock, before it gives another turn; the agents picked go to the front in
  the order decided, at the message's place in the queue. An empty decision
  asks no agent (a thanks, small talk).
- The candidates are the agents that asked the person, then `everyone`,
  less the agents the message does not reach and those that only listen; a
  decision naming anyone else is cut down to them. `DispatchTurn.asked`
  says which candidates wait for the person's answer, so the policy judges
  whether the message answers them.
- A name is never decided; `addressed_only` is refused with a policy.
- A policy that fails, takes over `dispatch_timeout` (5 s) or returns no
  readable decision leaves the message to rule 8 (the agents that asked the
  person, else every candidate), with the reason `fallback`. At most 16
  messages wait for a decision; past them, the oldest goes the same way.
- `ON_DISPATCH_DECISION` reports each decision applied, once per message
  (`DispatchDecisionEvent`:
  the message, the candidates, the `DispatchDecision` with its `reason` and
  `judgments`, `duration_ms`).
- `ClassifierDispatchPolicy` asks one yes/no question per candidate in one
  classifier call (`JevClassifier`, `OpenAIClassifier`, `SystemOneClassifier` for any
  `/v1/systemone` server such as Ollama or a LiteLLM proxy, `LlamaCppClassifier` for a
  decision model RoomKit runs through llama.cpp, `LLMClassifier`); `MockDispatchPolicy` scripts decisions for tests; any
  `DispatchPolicy.decide(DispatchTurn) -> DispatchDecision` will do.
  Example: `examples/discussion_dispatch.py`.

A full-screen terminal for such a room (`pip install roomkit[console]`): the
room on the left, the agents with their identity and live state on the right,
the speak queue below, your input at the bottom. What you type goes into the
room through the transport channel you name; `/listen @a`, `/talk @a`,
`/help`, `/quit`, plus the host's own slash commands. Agents show and are
named by their discussion handle (`@Dev` for `agent:8e41…`); a host's own
`AgentCard("Dev", channel_id="agent:8e41…")` does the same. With a WebSocket
channel, the answer an agent is writing shows as it streams. When an agent
has background work set up (or `tasks=True`), `/task @a ...`, `/tasks` and
`/cancel <task>` start, list and stop the room's tasks (`/cancel` only one
the kit runs for the room; an inline run it says is still running, and why),
each agent's card shows its running tasks, and a posted result names the
worker that wrote it.

```python
from roomkit.console import DiscussionConsole

kit.register_channel(WebSocketChannel("you"))
await kit.attach_channel(room_id, "you")
await DiscussionConsole(kit, room_id, channel_id="you", log_file="room.log").run()
```

Example: `examples/discussion_console.py`.

### Installing a strategy while the room lives

A room holds one strategy at a time, and can change it while it lives
(RFC §19.7). `install_strategy` registers and attaches the strategy's agents,
as `create_room(orchestration=...)` does, and keeps the timeline: the agents'
turns read the conversation so far. `uninstall_strategy` runs the strategy's
own `uninstall()` step, then takes back what its install added for the room:
room hooks, tools set up on the agents, the turn runners the kit keeps for
them, agents it attached, room metadata it wrote.

```python
await kit.install_strategy(room_id, Discussion([assistant, dev]))   # on a live room
await kit.uninstall_strategy(room_id)                               # back to the room's policy
await kit.install_strategy(room_id, Swarm(agents=[assistant, billing], entry="assistant"))
kit.room_strategy(room_id)                                          # the one installed
```

A second strategy is refused while one is installed, but for a Discussion
installed over a discussion, which takes its place and keeps its queue. A
host's own strategy gets the same treatment through `install_strategy`. If it is installed by
calling its `install()` directly, it should call
`await kit.claim_room_strategy(room_id, self)` first, as the built-in ones
do; with `organization_id=` (RFC §17.2), another organization's room is not
found. The claim reads the room as an install does: a discussion another
process stored refuses another strategy in a process that never saw it.
Example: `examples/strategy_on_the_fly.py`.

## Using Orchestration

```python
from roomkit import InboundMessage, TextContent, WebSocketChannel

# Register transport channel
ws = WebSocketChannel("ws-user")
kit.register_channel(ws)

# Create room — orchestration auto-registers agents, creates router, sets initial state
await kit.create_room(room_id="support")
await kit.attach_channel("support", "ws-user")

# Messages are automatically routed to the active agent
await kit.process_inbound(
    InboundMessage(
        channel_id="ws-user",
        sender_id="user",
        content=TextContent(body="I need help with billing."),
    )
)
```

## Handoff Protocol

Agents hand off conversations by calling the `handoff_conversation` tool (auto-injected by orchestration strategies):

```python
# The AI calls this tool automatically:
# handoff_conversation(target="handler", reason="Billing issue", summary="User needs invoice help")
```

The handoff:
1. Updates `ConversationState` in room metadata (phase, active agent, handoff count)
2. Mutes the outgoing agent, unmutes the incoming agent
3. Emits a system event visible to the new agent
4. Records the transition in `phase_history`

## Conversation State

`ConversationState` tracks conversation progress within a room. It's stored in `Room.metadata["_conversation_state"]` and persists across all message turns. Orchestration strategies create and update it automatically, but you can also read and modify it directly.

### ConversationState Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `phase` | `str` | `"intake"` | Current conversation phase. Can be any string. |
| `active_agent_id` | `str \| None` | `None` | Channel ID of the currently active agent. |
| `previous_agent_id` | `str \| None` | `None` | Agent before the last transition. |
| `handoff_count` | `int` | `0` | Total number of agent handoffs. |
| `phase_started_at` | `datetime` | now | When the current phase started. |
| `phase_history` | `list[PhaseTransition]` | `[]` | Immutable audit trail of all transitions. |
| `context` | `dict[str, Any]` | `{}` | **Arbitrary user data** — store custom key-value pairs across turns. |

Built-in phases: `ConversationPhase.INTAKE`, `QUALIFICATION`, `HANDLING`, `ESCALATION`, `RESOLUTION`, `FOLLOWUP`.

### Reading State

```python
from roomkit.orchestration import get_conversation_state

room = await kit.get_room("support")
state = get_conversation_state(room)

print(state.phase)            # "handling"
print(state.active_agent_id)  # "billing-agent"
print(state.handoff_count)    # 2
print(state.context)          # {"customer_tier": "premium", "issue_type": "refund"}

# Audit trail
for t in state.phase_history:
    print(f"{t.from_phase} -> {t.to_phase} by {t.from_agent} -> {t.to_agent} ({t.reason})")
```

### Persisting Custom Data Across Turns

Use `state.context` to store arbitrary data that survives across conversation turns:

```python
from roomkit.orchestration import get_conversation_state, save_conversation_state

room = await kit.get_room("support")
state = get_conversation_state(room)

# Store custom data in context (immutable update)
updated_state = state.model_copy(update={
    "context": {
        **state.context,
        "customer_tier": "premium",
        "issue_type": "refund",
        "attempts": state.context.get("attempts", 0) + 1,
    }
})

# Persist its metadata key alone: a full room write from the room read
# above would undo what was written since
await save_conversation_state(kit.store, room.id, updated_state)
```

### Retrieving Custom Data on Later Turns

```python
room = await kit.get_room("support")
state = get_conversation_state(room)
tier = state.context.get("customer_tier", "standard")
attempts = state.context.get("attempts", 0)
```

### Programmatic Phase Transitions

Use `state.transition()` to change phase and record an audit entry:

```python
from roomkit.orchestration import get_conversation_state, save_conversation_state

room = await kit.get_room("support")
state = get_conversation_state(room)

new_state = state.transition(
    to_phase="escalation",
    to_agent="supervisor-agent",
    reason="Customer requested manager",
    metadata={"escalation_priority": "high"},
)

await save_conversation_state(kit.store, room.id, new_state)
```

### PhaseTransition Audit Record

Each transition creates an immutable `PhaseTransition`:

| Field | Type | Description |
|-------|------|-------------|
| `from_phase` | `str` | Previous phase |
| `to_phase` | `str` | New phase |
| `from_agent` | `str \| None` | Previous agent channel ID |
| `to_agent` | `str \| None` | New agent channel ID |
| `reason` | `str` | Why the transition occurred |
| `timestamp` | `datetime` | When the transition occurred (UTC) |
| `metadata` | `dict[str, Any]` | Arbitrary metadata for this transition |

### Using State in Hooks

```python
from roomkit import (
    HookResult,
    HookTrigger,
    RoomContext,
    RoomEvent,
    TextContent,
    get_conversation_state,
    save_conversation_state,
)

@kit.hook(HookTrigger.BEFORE_BROADCAST)
async def track_sentiment(event: RoomEvent, ctx: RoomContext) -> HookResult:
    if not isinstance(event.content, TextContent):
        return HookResult.allow()
    state = get_conversation_state(ctx.room)
    updated_state = state.model_copy(update={
        "context": {
            **state.context,
            "message_count": state.context.get("message_count", 0) + 1,
        }
    })
    await save_conversation_state(kit.store, ctx.room.id, updated_state)
    return HookResult.allow()
```

## Conversation Router (Advanced)

`ConversationRouter` dynamically routes incoming messages to different agents based on conversation state, message content, origin channel, or custom logic.

### RoutingConditions Reference

All conditions are ANDed — every non-None field must match:

| Field | Type | Description |
|-------|------|-------------|
| `phases` | `set[str] \| None` | Match when `state.phase` is in this set |
| `channel_types` | `set[ChannelType] \| None` | Match when sender's channel type is in this set |
| `intents` | `set[str] \| None` | Match when `event.metadata["intent"]` is in this set |
| `source_channel_ids` | `set[str] \| None` | Match when sender's channel ID is in this set |
| `custom` | `Callable \| None` | Custom `(event, context, state) -> bool` for arbitrary logic |

### Routing by Phase

```python
from roomkit.orchestration import ConversationRouter, RoutingConditions, RoutingRule

router = ConversationRouter(
    rules=[
        RoutingRule(agent_id="billing-agent", conditions=RoutingConditions(phases={"billing"})),
        RoutingRule(agent_id="shipping-agent", conditions=RoutingConditions(phases={"shipping"})),
    ],
    default_agent_id="triage-agent",
)
```

### Routing by Channel Type (Origin)

```python
from roomkit.models import ChannelType

router = ConversationRouter(
    rules=[
        RoutingRule(agent_id="voice-specialist", conditions=RoutingConditions(channel_types={ChannelType.VOICE})),
        RoutingRule(agent_id="sms-agent", conditions=RoutingConditions(channel_types={ChannelType.SMS, ChannelType.WHATSAPP})),
    ],
    default_agent_id="general-agent",
)
```

### Routing by Intent (Content-Based)

Set `event.metadata["intent"]` via a classification hook, then route by intent:

```python
@kit.hook(HookTrigger.BEFORE_BROADCAST, priority=-200)  # Run before router
async def classify_intent(event: RoomEvent, ctx: RoomContext) -> HookResult:
    if isinstance(event.content, TextContent):
        body = event.content.body.lower()
        intent = "billing" if "invoice" in body or "charge" in body else "general"
        modified = event.model_copy(update={"metadata": {**(event.metadata or {}), "intent": intent}})
        return HookResult.modify(modified)
    return HookResult.allow()

router = ConversationRouter(
    rules=[RoutingRule(agent_id="billing-agent", conditions=RoutingConditions(intents={"billing"}))],
    default_agent_id="general-agent",
)
```

### Routing with Custom Logic

```python
def is_high_value_customer(event, ctx, state):
    return state.context.get("customer_tier") == "premium"

def contains_urgency(event, ctx, state):
    if isinstance(event.content, TextContent):
        return any(w in event.content.body.lower() for w in ["urgent", "emergency", "asap"])
    return False

router = ConversationRouter(
    rules=[
        RoutingRule(agent_id="senior-agent", conditions=RoutingConditions(custom=is_high_value_customer), priority=-1),
        RoutingRule(agent_id="escalation-agent", conditions=RoutingConditions(custom=contains_urgency), priority=0),
    ],
    default_agent_id="general-agent",
    supervisor_id="supervisor-agent",
)
```

### Combined Conditions

Combine multiple conditions (all are ANDed):

```python
# Only route SMS messages during billing phase to the billing specialist
RoutingRule(
    agent_id="sms-billing-agent",
    conditions=RoutingConditions(
        phases={"billing"},
        channel_types={ChannelType.SMS},
    ),
    priority=0,
)
```

### Installing the Router

```python
# Option 1: Manual hook
kit.hook(HookTrigger.BEFORE_BROADCAST, execution=HookExecution.SYNC, priority=-100)(router.as_hook())

# Option 2: install() — also sets up handoff tools
handler = router.install(kit, agents=[billing_agent, shipping_agent, triage_agent])
```

### Routing Selection Priority

0. **Address** — if the event carries `addressed_to`, it decides; the router returns untouched
1. **Agent affinity** — if `state.active_agent_id` is set and attached, stick with it
2. **Rules** — evaluate in ascending `priority` order; first match wins
3. **Fallback** — return `default_agent_id`
4. **Loop prevention** — events FROM intelligence channels are never routed

## Addressing

Routing rules answer *which agent handles this kind of event*. Addressing answers *which agent am I talking to right now*, per message:

```python
await kit.process_inbound(
    InboundMessage(
        channel_id="you",
        sender_id="user",
        content=TextContent(body="review hello.py"),
        addressed_to=["codex"],       # only this agent is asked to act
    )
)
```

| `addressed_to` | Meaning |
|---|---|
| `None` | Unaddressed — every eligible agent acts, or the router decides |
| `["codex"]` | Only `codex` is asked; the others see it and stay silent |
| `[]` | Nobody is asked — a decision, not an absence |

Addressing is **not** visibility: it narrows who is *asked*, never who may *see*. Transport delivery is untouched, so the humans in the room still get the message. The address is stored on the event, so a transcript shows who was asked.

Direct injection addresses the same way — `[]` is what an application needs when it stores a message and triggers the answer itself:

```python
await kit.send_event(
    room_id=room_id,
    channel_id="system",
    content=TextContent(body=body),
    addressed_to=[],       # stored, and asking nobody
)
```

### Agent Response Policy

An agent's own output solicits the other agents by default (the chaining a pipeline needs, bounded by `max_chain_depth`). In a room of independent agents that is a hazard:

```python
# At creation, kit-wide or per room
kit = RoomKit(agent_response_policy=AgentResponsePolicy.ADDRESSED_ONLY)
await kit.create_room(room_id="dev", agent_response_policy=AgentResponsePolicy.ADDRESSED_ONLY)

# On a live room — the one that just gained a second agent
await kit.attach_channel(room_id, "codex", category=ChannelCategory.INTELLIGENCE)
await kit.set_agent_response_policy(room_id, AgentResponsePolicy.ADDRESSED_ONLY)
```

| Policy | An agent's output solicits |
|---|---|
| `AGENT_CHAIN` | every eligible intelligence channel — the default |
| `ADDRESSED_ONLY` | only the channels it addressed, if any |

A policy change applies to events processed after it; setting the policy a room already holds is a no-op. A binding that is not solicited is skipped before any work is done for it, so a roster can be attached lazily and rehydrated one agent at a time.

RoomKit takes the decision, never the syntax: `@codex`, a `/agent` command or a picker all live in the application, which passes channel ids.

## Conversation Pipeline (Advanced)

Define stages explicitly:

```python
from roomkit.orchestration import ConversationPipeline, PipelineStage

pipeline = ConversationPipeline(stages=[
    PipelineStage(phase="triage", agent_id="triage", next="handling"),
    PipelineStage(phase="handling", agent_id="handler", next="resolution"),
    PipelineStage(phase="resolution", agent_id="resolver"),
])
```

## Status Bus

Agents can publish status updates for UI display:

```python
# Agents publish status via the status bus
# Subscribe to updates
async def on_status(update):
    print(f"Agent {update.agent_id}: {update.message} ({update.level})")

await kit.status_bus.subscribe(on_status)
```

## Delegation

Delegate tasks to background agents in child rooms:

```python
result = await kit.delegate(
    room_id="main-room",
    agent_id="researcher",
    task="Find the latest pricing for product X",
    context={"product": "X"},
)
```

`task` is the label the result is named by; `brief=` is what the worker is
told instead, when it says more; `conversation=N` hands the worker the room's
last N messages the notified channel may read, quoted as data;
`hand_back="post"` publishes the result as the notified agent's message,
with no turn of its model, instead of relaying it (`"relay"`, the default).
`context=` is host metadata, the child room's `task_context`: no model reads it,
so a requester's email stays away from the vendor; on a `delegate_task` call below,
a model wrote it, so a hook reading it reads a model's text.

An agent delegates with the `delegate_task` tool
(`setup_delegation(agent, DelegateHandler(kit, ...), tool=build_delegate_tool(targets))`).
What its model passes as `context` does reach the worker, as JSON quoted as data
and bounded at 4,000 characters, and so do the agent's own last three completed
tasks of the room with `DelegateHandler(cache=)`; `task` stays as written. A tool
built with `build_delegate_tool(targets)` refuses a call naming another worker.
`max_conversation=N` (0 by default: no such argument) gives the tool a
`conversation` argument, held to N whatever the call asks, that hands over the
room's last messages both the calling agent and the notified channel may read.
An intelligence channel shared into the child room stops the previous tasks and
the conversation, not `context`.
`DelegateHandler(cache=CompletedTaskCache())` answers a repeat, the same agent's
call of the same task with the same `context`, from what its completed task
returned; a call handing the conversation over always runs.

The child room belongs to the parent room's organization: a tool in it reads
the tenant from `current_tool_room().organization_id`, as in the parent, and
`get_room(child_id, organization_id=...)` finds it for that organization only.

## Memory Providers

Agents use memory providers to maintain conversation context across handoffs:

```python
from roomkit.memory import SlidingWindowMemory
from roomkit.orchestration import HandoffMemoryProvider

agent = Agent(
    "agent",
    provider=provider,
    memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=50)),
)
```

`HandoffMemoryProvider` wraps any memory provider to inject handoff context (summary from the previous agent) into the conversation history.
