# Orchestration

RoomKit provides four declarative orchestration strategies for multi-agent workflows. Pass a strategy to `RoomKit(orchestration=...)` or `create_room(orchestration=...)` — agents, routing, handoff tools, and conversation state are wired automatically.

## Strategies

### Pipeline

Linear agent chain: triage -> handler -> resolver. Each agent can only hand off to the next in sequence.

```python
from roomkit import Agent, Pipeline, RoomKit, WebSocketChannel
from roomkit.providers.ai.mock import MockAIProvider

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

```python
from roomkit import Loop

kit = RoomKit(orchestration=Loop(
    agent=writer_agent,
    reviewer=editor_agent,
    max_iterations=3,
))
```

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
from roomkit.orchestration.state import get_conversation_state

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
from roomkit.orchestration.state import get_conversation_state, set_conversation_state

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

# Persist back to room
updated_room = set_conversation_state(room, updated_state)
await kit.store.update_room(updated_room)
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
from roomkit.orchestration.state import get_conversation_state, set_conversation_state

room = await kit.get_room("support")
state = get_conversation_state(room)

new_state = state.transition(
    to_phase="escalation",
    to_agent="supervisor-agent",
    reason="Customer requested manager",
    metadata={"escalation_priority": "high"},
)

updated_room = set_conversation_state(room, new_state)
await kit.store.update_room(updated_room)
```

### Using State in Hooks

```python
from roomkit import HookTrigger, HookResult, RoomEvent, RoomContext, TextContent
from roomkit.orchestration.state import get_conversation_state, set_conversation_state

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
    updated_room = set_conversation_state(ctx.room, updated_state)
    await kit.store.update_room(updated_room)
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
from roomkit.orchestration.router import ConversationRouter, RoutingRule, RoutingConditions

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
from roomkit.models.enums import ChannelType

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

### Installing the Router

```python
# Option 1: Manual hook
kit.hook(HookTrigger.BEFORE_BROADCAST, execution=HookExecution.SYNC, priority=-100)(router.as_hook())

# Option 2: install() — also sets up handoff tools
handler = router.install(kit, agents=[billing_agent, shipping_agent, triage_agent])
```

### Routing Selection Priority

1. **Agent affinity** — if `state.active_agent_id` is set and attached, stick with it
2. **Rules** — evaluate in ascending `priority` order; first match wins
3. **Fallback** — return `default_agent_id`
4. **Loop prevention** — events FROM intelligence channels are never routed

## Conversation Pipeline (Advanced)

Define stages explicitly:

```python
from roomkit.orchestration.pipeline import ConversationPipeline, PipelineStage

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

## Memory Providers

Agents use memory providers to maintain conversation context across handoffs:

```python
from roomkit.memory.sliding_window import SlidingWindowMemory
from roomkit.orchestration.handoff import HandoffMemoryProvider

agent = Agent(
    "agent",
    provider=provider,
    memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=50)),
)
```

`HandoffMemoryProvider` wraps any memory provider to inject handoff context (summary from the previous agent) into the conversation history.
