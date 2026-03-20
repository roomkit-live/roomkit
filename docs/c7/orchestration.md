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
    producer=writer_agent,
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

```python
from roomkit.orchestration.state import get_conversation_state

room = await kit.get_room("support")
state = get_conversation_state(room)

print(state.phase)            # Current phase (e.g., "handling")
print(state.active_agent_id)  # Which agent is active
print(state.handoff_count)    # Number of handoffs so far

# Phase transition history
for t in state.phase_history:
    print(f"{t.from_phase} -> {t.to_phase} ({t.reason})")
```

## Conversation Router (Advanced)

For custom routing logic beyond strategies:

```python
from roomkit.orchestration.router import ConversationRouter, RoutingRule, RoutingConditions

router = ConversationRouter(
    rules=[
        RoutingRule(
            agent_id="billing-agent",
            conditions=RoutingConditions(phases=["billing"]),
            priority=0,
        ),
        RoutingRule(
            agent_id="shipping-agent",
            conditions=RoutingConditions(phases=["shipping"]),
            priority=0,
        ),
    ],
    default_agent_id="triage-agent",
)
```

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

kit.status_bus.subscribe(on_status)
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
