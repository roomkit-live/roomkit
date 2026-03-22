# Identity and Realtime Events

## Identity Resolution

The identity pipeline maps external sender IDs to known participants. It runs as part of the inbound pipeline, after `handle_inbound()` and before hooks.

### How It Works

```
Inbound message arrives with sender_id
  -> IdentityResolver.resolve(sender_id, channel_type)
  -> Returns IdentityResult with status:
     IDENTIFIED      -> participant_id stamped on event, processing continues
     AMBIGUOUS       -> ON_IDENTITY_AMBIGUOUS hook fires
     PENDING         -> ON_IDENTITY_AMBIGUOUS hook fires
     UNKNOWN         -> ON_IDENTITY_UNKNOWN hook fires
     REJECTED        -> ON_IDENTITY_UNKNOWN hook fires
```

### Identity Hooks

```python
from roomkit import RoomKit, HookTrigger
from roomkit.models.identity import IdentityHookResult, Identity

kit = RoomKit()

@kit.identity_hook(HookTrigger.ON_IDENTITY_UNKNOWN)
async def handle_unknown(event, ctx):
    # Option 1: Resolve to a known identity
    return IdentityHookResult.resolved(Identity(
        id="user-123",
        display_name="Alice",
    ))

    # Option 2: Challenge the sender to identify
    return IdentityHookResult.challenge(inject=InjectedEvent(
        content=TextContent(body="Please provide your account number."),
    ))

    # Option 3: Reject the message
    return IdentityHookResult.reject("Unknown sender")

    # Option 4: Keep as pending
    return IdentityHookResult.pending(candidates=[...])
```

### Custom Identity Resolver

```python
from roomkit.identity.base import IdentityResolver
from roomkit.models.identity import IdentityResult, Identity
from roomkit.models.enums import IdentificationStatus

class DatabaseIdentityResolver(IdentityResolver):
    async def resolve(self, message: InboundMessage, context: RoomContext) -> IdentityResult:
        user = await db.find_by_phone(message.sender_id)
        if user:
            return IdentityResult(
                status=IdentificationStatus.IDENTIFIED,
                identity=Identity(id=user.id, display_name=user.name),
            )
        return IdentityResult(status=IdentificationStatus.UNKNOWN)

kit = RoomKit(identity_resolver=DatabaseIdentityResolver())
```

### Manual Resolution

```python
# Resolve a pending participant to a known identity
await kit.resolve_participant(
    room_id="room-1",
    participant_id="pending-123",
    identity_id="user-456",
)
```

## Realtime Ephemeral Events

Ephemeral events (typing, presence, reactions) are not stored in conversation history. They're delivered in real-time to subscribers.

### Publishing Events

```python
from roomkit import RoomKit

kit = RoomKit()

# Typing indicator
await kit.publish_typing("room-1", "alice", is_typing=True)
await kit.publish_typing("room-1", "alice", is_typing=False)

# Presence
await kit.publish_presence("room-1", "alice", "online")   # online/away/offline

# Reaction
await kit.publish_reaction("room-1", "alice", target_event_id="evt-123", emoji="thumbsup")

# Read receipt
await kit.publish_read_receipt("room-1", "alice", event_id="evt-123")

# Tool call events (AIChannel publishes these automatically)
from roomkit.realtime.base import EphemeralEventType

await kit.publish_tool_call("room-1", "ai-agent", [
    {"id": "tc1", "name": "search", "arguments": {"q": "test"}}
], EphemeralEventType.TOOL_CALL_START)
```

### Subscribing to Events

```python
async def on_ephemeral(event):
    print(f"Ephemeral: {event.type} from {event.user_id}")

sub_id = await kit.subscribe_room("room-1", on_ephemeral)

# Unsubscribe later
await kit.unsubscribe_room(sub_id)
```

### Event Types

| Type | Description |
|------|-------------|
| `TYPING_START` | User started typing |
| `TYPING_STOP` | User stopped typing |
| `PRESENCE_ONLINE` | User came online |
| `PRESENCE_AWAY` | User went away |
| `PRESENCE_OFFLINE` | User went offline |
| `READ_RECEIPT` | User read a message |
| `REACTION` | User reacted to a message |
| `TOOL_CALL_START` | AI started executing a tool |
| `TOOL_CALL_END` | AI finished executing a tool |
| `CUSTOM` | Custom ephemeral event |

### Read Tracking

```python
# Mark a specific event as read
await kit.mark_read("room-1", "ws-user", "evt-123")

# Mark all events as read
await kit.mark_all_read("room-1", "ws-user")
```

### Custom Realtime Backend

The default `InMemoryRealtime` works for single-process deployments. For distributed systems, implement `RealtimeBackend`:

```python
from roomkit.realtime.base import RealtimeBackend

class RedisRealtimeBackend(RealtimeBackend):
    # Implement publish, subscribe, unsubscribe using Redis Pub/Sub
    ...

kit = RoomKit(realtime=RedisRealtimeBackend())
```
