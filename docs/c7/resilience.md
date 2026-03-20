# Resilience

RoomKit provides built-in resilience patterns for production deployments: rate limiting, circuit breakers, retry with backoff, chain depth limits, and room lifecycle timers.

## Rate Limiting

Apply rate limits per channel binding:

```python
from roomkit import RoomKit, RateLimit

kit = RoomKit()

await kit.attach_channel("room-1", "sms-out",
    rate_limit=RateLimit(max_per_second=1.0, max_per_minute=30.0),
)
```

Framework-level inbound rate limiting:

```python
from roomkit import RoomKit, RateLimit

kit = RoomKit(inbound_rate_limit=RateLimit(max_per_second=10.0))
```

## Circuit Breaker

Isolate provider failures with circuit breakers:

```python
from roomkit.core.circuit_breaker import CircuitBreaker

cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)

if cb.allow_request():
    try:
        result = await provider.send(event, to="+1234567890")
        cb.record_success()
    except Exception:
        cb.record_failure()  # Opens after 5 consecutive failures
```

States: CLOSED (normal) -> OPEN (failing, fast-reject) -> HALF_OPEN (testing recovery).

## Retry with Backoff

Retry failed operations with exponential backoff:

```python
from roomkit import RetryPolicy
from roomkit.core.retry import retry_with_backoff

policy = RetryPolicy(
    max_retries=3,
    base_delay_seconds=1.0,
    max_delay_seconds=60.0,
)

result = await retry_with_backoff(flaky_function, policy)
```

Apply per binding:

```python
await kit.attach_channel("room-1", "sms-out",
    retry_policy=RetryPolicy(max_retries=3, base_delay_seconds=1.0),
)
```

## Chain Depth Limit

Prevents infinite AI-to-AI loops. Default max depth is 5.

```python
kit = RoomKit(max_chain_depth=3)  # Stricter limit
```

When an AI response triggers another AI response, chain depth increments. Processing stops when the limit is reached.

## Room Lifecycle Timers

Auto-transition rooms based on inactivity:

```python
from roomkit import RoomKit, RoomTimers

kit = RoomKit()

room = await kit.create_room(room_id="session-1")
# Configure timers on the room:
# inactive_after_seconds -> ACTIVE to PAUSED
# closed_after_seconds -> to CLOSED

# Check timers for one room
room = await kit.check_room_timers("session-1")

# Batch check all rooms (call periodically)
transitioned = await kit.check_all_timers()
```

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
from roomkit import RoomKit, RateLimit, RetryPolicy
from roomkit.store.postgres import PostgresStore

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
