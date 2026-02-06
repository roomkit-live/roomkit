"""Circuit breaker and retry patterns.

Demonstrates RoomKit's built-in resilience primitives for handling
provider failures. Shows:
- CircuitBreaker: Fault isolation with closed/open/half-open states
- retry_with_backoff: Exponential backoff retry using RetryPolicy
- How these work together to handle flaky providers

Run with:
    uv run python examples/circuit_breaker_retry.py
"""

from __future__ import annotations

import asyncio
import time

from roomkit import RetryPolicy
from roomkit.core.circuit_breaker import CircuitBreaker
from roomkit.core.retry import retry_with_backoff


async def main() -> None:
    # =====================================================
    # Part 1: Circuit Breaker
    # =====================================================
    print("=== Circuit Breaker Demo ===\n")

    # Create a circuit breaker: trips after 3 failures, recovers after 2 seconds
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=2.0)

    print(f"Initial state: closed={cb.is_closed}, open={cb.is_open}")

    # Simulate successful requests
    print("\n--- Successful requests ---")
    for i in range(3):
        if cb.allow_request():
            cb.record_success()
            print(f"  Request {i + 1}: OK (closed={cb.is_closed})")

    # Simulate failures until the breaker trips
    print("\n--- Failing requests ---")
    for i in range(4):
        if cb.allow_request():
            cb.record_failure()
            print(f"  Request {i + 1}: FAILED (closed={cb.is_closed}, open={cb.is_open})")
        else:
            print(f"  Request {i + 1}: REJECTED (circuit is open)")

    # Breaker is now open — requests are rejected immediately
    print(f"\nCircuit state: closed={cb.is_closed}, open={cb.is_open}")
    allowed = cb.allow_request()
    print(f"Can we send a request? {allowed}")

    # Wait for recovery timeout
    print(f"\nWaiting 2 seconds for recovery timeout...")
    await asyncio.sleep(2.1)

    # Breaker is now half-open — allows one probe request
    print(f"After recovery: half_open={cb.is_half_open}")
    if cb.allow_request():
        print("  Probe request allowed (half-open)")
        # If the probe succeeds, breaker closes again
        cb.record_success()
        print(f"  Probe succeeded! closed={cb.is_closed}")

    # =====================================================
    # Part 2: Retry with Backoff
    # =====================================================
    print("\n\n=== Retry with Backoff Demo ===\n")

    call_count = 0

    async def flaky_operation() -> str:
        """Simulates an operation that fails twice then succeeds."""
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise ConnectionError(f"Connection refused (attempt {call_count})")
        return f"Success on attempt {call_count}"

    # Define a retry policy
    policy = RetryPolicy(
        max_retries=5,
        base_delay_seconds=0.1,   # Start with 100ms delay
        max_delay_seconds=2.0,    # Cap at 2s
        exponential_base=2.0,     # Double each retry
    )

    # Retry with exponential backoff
    start = time.monotonic()
    result = await retry_with_backoff(flaky_operation, policy)
    elapsed = time.monotonic() - start

    print(f"Result: {result}")
    print(f"Total attempts: {call_count}")
    print(f"Elapsed: {elapsed:.2f}s")

    # =====================================================
    # Part 3: Combined Pattern
    # =====================================================
    print("\n\n=== Combined: Circuit Breaker + Retry ===\n")

    cb2 = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
    retry_policy = RetryPolicy(max_retries=2, base_delay_seconds=0.05)
    attempt_log: list[str] = []

    async def send_with_resilience(message: str) -> str:
        """Send a message with circuit breaker protection and retry."""
        if not cb2.allow_request():
            attempt_log.append(f"REJECTED by circuit breaker: {message}")
            return "REJECTED"

        try:
            result = await retry_with_backoff(_simulate_send, retry_policy)
            cb2.record_success()
            attempt_log.append(f"OK: {message}")
            return result
        except ConnectionError:
            cb2.record_failure()
            attempt_log.append(f"FAILED (breaker failures={cb2._failure_count}): {message}")
            return "FAILED"

    send_count = 0

    async def _simulate_send() -> str:
        nonlocal send_count
        send_count += 1
        # Fail for the first 8 sends to trip the breaker
        if send_count <= 8:
            raise ConnectionError("Provider down")
        return "delivered"

    # Send messages — first few will fail and trip the breaker
    for i in range(6):
        result = await send_with_resilience(f"Message {i + 1}")
        print(f"  Message {i + 1}: {result}")

    print(f"\nAttempt log:")
    for entry in attempt_log:
        print(f"  {entry}")
    print(f"\nFinal circuit state: closed={cb2.is_closed}, open={cb2.is_open}")


if __name__ == "__main__":
    asyncio.run(main())
