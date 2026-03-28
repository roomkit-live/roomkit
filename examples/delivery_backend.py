"""Persistent delivery with InMemoryDeliveryBackend.

Demonstrates the delivery backend pattern — enqueue/dequeue lifecycle
with a background worker. Items are enqueued by ``kit.deliver()`` and
executed asynchronously by the worker loop.

This example uses the in-memory backend (no external deps). For
production, swap with ``RedisDeliveryBackend``.

Run with:
    uv run python examples/delivery_backend.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import (
    Agent,
    CLIChannel,
    HookExecution,
    HookTrigger,
    InMemoryDeliveryBackend,
    RoomKit,
    Supervisor,
    WaitForIdle,
)
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.providers.ai.mock import MockAIProvider

logger = setup_logging("delivery_backend")
logging.getLogger("roomkit.delivery").setLevel(logging.DEBUG)


async def main() -> None:
    supervisor = Agent(
        "agent-supervisor",
        provider=MockAIProvider(responses=["Let me check.", "Here are the results."]),
        role="Supervisor",
        system_prompt="You coordinate work.",
    )

    worker = Agent(
        "agent-worker",
        provider=MockAIProvider(responses=["Analysis complete: all systems green."]),
        role="Analyst",
    )

    # InMemoryDeliveryBackend: items are enqueued and processed by a
    # background worker task.  Replace with RedisDeliveryBackend for
    # production persistence.
    backend = InMemoryDeliveryBackend()

    kit = RoomKit(
        delivery_strategy=WaitForIdle(buffer=1.0),
        delivery_backend=backend,
        orchestration=Supervisor(
            supervisor=supervisor,
            workers=[worker],
            strategy="sequential",
            auto_delegate=True,
        ),
    )

    @kit.hook(HookTrigger.AFTER_DELIVER, execution=HookExecution.ASYNC)
    async def on_delivered(event: RoomEvent, ctx: RoomContext) -> None:
        error = event.metadata.get("error")
        status = "FAILED" if error else "OK"
        logger.info("Delivery %s: %s", status, error or "success")

    cli = CLIChannel("cli")
    kit.register_channel(cli)

    async with kit:
        await kit.create_room(room_id="demo")
        await kit.attach_channel("demo", "cli")

        depth = await backend.get_queue_depth()
        logger.info("Queue depth at start: %d", depth)

        await cli.run(
            kit,
            room_id="demo",
            welcome="=== Delivery Backend Demo ===\nType a message.\n",
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
