"""Redis-backed persistent delivery.

Demonstrates ``RedisDeliveryBackend`` for persistent, distributed
delivery.  Items survive process restarts and are distributed across
workers via Redis Streams consumer groups.

``kit.deliver()`` adds the item to a Redis Stream and returns ``queued``;
the worker started by ``async with kit`` reads it through its consumer
group, publishes the content into the room (a Claude agent tells the user
about it) and fires ``AFTER_DELIVER``. The script prints the stream depth
before and after, and waits for every ``AFTER_DELIVER``.

Requires a running Redis instance, an Anthropic key and::

    pip install roomkit[redis,anthropic]

Start a throwaway Redis with:
    docker run --rm -d --name roomkit-redis -p 6379:6379 redis:7

Run with:
    ANTHROPIC_API_KEY=sk-... uv run python examples/delivery_redis.py

Environment variables:
    ANTHROPIC_API_KEY  Anthropic API key (required)
    REDIS_URL          Redis URL (default: redis://localhost:6379)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import require_env, setup_logging

from roomkit import (
    Agent,
    ChannelCategory,
    HookExecution,
    HookTrigger,
    RoomKit,
    WaitForIdle,
    WebSocketChannel,
)
from roomkit.delivery import RedisDeliveryBackend
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent, TextContent
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig

try:
    import redis.asyncio as aioredis
    from redis.exceptions import RedisError
except ImportError:
    print("Error: install the redis extra: pip install roomkit[redis]")
    sys.exit(1)

logger = setup_logging("delivery_redis")
logging.getLogger("roomkit.delivery").setLevel(logging.DEBUG)

NOTIFICATIONS = [
    "Background job 17 finished: the nightly report is ready.",
    "Background job 18 failed: the export server timed out.",
]


async def connect_redis(url: str) -> aioredis.Redis:
    """Return a connected client, or exit with a clean message."""
    client = aioredis.from_url(url)
    try:
        await client.ping()
    except (RedisError, OSError) as exc:
        await client.aclose()
        # Print host and port only: a Redis URL may carry a password.
        where = urlsplit(url)
        print(
            f"Error: cannot reach Redis at {where.hostname}:{where.port or 6379} ({exc}).\n"
            "Start one with: docker run --rm -d -p 6379:6379 redis:7, or set REDIS_URL."
        )
        sys.exit(1)
    return client


async def wait_for_empty_stream(backend: RedisDeliveryBackend, timeout: float = 5.0) -> None:
    """The stream entry is deleted on ack, just after AFTER_DELIVER fires."""
    async with asyncio.timeout(timeout):
        while await backend.get_queue_depth():
            await asyncio.sleep(0.05)


async def main() -> None:
    env = require_env("ANTHROPIC_API_KEY")
    client = await connect_redis(os.environ.get("REDIS_URL", "redis://localhost:6379"))

    config = AnthropicConfig(
        api_key=env["ANTHROPIC_API_KEY"],
        model="claude-haiku-5-5",
    )

    assistant = Agent(
        "agent-assistant",
        provider=AnthropicAIProvider(config),
        role="Assistant",
        system_prompt=(
            "You receive background job results. Tell the user about each one in one sentence."
        ),
    )

    # Redis-backed delivery: items persist in Redis Streams. The client is
    # injected, so the backend does not close it; this script does.
    backend = RedisDeliveryBackend(client=client)

    kit = RoomKit(delivery_strategy=WaitForIdle(buffer=1.0), delivery_backend=backend)

    delivered = asyncio.Event()
    outcomes: list[str] = []

    @kit.hook(HookTrigger.BEFORE_DELIVER, execution=HookExecution.ASYNC)
    async def on_before(event: RoomEvent, ctx: RoomContext) -> None:
        logger.info("BEFORE_DELIVER: %s", event.content)

    @kit.hook(HookTrigger.AFTER_DELIVER, execution=HookExecution.ASYNC)
    async def on_after(event: RoomEvent, ctx: RoomContext) -> None:
        status = event.metadata.get("delivery_outcome", {}).get("status", "unknown")
        error = event.metadata.get("error")
        if error:
            logger.error("AFTER_DELIVER status=%s error=%s", status, error)
        else:
            logger.info("AFTER_DELIVER status=%s", status)
        outcomes.append(status)
        if len(outcomes) == len(NOTIFICATIONS):
            delivered.set()

    ws = WebSocketChannel("ws-user")
    kit.register_channel(ws)
    kit.register_channel(assistant)

    async def on_user_receives(_conn: str, event: RoomEvent) -> None:
        if isinstance(event.content, TextContent):
            logger.info("User sees [%s]: %s", event.source.channel_id, event.content.body)

    ws.register_connection("user-conn", on_user_receives, room_id="demo")

    try:
        async with kit:  # creates the consumer group and starts the worker
            await kit.create_room(room_id="demo")
            await kit.attach_channel("demo", "ws-user")
            await kit.attach_channel(
                "demo", "agent-assistant", category=ChannelCategory.INTELLIGENCE
            )

            logger.info("Stream depth before: %d", await backend.get_queue_depth())

            for text in NOTIFICATIONS:
                outcome = await kit.deliver("demo", text, channel_id="ws-user")
                logger.info(
                    "kit.deliver() -> %s (item %s)", outcome.status, outcome.delivery_item_id
                )

            logger.info("Stream depth after enqueue: %d", await backend.get_queue_depth())

            await asyncio.wait_for(delivered.wait(), timeout=60.0)
            await wait_for_empty_stream(backend)
            logger.info("Stream depth after the worker ran: %d", await backend.get_queue_depth())
    finally:
        await client.aclose()

    if any(status != "sent" for status in outcomes):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
