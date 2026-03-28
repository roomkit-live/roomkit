"""Redis-backed persistent delivery.

Demonstrates ``RedisDeliveryBackend`` for persistent, distributed
delivery.  Items survive process restarts and are distributed across
workers via Redis Streams consumer groups.

Requires a running Redis instance and ``pip install roomkit[redis]``.

Run with:
    ANTHROPIC_API_KEY=sk-... uv run python examples/delivery_redis.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import require_env, setup_logging

from roomkit import (
    Agent,
    CLIChannel,
    HookExecution,
    HookTrigger,
    RoomKit,
    Supervisor,
    WaitForIdle,
)
from roomkit.delivery import RedisDeliveryBackend
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig

logger = setup_logging("delivery_redis")
logging.getLogger("roomkit.delivery").setLevel(logging.DEBUG)


async def main() -> None:
    env = require_env("ANTHROPIC_API_KEY")

    config = AnthropicConfig(
        api_key=env["ANTHROPIC_API_KEY"],
        model="claude-haiku-4-5-20251001",
    )

    supervisor = Agent(
        "agent-supervisor",
        provider=AnthropicAIProvider(config),
        role="Supervisor",
        system_prompt="You coordinate analysis. Present combined results.",
    )

    researcher = Agent(
        "agent-researcher",
        provider=AnthropicAIProvider(config),
        role="Researcher",
        system_prompt="Research the given topic. Be concise (3-4 points).",
    )

    # Redis-backed delivery: items persist in Redis Streams
    backend = RedisDeliveryBackend("redis://localhost:6379")

    kit = RoomKit(
        delivery_strategy=WaitForIdle(buffer=1.0),
        delivery_backend=backend,
        orchestration=Supervisor(
            supervisor=supervisor,
            workers=[researcher],
            strategy="sequential",
            auto_delegate=True,
        ),
    )

    @kit.hook(HookTrigger.BEFORE_DELIVER, execution=HookExecution.ASYNC)
    async def on_before(event: RoomEvent, ctx: RoomContext) -> None:
        logger.info("before_deliver: %s", event.content)

    @kit.hook(HookTrigger.AFTER_DELIVER, execution=HookExecution.ASYNC)
    async def on_after(event: RoomEvent, ctx: RoomContext) -> None:
        error = event.metadata.get("error")
        if error:
            logger.error("after_deliver FAILED: %s", error)
        else:
            logger.info("after_deliver OK")

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
            welcome="=== Redis Delivery Demo ===\nType a message.\n",
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
