"""An organization's agents on a pool of generic executors, carried by Redis
(RFC §10.6.7).

No executor is told which agents it runs. Each declares a rule, every agent
whose id starts with ``acme-``, and builds an agent from its configuration at
the agent's first turn there: here a dict stands for the service that holds
the organization's agents, the same for every pool process. Each keeps one
agent built at most (``max_channels=1``, kept low to show it), and says on
stderr each one it builds. The room's owner (this process) holds each agent
by its declaration alone:

1. Two pool processes start, each declaring ``acme-*``.
2. Three agents are created on the fly, each in a room of its own: attaching
   them builds nothing. Their first turns spread over the pool, and each
   room's next turns stay where its first one ran; a pool running two of
   them builds each anew for its turn, past its bound.
3. ``acme-legal``, which the service has no configuration for, fails with
   ``no_executor``, as does ``globex-sales``, under no executor's rule.
4. One pool process is killed. Once its silence shows (the heartbeat
   timeout), the next turns of its rooms run on the other.

Requirements:
    pip install roomkit[redis]
    a Redis server: docker run --rm -d -p 6379:6379 redis:7, or set REDIS_URL

Run with:
    uv run python examples/turns_executor_pool.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import (
    Agent,
    AgentIdentity,
    Channel,
    ChannelCategory,
    ChannelDeclaration,
    EventType,
    InboundMessage,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.providers.ai import MockAIProvider
from roomkit.turns import TurnExecutor, TurnTimeouts

try:
    from redis.asyncio import from_url
    from redis.exceptions import RedisError

    from roomkit import RedisTurnBackend
except ImportError:
    print("Error: install the redis extra: pip install roomkit[redis]")
    sys.exit(1)

logger = setup_logging("example.turns_executor_pool")

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
# Every process keeps the same bounds: short ones, so the loss shows quickly.
TIMEOUTS = TurnTimeouts(heartbeat_interval=0.5, heartbeat_timeout=2.0, accept=5.0, cancel=2.0)
# What each pool executor declares: the organization's agents, whichever.
RULE = "acme-*"
# How many agents a pool executor keeps built: low, so its letting go shows.
MAX_CHANNELS = 1


# -- A pool process: builds any agent the service knows ---------------------------


# The service holding the organization's agents, as a pool reads it.
CONFIGURED = {
    "acme-sales": "You sell Acme's plans.",
    "acme-support": "You fix Acme's accounts.",
    "acme-billing": "You explain Acme's invoices.",
}


def build_agent(pool_id: str) -> Callable[[str], Channel | None]:
    """The pool's factory: an agent from its configuration, or nothing."""

    def build(channel_id: str) -> Channel | None:
        instructions = CONFIGURED.get(channel_id)
        if instructions is None:
            # Unknown to the service: the turn fails with no_executor.
            return None
        answer = f"{channel_id} here, running on {pool_id}."
        logger.info("%s builds %s", pool_id, channel_id)
        return Agent(
            channel_id,
            provider=MockAIProvider(responses=[answer] * 10),
            system_prompt=instructions,
        )

    return build


async def run_pool(url: str, prefix: str, pool_id: str) -> None:
    """Serve every ``acme-`` agent until the owner says stop (or kills us)."""
    backend = RedisTurnBackend(url, key_prefix=prefix, timeouts=TIMEOUTS)
    executor = TurnExecutor(
        build_agent(pool_id),
        hosts={RULE},
        executor_id=pool_id,
        timeouts=TIMEOUTS,
        max_channels=MAX_CHANNELS,
    )
    await backend.serve(executor)
    print("ready", flush=True)
    try:
        await asyncio.to_thread(sys.stdin.readline)
    finally:
        await backend.close()
        await executor.close()


# -- The room owner's process -----------------------------------------------------


class PoolProcess:
    """This script run as ``pool``, in a process of its own."""

    def __init__(self, process: asyncio.subprocess.Process, pool_id: str) -> None:
        self.process = process
        self.pool_id = pool_id

    @classmethod
    async def start(cls, prefix: str, pool_id: str) -> PoolProcess:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            __file__,
            "pool",
            REDIS_URL,
            prefix,
            pool_id,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        assert process.stdout is not None
        line = await asyncio.wait_for(process.stdout.readline(), 30)
        if line.strip() != b"ready":
            raise RuntimeError(f"pool {pool_id} did not start")
        logger.info("%s declares %r, in process %d", pool_id, RULE, process.pid)
        return cls(process, pool_id)

    async def stop(self) -> None:
        if self.process.returncode is None:
            assert self.process.stdin is not None
            self.process.stdin.write(b"stop\n")
            await self.process.stdin.drain()
            await self.process.wait()

    async def kill(self) -> None:
        self.process.kill()
        await self.process.wait()
        logger.info("%s killed", self.pool_id)


async def check_redis() -> None:
    """Exit with a clear message when Redis cannot be reached."""
    client = from_url(REDIS_URL)
    try:
        await client.ping()
    except (RedisError, OSError) as exc:
        where = urlsplit(REDIS_URL)  # host and port only: a URL may carry a password
        print(
            f"Error: cannot reach Redis at {where.hostname}:{where.port or 6379} ({exc}).\n"
            "Start one with: docker run --rm -d -p 6379:6379 redis:7, or set REDIS_URL."
        )
        sys.exit(1)
    finally:
        await client.aclose()


async def forget_run(prefix: str) -> None:
    """Delete what this run left in Redis: the killed pool's beat stays."""
    client = from_url(REDIS_URL)
    try:
        keys = [key async for key in client.scan_iter(f"{prefix}:*")]
        if keys:
            await client.delete(*keys)
    finally:
        await client.aclose()


async def create_agent(kit: RoomKit, agent_id: str) -> None:
    """An agent created on the fly: declared to the kit, in a room of its own."""
    kit.register_channel(ChannelDeclaration(channel_id=agent_id, identity=AgentIdentity()))
    await kit.create_room(room_id=agent_id)
    await kit.attach_channel(agent_id, "web")
    await kit.attach_channel(agent_id, agent_id, category=ChannelCategory.INTELLIGENCE)


async def ask(kit: RoomKit, agent_id: str, text: str) -> None:
    """A person writes in the agent's room; prints its answer, or why none came."""
    message = InboundMessage(channel_id="web", sender_id="marie", content=TextContent(body=text))
    result = await kit.process_inbound(message, room_id=agent_id)
    if result.error is not None:
        reason = getattr(result.error, "reason", type(result.error).__name__)
        print(f"  {agent_id:<13} no answer: {reason}")
        return
    answers = [
        event.content.body
        for event in await kit.store.list_events(agent_id)
        if event.source.channel_id == agent_id
        and event.type == EventType.MESSAGE
        and isinstance(event.content, TextContent)
    ]
    print(f"  {agent_id:<13} {answers[-1]}")


async def run_owner() -> None:
    await check_redis()
    prefix = f"roomkit:example:{uuid4().hex[:8]}"
    kit = RoomKit(turn_backend=RedisTurnBackend(REDIS_URL, key_prefix=prefix, timeouts=TIMEOUTS))
    kit.register_channel(WebSocketChannel("web"))
    pools = [await PoolProcess.start(prefix, pool_id) for pool_id in ("pool-1", "pool-2")]
    agents = list(CONFIGURED)
    try:
        logger.info("1. Agents created on the fly: their first turns spread over the pool")
        for agent_id in agents:
            await create_agent(kit, agent_id)
            await ask(kit, agent_id, "Hello?")

        logger.info("2. Each room's next turn stays where its first one ran")
        for agent_id in agents:
            await ask(kit, agent_id, "Still there?")

        logger.info("3. Agents no executor serves")
        for agent_id in ("acme-legal", "globex-sales"):
            await create_agent(kit, agent_id)
            await ask(kit, agent_id, "Hello?")

        logger.info("4. pool-1 killed: once its silence shows, its rooms run on pool-2")
        await pools[0].kill()
        await asyncio.sleep(TIMEOUTS.heartbeat_timeout + TIMEOUTS.heartbeat_interval)
        for agent_id in agents:
            await ask(kit, agent_id, "And now?")
    finally:
        for pool in pools:
            await pool.stop()
        await kit.close()
        await forget_run(prefix)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pool":
        asyncio.run(run_pool(*sys.argv[2:5]))
    else:
        asyncio.run(run_owner())
