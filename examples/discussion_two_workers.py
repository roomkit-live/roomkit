"""One discussion served by two worker processes (RFC §19.7.5 rule 16).

An application often runs several workers behind a load balancer, all on one
Postgres (``PostgresStore`` and ``PostgresAdvisoryLockManager``). Here two
kits share one store and one lock manager to play those two workers, each
with its own agent objects. Three scenes:

1. A message reaches worker B, where the host never installed the
   discussion. B follows the discussion stored with the room: it asks no
   agent at broadcast and queues the turns, which worker A gives one at a
   time. Before the discussion was stored with the room, B solicited every
   agent at once and they answered each other until the depth limit (45
   messages for one question in the same setup).
2. Two messages reach the two workers at once. One queue in the store, one
   worker giving the turns under the room's lease: the agents speak one
   after the other. Before, each worker kept its own queue and two agents
   spoke at the same time.
3. Worker A stops without a word (a crash). Its lease expires and worker B,
   which installed the discussion too, takes the turns over.

The models are scripted mocks, so the example runs without keys.

Run with:
    uv run python examples/discussion_two_workers.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import Discussion, InboundMessage, RoomKit, TextContent, WebSocketChannel
from roomkit.channels.ai import AIChannel
from roomkit.core.locks import InMemoryLockManager
from roomkit.models.enums import EventType
from roomkit.providers.ai.base import AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.store.memory import InMemoryStore

logger = setup_logging("example.discussion_two_workers")

ROOM = "incident-room"
AGENTS = ("investigator", "sre", "comms")
START = time.monotonic()


class Timed(MockAIProvider):
    """A scripted model that logs when it starts and stops generating."""

    def __init__(self, who: str, worker: str, seconds: float = 0.5) -> None:
        super().__init__(ai_responses=[AIResponse(content=f"{who} here")])
        self.who, self.worker, self.seconds = who, worker, seconds

    async def generate(self, context):  # type: ignore[no-untyped-def]
        logger.info("  %5.2fs  %s starts (on worker %s)", clock(), self.who, self.worker)
        await asyncio.sleep(self.seconds)
        logger.info("  %5.2fs  %s ends", clock(), self.who)
        return await super().generate(context)


def clock() -> float:
    return time.monotonic() - START


def discussion(worker: str) -> Discussion:
    """The discussion as one worker builds it: its own agent objects."""
    return Discussion([AIChannel(a, provider=Timed(a, worker)) for a in AGENTS])


async def worker(name: str, store: InMemoryStore, locks: InMemoryLockManager) -> RoomKit:
    kit = RoomKit(store=store, lock_manager=locks)
    kit.register_channel(WebSocketChannel("ops"))
    logger.info("worker %s up", name)
    return kit


async def say(kit: RoomKit, body: str) -> None:
    logger.info("  %5.2fs  ops: %s", clock(), body)
    await kit.process_inbound(
        InboundMessage(channel_id="ops", sender_id="ops", content=TextContent(body=body))
    )


async def answers(store: InMemoryStore) -> list[str]:
    events = await store.list_events(ROOM, limit=200)
    return [
        e.source.channel_id
        for e in events
        if e.type == EventType.MESSAGE and e.source.channel_id in AGENTS
    ]


async def until(store: InMemoryStore, count: int) -> None:
    async with asyncio.timeout(30):
        while len(await answers(store)) < count:
            await asyncio.sleep(0.05)


async def main() -> None:
    store, locks = InMemoryStore(), InMemoryLockManager()
    a = await worker("A", store, locks)
    await a.create_room(room_id=ROOM, orchestration=discussion("A"))
    await a.attach_channel(ROOM, "ops")
    b = await worker("B", store, locks)

    logger.info("1. a message reaches worker B, which only follows the discussion")
    await say(b, "checkout is failing, what happened?")
    await until(store, 3)
    logger.info("   answers: %s", await answers(store))

    logger.info("2. worker B installs the discussion too; two messages reach two workers")
    await discussion("B").install(b, ROOM)
    await say(a, "@investigator what do the logs say?")
    await say(b, "@sre and the metrics?")
    await until(store, 5)

    logger.info("3. worker A crashes; B takes the turns over once A's lease expires (15 s)")
    held = a._discussions[ROOM]
    held.closed = True
    held.driver._task.cancel()  # no stop(): the lease is not given back
    await say(b, "@comms post a status update")
    await until(store, 6)
    logger.info("   answers: %s", await answers(store))
    await b.close()


if __name__ == "__main__":
    asyncio.run(main())
