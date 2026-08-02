"""Distributed ephemeral events and status bus over Redis.

Demonstrates ``RedisRealtimeBackend`` (typing, presence, reactions
crossing process boundaries via Redis pub/sub) and ``RedisStatusBackend``
(multi-agent status bus with shared history).

Requires a running Redis instance and ``pip install roomkit[redis]``.

Run in two terminals:
    uv run python examples/realtime_redis.py listen
    uv run python examples/realtime_redis.py send

The listener prints every ephemeral event and status entry published by
the sender — from a different process, which ``InMemoryRealtime`` and
``InMemoryStatusBackend`` cannot do.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import RoomKit
from roomkit.orchestration import RedisStatusBackend
from roomkit.orchestration.status_bus import StatusBus, StatusEntry, StatusLevel
from roomkit.realtime import EphemeralEvent, RedisRealtimeBackend

logger = setup_logging("realtime_redis")

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
ROOM_ID = "demo"


def _make_kit() -> RoomKit:
    return RoomKit(
        realtime=RedisRealtimeBackend(REDIS_URL),
        status_bus=StatusBus(backend=RedisStatusBackend(REDIS_URL)),
    )


async def listen() -> None:
    kit = _make_kit()
    async with kit:

        async def on_event(event: EphemeralEvent) -> None:
            logger.info(
                "ephemeral: %s from %s | %s",
                event.type.value,
                event.user_id,
                event.data,
            )

        async def on_status(entry: StatusEntry) -> None:
            logger.info(
                "status: [%s] %s -> %s | %s",
                entry.agent_id,
                entry.action,
                entry.status,
                entry.detail,
            )

        await kit.subscribe_room(ROOM_ID, on_event)
        await kit.status_bus.subscribe(on_status)

        history = await kit.status_bus.recent_text(5)
        logger.info("Recent status history:\n%s", history)

        logger.info("Listening on room %r (Ctrl-C to stop)...", ROOM_ID)
        await asyncio.Event().wait()


async def send() -> None:
    kit = _make_kit()
    async with kit:
        logger.info("Publishing ephemeral events to room %r...", ROOM_ID)
        await kit.publish_typing(ROOM_ID, "alice", is_typing=True)
        await asyncio.sleep(1.0)
        await kit.publish_typing(ROOM_ID, "alice", is_typing=False)
        await kit.publish_presence(ROOM_ID, "alice", "online")
        await kit.publish_reaction(ROOM_ID, "alice", "evt-1", "+1")

        logger.info("Posting to the shared status bus...")
        await kit.status_bus.post_async(
            "exec", "search_google", StatusLevel.OK, detail="Found 7 results"
        )
        await kit.status_bus.post_async(
            "system", "task_completed", StatusLevel.COMPLETED, detail="Done in 2s"
        )

        # Let the pub/sub round-trip finish before tearing down.
        await asyncio.sleep(0.5)
        logger.info("Sent. Check the listener terminal.")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("listen", "send"):
        print(f"Usage: {sys.argv[0]} listen|send")
        raise SystemExit(1)
    try:
        asyncio.run(listen() if mode == "listen" else send())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
