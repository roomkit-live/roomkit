"""A discussion's pilot in a process of its own, for
tests/orchestration/test_discussion_process_death.py.

``python tests/postgres_discussion_pilot.py <dsn> <room id>`` opens a
PostgresStore and a PostgresAdvisoryLockManager on *dsn*, as another process
of the deployment does, creates the room with a Discussion of
``investigator`` and ``sre``, binds the transport ``ops``, and prints
``installed <process id>``: the holder its lease names. The lease lasts
``LEASE`` seconds and the queue is read again every ``POLL`` seconds, the
test's time scale. A turn it gives prints ``generating <agent>`` and never
ends, until the process is killed.
"""

from __future__ import annotations

import asyncio
import sys

from roomkit import (
    AIChannel,
    Channel,
    ChannelBinding,
    ChannelOutput,
    ChannelType,
    Discussion,
    EventSource,
    InboundMessage,
    RoomContext,
    RoomEvent,
    RoomKit,
)
from roomkit.orchestration.strategies.discussion import _driver, _shared
from roomkit.providers.ai import AIContext, AIResponse, MockAIProvider
from roomkit.store import PostgresAdvisoryLockManager, PostgresStore

AGENTS = ("investigator", "sre")
LEASE = 1.0
POLL = 0.05


class _Transport(Channel):
    channel_type = ChannelType.WEBSOCKET

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            source=EventSource(channel_id=self.channel_id, channel_type=self.channel_type),
            content=message.content,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


class _Holding(MockAIProvider):
    """Generates forever, once it said so."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.who = name

    async def generate(self, context: AIContext) -> AIResponse:
        print(f"generating {self.who}", flush=True)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


async def main(dsn: str, room_id: str) -> None:
    # The test's time scale, as tests/orchestration's fixtures set it.
    _driver.POLL_SECONDS = POLL
    _shared.LEASE_SECONDS = LEASE
    _shared._RENEW_BEFORE = LEASE * 2 / 3
    store = PostgresStore(dsn=dsn)
    await store.init(min_size=1, max_size=3)
    locks = PostgresAdvisoryLockManager(dsn, max_size=8)
    await locks.init()
    kit = RoomKit(store=store, lock_manager=locks)
    kit.register_channel(_Transport("ops"))
    discussion = Discussion([AIChannel(name, provider=_Holding(name)) for name in AGENTS])
    await kit.create_room(room_id=room_id, orchestration=discussion)
    await kit.attach_channel(room_id, "ops")
    print(f"installed {kit._instance_id}", flush=True)
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:3]))
