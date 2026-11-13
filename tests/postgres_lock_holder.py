"""A kit in a process of its own, for tests/test_postgres_multiprocess.py.

``ROOMKIT_TEST_HOLDER_DSN=<dsn> python tests/postgres_lock_holder.py <room id>
room-lock|delivery-claim`` opens a PostgresStore and a PostgresAdvisoryLockManager
on the DSN, as another process of the deployment does, and sends one message to
the room, whose channels ``src`` and ``out`` the test attached. It then holds,
until it is killed or its standard input closes (the test's process ended):

- ``room-lock``: the room's lock, in a sync BEFORE_BROADCAST hook that never
  returns, before the commit; it prints ``holding`` from the hook;
- ``delivery-claim``: the room's delivery claim, in ``out``'s delivery of the
  committed message, which never returns; it prints ``delivering <index>``
  from the delivery.
"""

from __future__ import annotations

import asyncio
import os
import sys
import traceback

from roomkit import (
    Channel,
    ChannelBinding,
    ChannelOutput,
    ChannelType,
    EventSource,
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomContext,
    RoomEvent,
    RoomKit,
    TextContent,
)
from roomkit.store import PostgresAdvisoryLockManager, PostgresStore

# Longer than any test waits: neither the hook nor the locked phase ends the hold.
_HOLD_FOR = 600.0


class _Transport(Channel):
    """Turns an inbound message into a RoomEvent; holds the delivery it is
    handed when *holds_delivery*."""

    channel_type = ChannelType.WEBSOCKET

    def __init__(self, channel_id: str, *, holds_delivery: bool = False) -> None:
        super().__init__(channel_id)
        self._holds_delivery = holds_delivery

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            source=EventSource(channel_id=self.channel_id, channel_type=self.channel_type),
            content=message.content,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if self._holds_delivery:
            print(f"delivering {event.index}", flush=True)
            await asyncio.Event().wait()
        return ChannelOutput.empty()


async def main(dsn: str, room_id: str, hold: str) -> None:
    store = PostgresStore(dsn=dsn)
    await store.init(min_size=1, max_size=2)
    locks = PostgresAdvisoryLockManager(dsn, max_size=4)
    await locks.init()
    kit = RoomKit(store=store, lock_manager=locks, process_timeout=_HOLD_FOR)
    kit.register_channel(_Transport("src"))
    kit.register_channel(_Transport("out", holds_delivery=hold == "delivery-claim"))
    if hold == "room-lock":

        @kit.hook(HookTrigger.BEFORE_BROADCAST, timeout=_HOLD_FOR)
        async def hold_the_room(event: RoomEvent, context: RoomContext) -> HookResult:
            print("holding", flush=True)
            await asyncio.Event().wait()
            return HookResult.allow()

    message = InboundMessage(
        channel_id="src", sender_id="u1", content=TextContent(body="from the killed process")
    )
    holding = asyncio.ensure_future(kit.process_inbound(message, room_id=room_id))
    # The test's end of the pipe closes with its process, killed or not: the
    # holder never outlives it.
    closed = asyncio.ensure_future(asyncio.to_thread(sys.stdin.readline))
    done, _ = await asyncio.wait({holding, closed}, return_when=asyncio.FIRST_COMPLETED)
    if holding in done and (error := holding.exception()) is not None:
        traceback.print_exception(error)
    # At once: the thread reading stdin would hold a normal exit until it closes.
    os._exit(0 if closed in done else 1)


if __name__ == "__main__":
    asyncio.run(main(os.environ["ROOMKIT_TEST_HOLDER_DSN"], *sys.argv[1:3]))
