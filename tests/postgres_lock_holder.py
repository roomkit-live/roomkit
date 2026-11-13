"""A kit in a process of its own, for tests/test_postgres_multiprocess.py.

``python tests/postgres_lock_holder.py <dsn> <room id> room-lock|delivery-claim``
opens a PostgresStore and a PostgresAdvisoryLockManager on *dsn*, as another
process of the deployment does, and sends one message to the room, whose
channels ``src`` and ``out`` the test attached. It then holds, until it is
killed:

- ``room-lock``: the room's lock, in a sync BEFORE_BROADCAST hook that never
  returns, before the commit; it prints ``holding`` from the hook;
- ``delivery-claim``: the room's delivery claim, in ``out``'s delivery of the
  committed message, which never returns; it prints ``delivering <index>``
  from the delivery.
"""

from __future__ import annotations

import asyncio
import sys

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
    kit = RoomKit(store=store, lock_manager=locks)
    kit.register_channel(_Transport("src"))
    kit.register_channel(_Transport("out", holds_delivery=hold == "delivery-claim"))
    if hold == "room-lock":

        @kit.hook(HookTrigger.BEFORE_BROADCAST, timeout=600)
        async def hold_the_room(event: RoomEvent, context: RoomContext) -> HookResult:
            print("holding", flush=True)
            await asyncio.Event().wait()
            return HookResult.allow()

    await kit.process_inbound(
        InboundMessage(
            channel_id="src", sender_id="u1", content=TextContent(body="from the killed process")
        ),
        room_id=room_id,
    )
    print("returned", flush=True)


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:4]))
