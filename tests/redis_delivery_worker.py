"""A delivery worker in a process of its own, for tests/delivery/test_redis_integration.py.

``python tests/redis_delivery_worker.py <redis url> <stream prefix> <claim idle
seconds> <sqlite path>`` runs a kit on the SQLite store the test shares, its
deliveries queued on a RedisDeliveryBackend, and starts the kit's worker. The
worker takes a batch of the queued items, and the delivery of the first never
ends: a BEFORE_DELIVER hook prints ``delivering <item id>`` and holds it, so
the worker acks no item of the batch until it is killed, while it renews
their leases.
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
)
from roomkit.delivery import RedisDeliveryBackend
from roomkit.store import SQLiteStore


class _Transport(Channel):
    channel_type = ChannelType.SMS

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


async def main(url: str, prefix: str, claim_idle: str, database: str) -> None:
    backend = RedisDeliveryBackend(
        url, stream_prefix=prefix, claim_idle_seconds=float(claim_idle), retry_delay=0
    )
    kit = RoomKit(store=SQLiteStore(database), delivery_backend=backend)
    kit.register_channel(_Transport("out"))

    @kit.hook(HookTrigger.BEFORE_DELIVER, timeout=600)
    async def hold(event: RoomEvent, context: RoomContext) -> HookResult:
        print(f"delivering {event.metadata['delivery_item_id']}", flush=True)
        await asyncio.Event().wait()
        return HookResult.allow()

    async with kit:
        await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:5]))
