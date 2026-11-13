"""A room owner in a process of its own, for tests/test_turns_redis.py.

``python tests/redis_turn_owner.py <redis url> <key prefix>`` builds a kit
whose turns go across Redis, the room ``r`` binding the transport ``tx`` and
the agent ``worker``, which an executor of another process serves. It
declares the host verb ``app.hold``, which prints ``called`` and never
answers, prints ``owner <node id>``, then sends the room one message and
waits on the turn it gives ``worker``, until it is killed.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from roomkit import (
    AgentIdentity,
    Channel,
    ChannelBinding,
    ChannelCategory,
    ChannelDeclaration,
    ChannelOutput,
    ChannelType,
    EventSource,
    InboundMessage,
    RoomContext,
    RoomEvent,
    RoomKit,
    TextContent,
)
from roomkit.turns import TurnTimeouts
from roomkit.turns.redis import RedisTurnBackend

# As tests/test_turns_redis.py's TIMEOUTS: the executor hears this owner beat
# every interval, and takes it for lost after the timeout.
TIMEOUTS = TurnTimeouts(heartbeat_interval=0.2, heartbeat_timeout=1.0, accept=3.0, cancel=1.0)


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


async def _hold(arguments: dict[str, Any]) -> None:
    print("called", flush=True)
    await asyncio.Event().wait()


async def main(url: str, prefix: str) -> None:
    backend = RedisTurnBackend(url, key_prefix=prefix, timeouts=TIMEOUTS)
    kit = RoomKit(turn_backend=backend)
    kit.register_channel(_Transport("tx"))
    kit.register_channel(ChannelDeclaration(channel_id="worker", identity=AgentIdentity()))
    kit.declare_host_verb("app.hold", _hold)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "tx")
    await kit.attach_channel("r", "worker", category=ChannelCategory.INTELLIGENCE)
    print(f"owner {backend.node_id}", flush=True)
    await kit.process_inbound(
        InboundMessage(channel_id="tx", sender_id="u", content=TextContent(body="Hold on.")),
        room_id="r",
    )
    print("returned", flush=True)


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:3]))
