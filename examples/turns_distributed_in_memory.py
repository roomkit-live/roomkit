"""A room's turn runs on an executor of another node (RFC §10.6.7, §10.6.8).

``DistributedTurnBackend`` carries an agent's turns between processes. It
holds the logic; two services carry what crosses, by their ports:

- a ``TurnBroker`` carries the messages, one inbox a node;
- a ``TurnRegistry`` keeps what the nodes share: their beats, which executor
  holds a room, the presence last published.

Every node given the same broker and registry reaches the others. Here both
nodes live in this process, on ``InMemoryTurnBroker`` and
``InMemoryTurnRegistry``. ``CountingBroker`` is a broker of your own in its
simplest form: it wraps the in-memory one and counts the messages of each
kind it carries. ``check_turn_broker`` and ``check_turn_registry``
(``roomkit.testing``) check a service of your own against the contract.

1. The worker's node serves an executor of the ``meteo`` agent.
2. The room's node knows the agent by its declaration alone, and lists the
   executor of the other node with its last beat.
3. A message: its turn runs on the worker's node, every message of the turn
   crossing the broker.

On Redis, ``RedisTurnBackend(url)`` is this backend with Redis's two
services, one per process; a service of your own (NATS, a database) gives
``DistributedTurnBackend(broker, registry)`` its own broker and registry.

Run with:
    uv run python examples/turns_distributed_in_memory.py
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Sequence

from roomkit import (
    Agent,
    AgentIdentity,
    BrokerMessage,
    ChannelCategory,
    ChannelDeclaration,
    DistributedTurnBackend,
    InboundMessage,
    InMemoryTurnBroker,
    InMemoryTurnRegistry,
    RoomKit,
    TextContent,
    TurnExecutor,
    WebSocketChannel,
)
from roomkit.providers.ai import MockAIProvider
from roomkit.testing import check_turn_broker, check_turn_registry


class CountingBroker(InMemoryTurnBroker):
    """The in-memory broker, counting the messages of each kind it carries."""

    def __init__(self) -> None:
        super().__init__()
        self.carried: Counter[str] = Counter()

    async def post(self, node: str, messages: Sequence[BrokerMessage]) -> None:
        self.carried.update(message.kind for message in messages)
        await super().post(node, messages)


def build_agent(channel_id: str) -> Agent | None:
    if channel_id == "meteo":
        return Agent("meteo", provider=MockAIProvider(responses=["Sunny, 21 °C."]))
    return None


async def main() -> None:
    # A service of your own keeps the contract the backend relies on.
    await check_turn_broker(CountingBroker())
    await check_turn_registry(InMemoryTurnRegistry())

    broker, registry = CountingBroker(), InMemoryTurnRegistry()
    rooms_node = DistributedTurnBackend(broker, registry)
    workers_node = DistributedTurnBackend(broker, registry)

    print("1. The worker's node serves the meteo agent:")
    executor = TurnExecutor(build_agent, hosts={"meteo"}, executor_id="meteo-worker")
    await workers_node.serve(executor)
    print(f"    node {workers_node.node_id}, executor {executor.executor_id}")

    kit = RoomKit(turn_backend=rooms_node)
    kit.register_channel(WebSocketChannel("chat"))
    kit.register_channel(ChannelDeclaration(channel_id="meteo", identity=AgentIdentity()))
    await kit.create_room(room_id="room")
    await kit.attach_channel("room", "chat")
    await kit.attach_channel("room", "meteo", category=ChannelCategory.INTELLIGENCE)

    print(f"2. The room's node ({rooms_node.node_id}) lists it:")
    for info in await kit.executors():
        beat = info.last_beat.isoformat(timespec="seconds") if info.last_beat else "-"
        print(
            f"    {info.executor_id} on node {info.node}, online {info.online}, last beat {beat}"
        )

    print("3. A message; its turn runs on the worker's node:")
    message = InboundMessage(channel_id="chat", sender_id="ana", content=TextContent(body="Hi?"))
    await kit.process_inbound(message, room_id="room")
    for event in await kit.store.list_events("room"):
        if event.source.channel_id == "meteo" and isinstance(event.content, TextContent):
            print(f"    meteo: {event.content.body}")
    print(f"    the broker carried: {dict(sorted(broker.carried.items()))}")

    await kit.close()
    await executor.close()
    await workers_node.close()


if __name__ == "__main__":
    asyncio.run(main())
