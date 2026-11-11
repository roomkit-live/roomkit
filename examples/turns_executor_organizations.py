"""Executors that serve the organizations they declare (RFC §10.6.7, §17.2).

Two executors run the same agent, ``support``, held by the room's owner by
its declaration alone:

- ``acme-pool`` declares ``organizations={"acme"}``: it takes the turns of
  acme's rooms, and of no other;
- ``platform-pool`` declares no organization: it takes the turns of the rooms
  without one, and never a tenant's.

The agent is asked in a room of acme, in a room without an organization, and
in a room of globex, which no executor serves: that turn fails with
``no_executor``, and the agent reads offline there. An executor serving every
organization declares ``organizations={"*"}``.

Everything runs in this process (``InMemoryTurnBackend``); across processes,
``RedisTurnBackend`` carries the same declarations (examples/
turns_executor_pool.py).

Run with:
    uv run python examples/turns_executor_organizations.py
"""

from __future__ import annotations

import asyncio

from roomkit import (
    Agent,
    AgentIdentity,
    ChannelCategory,
    ChannelDeclaration,
    InboundMessage,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.providers.ai import MockAIProvider
from roomkit.turns import InMemoryTurnBackend, TurnExecutor

ROOMS = {"acme-room": "acme", "platform-room": None, "globex-room": "globex"}


def pool(executor_id: str, organizations: set[str] | None) -> TurnExecutor:
    """An executor building ``support`` with a model that names it."""

    def build(channel_id: str) -> Agent:
        reply = f"{channel_id} on {executor_id}."
        return Agent(channel_id, provider=MockAIProvider(responses=[reply] * 4))

    return TurnExecutor(
        build, hosts={"support"}, organizations=organizations, executor_id=executor_id
    )


async def main() -> None:
    backend = InMemoryTurnBackend()
    executors = [pool("acme-pool", {"acme"}), pool("platform-pool", None)]
    for executor in executors:
        await backend.serve(executor)
    kit = RoomKit(turn_backend=backend)
    kit.register_channel(WebSocketChannel("chat"))
    kit.register_channel(ChannelDeclaration(channel_id="support", identity=AgentIdentity()))

    print("The executors, and the organizations each serves:")
    for info in await kit.executors():
        served = sorted(info.organizations) if info.organizations else "rooms without one"
        print(f"  {info.executor_id:<14} {served}")

    print("The agent asked in each room:")
    for room_id, organization in ROOMS.items():
        await kit.create_room(room_id=room_id, organization_id=organization)
        await kit.attach_channel(room_id, "chat")
        await kit.attach_channel(room_id, "support", category=ChannelCategory.INTELLIGENCE)
        result = await kit.process_inbound(
            InboundMessage(channel_id="chat", sender_id="ana", content=TextContent(body="Hi.")),
            room_id=room_id,
        )
        answers = [
            event.content.body
            for event in await kit.store.list_events(room_id)
            if event.source.channel_id == "support" and isinstance(event.content, TextContent)
        ]
        presence = await kit.agent_presence(room_id)
        state = "online" if presence[0].online else "offline"
        heard = answers[-1] if answers else f"no answer: {result.error}"
        print(f"  {room_id:<14} org {organization or '-':<7} {state:<8} {heard}")

    await kit.close()
    for executor in executors:
        await executor.close()


if __name__ == "__main__":
    asyncio.run(main())
