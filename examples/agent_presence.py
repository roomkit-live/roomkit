"""Which executors serve a room's agents, and their presence as it changes
(RFC §10.6.7).

The room holds two agents by their declaration alone: an executor serves
``coder``; nothing serves ``reviewer``.

1. ``kit.executors()`` lists the executor and what it declares.
2. ``kit.agent_presence(room_id)`` says which agent can answer, and on which
   executor.
3. A subscriber to the room hears the agents' presence as it changes, as a
   UI showing people's presence would: the executor closed takes ``coder``
   offline, and another one starting brings it back.
4. A discussion of the same agents skips the offline ``reviewer``'s turn,
   announced as ``turn_skipped`` (RFC §19.7.5 rule 17), and ``coder`` answers.

Across processes (``RedisTurnBackend``) the same reads list the executors
of every process, and one room owner publishes each change.

Run with:
    uv run python examples/agent_presence.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import (
    Agent,
    ChannelCategory,
    ChannelDeclaration,
    Discussion,
    HookExecution,
    HookTrigger,
    InboundMessage,
    RoomKit,
    SpeakQueueEvent,
    TextContent,
    WebSocketChannel,
)
from roomkit.providers.ai import MockAIProvider
from roomkit.realtime import EphemeralEvent, EphemeralEventType
from roomkit.turns import InMemoryTurnBackend, TurnExecutor

logger = setup_logging("example.agent_presence")

ROOM = "review"
TALK = "talk"
PRESENCE = {EphemeralEventType.PRESENCE_ONLINE, EphemeralEventType.PRESENCE_OFFLINE}


def coder_executor(executor_id: str) -> TurnExecutor:
    """An executor that builds ``coder`` and declares it alone."""
    return TurnExecutor(
        lambda channel_id: Agent(channel_id, provider=MockAIProvider(responses=["Done."])),
        hosts={"coder"},
        executor_id=executor_id,
    )


async def show_presence(kit: RoomKit) -> None:
    for presence in await kit.agent_presence(ROOM):
        where = ", ".join(presence.executors) or "no executor"
        state = "online" if presence.online else "offline"
        print(f"  {presence.channel_id:<9} {state:<8} ({where})")


async def main() -> None:
    backend = InMemoryTurnBackend()
    first = coder_executor("executor-a")
    await backend.serve(first)

    kit = RoomKit(turn_backend=backend)
    kit.register_channel(WebSocketChannel("web"))
    for agent in ("coder", "reviewer"):
        kit.register_channel(ChannelDeclaration(channel_id=agent))
    await kit.create_room(room_id=ROOM)
    await kit.attach_channel(ROOM, "web")

    async def heard(event: EphemeralEvent) -> None:
        if event.type in PRESENCE:
            print(f"  heard: {event.user_id} is {event.type.value.removeprefix('presence_')}")

    await kit.realtime.subscribe_to_room(ROOM, heard)
    second = coder_executor("executor-b")
    try:
        logger.info("Agents attached: each one's presence is published in the room")
        for agent in ("coder", "reviewer"):
            await kit.attach_channel(ROOM, agent, category=ChannelCategory.INTELLIGENCE)
        await asyncio.sleep(0)  # the realtime backend hands its events on

        logger.info("1. The executors the turn backend carries turns to")
        for info in await kit.executors():
            print(
                f"  {info.executor_id}: declares {sorted(info.hosts or [])}, online {info.online}"
            )

        logger.info("2. The room's agents, and where each can answer")
        await show_presence(kit)

        logger.info("3. executor-a closes, then executor-b starts")
        await first.close()
        await backend.serve(second)
        await asyncio.sleep(0)  # the realtime backend hands its events on
        await show_presence(kit)

        logger.info("4. A discussion of both agents: the offline reviewer is skipped")
        await discuss(kit)
    finally:
        await kit.close()
        await second.close()


async def discuss(kit: RoomKit) -> None:
    @kit.hook(HookTrigger.ON_SPEAK_QUEUE, execution=HookExecution.ASYNC)
    async def follow(event: SpeakQueueEvent, ctx: object) -> None:
        print(f"  queue: {event.change.value} {list(event.channel_ids)}")

    await kit.create_room(room_id=TALK, orchestration=Discussion(agents=["coder", "reviewer"]))
    await kit.attach_channel(TALK, "web")
    said = InboundMessage(channel_id="web", sender_id="dana", content=TextContent(body="Status?"))
    await kit.process_inbound(said, room_id=TALK)
    for _ in range(200):  # the turns come after the message, in the discussion's driver
        if any(e.source.channel_id == "coder" for e in await kit.store.list_events(TALK)):
            break
        await asyncio.sleep(0.01)
    for event in await kit.store.list_events(TALK):
        if isinstance(event.content, TextContent):
            print(f"  {event.source.channel_id}: {event.content.body}")


if __name__ == "__main__":
    asyncio.run(main())
