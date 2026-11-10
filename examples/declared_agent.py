"""The room's owner holds an agent by its declaration alone (RFC §10.6.2).

An agent holds live objects: its AI provider, its MCP clients, its memory.
The kit that owns the rooms does not need them. It needs to know what the
agent is, to bind it, route to it, greet with it and delegate to it. So the
kit registers a ``ChannelDeclaration``, and a ``TurnExecutor`` builds the
agent where its turns run:

1. The kit binds the agent from its declaration (its capabilities, its
   category) and greets with its identity's greeting.
2. A person's message reaches the agent: the turn crosses the turn backend to
   the executor, which builds the agent and runs it. The answer comes back
   to the kit, which stores and delivers it.
3. The host starts a background task for the agent (``kit.start_task``):
   the work runs on the executor too, and its result is relayed to the agent,
   which says it.

Here the executor shares the kit's process and its in-memory backend; with a
distributed backend it runs in another process, and the kit's code does not
change. The model is a scripted mock, so the example runs without keys.

Run with:
    uv run python examples/declared_agent.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import (
    Agent,
    AgentIdentity,
    ChannelDeclaration,
    EventType,
    HookResult,
    HookTrigger,
    InboundMessage,
    InMemoryStore,
    RoomKit,
    SMSChannel,
    TextContent,
)
from roomkit.providers.ai import MockAIProvider
from roomkit.providers.sms import MockSMSProvider
from roomkit.turns import InMemoryTurnBackend, TurnExecutor

logger = setup_logging("example.declared_agent")

PHONE = "+15145550100"
ROOM = "weather"


# -- The host's side: where the agent's live objects are --------------------


def build_agent(channel_id: str) -> Agent | None:
    """The host's factory: the executor builds the agents it hosts."""
    if channel_id != "meteo":
        return None
    provider = MockAIProvider(
        responses=[
            "Tomorrow in Quebec: 8 C and showers.",
            "Montreal tomorrow: sunny, 21 C.",
            "Good news for Montreal: sunny tomorrow, 21 C.",
        ]
    )
    return Agent(channel_id, provider=provider, name="Meteo", role="Weather agent")


# -- The owner's side: the kit knows the agent by its declaration ------------


METEO = ChannelDeclaration(
    channel_id="meteo",
    identity=AgentIdentity(name="Meteo", role="Weather agent", greeting="Hi, ask me the weather."),
)


async def main() -> None:
    store = InMemoryStore()
    backend = InMemoryTurnBackend()
    executor = TurnExecutor(
        build_agent,
        hosts=lambda channel_id: channel_id == "meteo",
    )
    await backend.serve(executor)

    kit = RoomKit(store=store, turn_backend=backend)
    kit.register_channel(SMSChannel("sms", provider=MockSMSProvider()))
    kit.register_channel(METEO)  # no Agent object on the kit

    @kit.hook(HookTrigger.BEFORE_BROADCAST, event_types={EventType.MESSAGE})
    async def show(event, ctx):
        if event.source.channel_id == "meteo":
            logger.info("  meteo: %s", event.content.body)
        return HookResult.allow()

    await kit.create_room(room_id=ROOM)
    await kit.attach_channel(ROOM, "sms", metadata={"phone_number": PHONE})
    binding = await kit.attach_channel(ROOM, "meteo")
    logger.info("bound meteo as %s, from its declaration", binding.category.value)

    # 1. The greeting is the declaration's.
    await kit.send_greeting(ROOM, agent_id="meteo")
    greeting = (await store.list_events(ROOM))[-1]
    logger.info("  %s greets: %s", greeting.source.channel_id, greeting.content.body)

    # 2. A turn runs on the executor that builds the agent.
    logger.info("[person] What is the weather tomorrow in Quebec?")
    await kit.process_inbound(
        InboundMessage(
            channel_id="sms",
            sender_id=PHONE,
            content=TextContent(body="What is the weather tomorrow in Quebec?"),
        ),
        room_id=ROOM,
    )

    # 3. A background task's work runs there too, its result relayed.
    task = await kit.start_task(ROOM, "meteo", "The weather tomorrow in Montreal.")
    result = await task.wait(timeout=5)
    logger.info("background task %s: %s", result.status, result.output)
    await asyncio.sleep(0.5)  # the relayed result's turn

    logger.info("kit.channels['meteo'] is a %s", type(kit.channels["meteo"]).__name__)
    await kit.close()
    await executor.close()


if __name__ == "__main__":
    asyncio.run(main())
