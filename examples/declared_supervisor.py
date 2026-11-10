"""A supervisor and its worker the kit holds by their declarations alone.

The kit that owns the rooms registers each agent as a ``ChannelDeclaration``:
it never holds their providers, their tools or their memory. A
``TurnExecutor`` builds them where their turns run. The Supervisor is given
the agents by id (RFC §10.6.2):

1. The person asks the supervisor for the weather.
2. The supervisor's turn, on the executor, is given ``delegate_to_meteo``:
   the kit keeps that tool beside the supervisor's declaration and lists it
   in each of its turns (RFC §10.6.5).
3. The supervisor calls it; the kit runs the delegation, and the worker's
   turn in its child room runs on the executor too.
4. The worker's answer comes back to the supervisor, which says it.

Here the executor shares the kit's process and its in-memory backend; with a
distributed backend it runs in another process, and the kit's code does not
change. The models are scripted mocks, so the example runs without keys.

Run with:
    uv run python examples/declared_supervisor.py
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
    InboundMessage,
    InMemoryStore,
    RoomKit,
    SMSChannel,
    Supervisor,
    TextContent,
)
from roomkit.providers.ai import AIResponse, AIToolCall, MockAIProvider
from roomkit.providers.sms import MockSMSProvider
from roomkit.turns import InMemoryTurnBackend, TurnExecutor

logger = setup_logging("example.declared_supervisor")

PHONE = "+15145550100"
ROOM = "front-desk"


# -- The host's side: where the agents' live objects are --------------------


def build_agent(channel_id: str) -> Agent | None:
    """The host's factory: the executor builds the agents it hosts."""
    if channel_id == "desk":
        delegate = AIToolCall(
            id="d1", name="delegate_to_meteo", arguments={"task": "Tomorrow's weather in Quebec."}
        )
        script = [
            AIResponse(content="", finish_reason="tool_calls", tool_calls=[delegate]),
            AIResponse(content="Meteo says: tomorrow in Quebec, 8 C and showers."),
        ]
        return Agent("desk", provider=MockAIProvider(ai_responses=script), role="Front desk")
    if channel_id == "meteo":
        provider = MockAIProvider(responses=["Tomorrow in Quebec: 8 C and showers."])
        return Agent("meteo", provider=provider, role="Weather agent")
    return None


# -- The owner's side: the kit knows the agents by their declarations --------


DESK = ChannelDeclaration(channel_id="desk", identity=AgentIdentity(role="Front desk"))
METEO = ChannelDeclaration(
    channel_id="meteo",
    identity=AgentIdentity(description="Forecasts the weather", purpose="the weather"),
)


async def main() -> None:
    store = InMemoryStore()
    backend = InMemoryTurnBackend()
    executor = TurnExecutor(
        build_agent,
        hosts=lambda channel_id: channel_id in ("desk", "meteo"),
    )
    await backend.serve(executor)

    kit = RoomKit(store=store, turn_backend=backend)
    kit.register_channel(SMSChannel("sms", provider=MockSMSProvider()))
    kit.register_channel(DESK)  # no Agent object on the kit
    kit.register_channel(METEO)

    await kit.create_room(
        room_id=ROOM, orchestration=Supervisor(supervisor="desk", workers=["meteo"])
    )
    await kit.attach_channel(ROOM, "sms", metadata={"phone_number": PHONE})

    logger.info("[person] What is the weather tomorrow in Quebec?")
    await kit.process_inbound(
        InboundMessage(
            channel_id="sms",
            sender_id=PHONE,
            content=TextContent(body="What is the weather tomorrow in Quebec?"),
        ),
        room_id=ROOM,
    )
    await asyncio.sleep(0.5)

    for event in await store.list_events(ROOM):
        if event.source.channel_id == "desk" and isinstance(event.content, TextContent):
            logger.info("  desk: %s", event.content.body)
    held = {cid: type(kit.channels[cid]).__name__ for cid in ("desk", "meteo")}
    logger.info("the kit holds %s", held)
    await kit.close()
    await executor.close()


if __name__ == "__main__":
    asyncio.run(main())
