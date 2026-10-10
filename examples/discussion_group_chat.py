"""A group chat of agents and a person: the Discussion strategy.

An on-call engineer and three agents hold one conversation about an incident.
Nobody fixes the order in advance: who speaks next follows from who was
addressed, one agent at a time (RFC §19.7.5).

1. "The checkout API returns 500s since 14:00. What happened?" names nobody,
   so it asks ``everyone``, in the host's order: the investigator opens and
   asks ``@dev`` to confirm the deploy; dev, already queued, answers both in
   one turn and asks ``@sre`` to roll back; sre asks ``@oncall`` for approval,
   and the discussion waits for that person.
2. "Yes, roll it back." names nobody either: it answers the agent that asked,
   sre alone.
3. "@dev please open a fix ticket" asks dev alone.

Each turn reads the room as it is when the turn starts, and every change of
the speak queue reaches ``ON_SPEAK_QUEUE``: a console follows it to show who
speaks, who is next and whether the room waits for a person. The models are
scripted mocks, so the example runs without keys.

Run with:
    uv run python examples/discussion_group_chat.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import (
    Agent,
    Discussion,
    EventType,
    HookExecution,
    HookTrigger,
    InboundMessage,
    RoomKit,
    SpeakQueueEvent,
    TextContent,
    WebSocketChannel,
)
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.providers.ai.base import AIResponse
from roomkit.providers.ai.mock import MockAIProvider

logger = setup_logging("example.discussion_group_chat")

ROOM = "incident-room"


def scripted(*answers: str) -> MockAIProvider:
    """A model that answers its turns with *answers*, in order."""
    return MockAIProvider(ai_responses=[AIResponse(content=a) for a in answers])


async def until_quiet(kit: RoomKit) -> None:
    """Wait until no agent speaks and none can take a turn."""
    quiet = 0
    while quiet < 3:
        await asyncio.sleep(0.02)
        queue = kit.speak_queue(ROOM)
        assert queue is not None
        idle = not queue.queue or queue.waiting or queue.over
        quiet = quiet + 1 if queue.speaking is None and idle else 0


async def main() -> None:
    investigator = Agent(
        "investigator",
        provider=scripted(
            "Error logs show a null pointer in the payment client from 13:55. "
            "@dev can you confirm what was deployed then?"
        ),
        role="Investigator",
        description="Reads logs and metrics first",
    )
    dev = Agent(
        "dev",
        provider=scripted(
            "Confirmed: release 4.2 went out at 13:55 and touched the payment client. "
            "@sre can you roll it back?",
            "Ticket PAY-101 opened for the null check in the payment client.",
        ),
        role="Developer",
        description="Knows the code and the releases",
    )
    sre = Agent(
        "sre",
        provider=scripted(
            "Rolling back production needs approval. @oncall may I roll back release 4.2?",
            "Rolled back to 4.1. The error rate is back to 0.1%.",
        ),
        role="Site reliability engineer",
        description="Runs production changes",
    )

    kit = RoomKit()

    @kit.hook(HookTrigger.ON_SPEAK_QUEUE, execution=HookExecution.ASYNC)
    async def follow(event: SpeakQueueEvent, context: RoomContext) -> None:
        queue = event.queue
        logger.info(
            "  [queue] %s %s | speaking=%s next=%s waiting=%s",
            event.change,
            ",".join(event.channel_ids) or "-",
            queue.speaking or "-",
            ",".join(queue.queue) or "-",
            queue.waiting,
        )

    oncall = WebSocketChannel("oncall")

    async def on_receive(_connection: str, event: RoomEvent) -> None:
        if event.type == EventType.MESSAGE and event.source.channel_id != "oncall":
            logger.info("%s: %s", event.source.channel_id, event.content.body)  # type: ignore[union-attr]

    oncall.register_connection("laptop", on_receive, room_id=ROOM)
    kit.register_channel(oncall)

    await kit.create_room(
        room_id=ROOM,
        orchestration=Discussion(
            agents=[investigator, dev, sre],
            people=["oncall"],
            everyone=["investigator", "dev", "sre"],
        ),
    )
    await kit.attach_channel(ROOM, "oncall")

    for said in (
        "The checkout API returns 500s since 14:00. What happened?",
        "Yes, roll it back.",
        "@dev please open a fix ticket",
    ):
        logger.info("oncall: %s", said)
        await kit.process_inbound(
            InboundMessage(channel_id="oncall", sender_id="oncall", content=TextContent(body=said))
        )
        await until_quiet(kit)

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
