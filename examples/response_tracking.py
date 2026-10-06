"""Response tracking — every answer names the request it answers (RFC §8.5).

Two requests reach the assistant: a question, then an application instruction
(as a background task's result would arrive). Each turn is told what it answers
(``BEFORE_AI_GENERATION`` sees ``trigger``), every event of its answer carries
``responds_to``, and the answers to a request can be listed with
``EventFilter(responds_to=...)``.

Run with:
    uv run python examples/response_tracking.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import (
    ChannelCategory,
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.models.store_filter import EventFilter
from roomkit.providers.ai.mock import MockAIProvider

ROOM = "tracking-room"
logger = setup_logging("example.response_tracking")


def _text(event: RoomEvent) -> str:
    return getattr(event.content, "body", f"<{event.type}>")


async def main() -> None:
    kit = RoomKit()
    kit.register_channel(WebSocketChannel("ws-user"))
    kit.register_channel(
        AIChannel(
            "assistant",
            provider=MockAIProvider(
                responses=["Il est midi.", "La météo est arrivée : 8 degrés demain."]
            ),
        )
    )
    await kit.create_room(room_id=ROOM)
    await kit.attach_channel(ROOM, "ws-user")
    await kit.attach_channel(ROOM, "assistant", category=ChannelCategory.INTELLIGENCE)

    # Each turn is told what it answers: a host can brief the model with it.
    @kit.hook(HookTrigger.BEFORE_AI_GENERATION)
    async def what_this_turn_answers(event: Any, ctx: Any) -> HookResult:
        trigger = event.trigger
        logger.info(
            "turn of %s answers %s (%s): %r",
            event.channel_id,
            trigger.id[:8],
            trigger.type,
            _text(trigger),
        )
        return HookResult.allow()

    # A person's question.
    asked = await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="sylvain",
            content=TextContent(body="Quelle heure est-il ?"),
        ),
        room_id=ROOM,
    )
    assert asked.event is not None

    # An application instruction, as a background task's result is handed back.
    await kit.deliver(
        ROOM,
        "Share the weather result: 8 degrees tomorrow.",
        instruction=True,
        addressed_to=["assistant"],
    )

    logger.info("Timeline:")
    for event in await kit.store.list_events(ROOM, offset=0, limit=20):
        answers = f" → answers {event.responds_to[:8]}" if event.responds_to else ""
        logger.info("  #%d %s: %r%s", event.index, event.source.channel_id, _text(event), answers)

    answers = await kit.store.list_events(
        ROOM, offset=0, limit=10, event_filter=EventFilter(responds_to=asked.event.id)
    )
    logger.info("Answers to %s: %s", asked.event.id[:8], [_text(e) for e in answers])
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
