"""Deferred delivery — return the committed event, let the agent's turn follow.

``process_inbound(..., defer_delivery=True)`` comes back at the commit point
(RFC §10.1 step 18 detached completion): the caller holds the committed event
— exactly what an HTTP route needs to build its 200 response — while the
delivery set and the AI reply run on in the room's delivery lane. A hook
refusal is still decided under the room lock, so a refused message still
refuses the call synchronously. ``result.delivery`` is the handle on the
in-flight turn: ``wait()`` resolves once everything ran (streamed responses
included) and backfills ``delivery_results``, ``response_events`` and ``error``.
The response collection stays attached to this request even after another turn.

Uses the mock AI provider, so it runs without any API key:

    uv run python examples/deferred_inbound.py
"""

from __future__ import annotations

import asyncio
import logging

from roomkit import (
    AIChannel,
    Channel,
    ChannelBinding,
    ChannelCategory,
    ChannelOutput,
    ChannelType,
    EventSource,
    InboundMessage,
    RoomContext,
    RoomEvent,
    RoomKit,
    TextContent,
)
from roomkit.providers.ai.base import AIContext, AIResponse
from roomkit.providers.ai.mock import MockAIProvider

logger = logging.getLogger(__name__)


class RestChannel(Channel):
    """A minimal transport standing in for an HTTP API surface."""

    channel_type = ChannelType.WEBHOOK

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            source=EventSource(channel_id=self.channel_id, channel_type=self.channel_type),
            content=message.content,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        logger.info("Delivered to %s: %s", self.channel_id, event.content.body)
        return ChannelOutput.empty()


class SlowAIProvider(MockAIProvider):
    """Mock provider with a visible think time, so the deferral shows."""

    async def generate(self, context: AIContext) -> AIResponse:
        await asyncio.sleep(1.0)
        return await super().generate(context)


async def main() -> None:
    kit = RoomKit()
    kit.register_channel(RestChannel("rest"))
    kit.register_channel(
        AIChannel(
            "assistant",
            provider=SlowAIProvider(responses=["Here is the first answer.", "A second answer."]),
        )
    )

    await kit.create_room(room_id="support")
    await kit.attach_channel("support", "rest")
    await kit.attach_channel("support", "assistant", category=ChannelCategory.INTELLIGENCE)

    # The shape of a REST route: ONE process_inbound, deferred. The result is
    # back immediately with the committed event — the body of the 200 — while
    # the agent thinks in the background.
    result = await kit.process_inbound(
        InboundMessage(
            channel_id="rest",
            sender_id="customer",
            content=TextContent(body="What is the answer?"),
            addressed_to=["assistant"],
        ),
        room_id="support",
        defer_delivery=True,
    )
    assert result.event is not None
    assert result.delivery is not None
    logger.info("Committed event %s (agent turn in flight)", result.event.id)

    # Later — a tracker, a test, a metrics blip — awaits the rest of the turn.
    await result.delivery.wait()
    # Another request may finish before the tracker reads its answer.
    await kit.process_inbound(
        InboundMessage(
            channel_id="rest",
            sender_id="customer",
            content=TextContent(body="A different question."),
            addressed_to=["assistant"],
        ),
        room_id="support",
    )
    replies = [
        event.content.body
        for event in result.response_events
        if isinstance(event.content, TextContent)
    ]
    assert replies == ["Here is the first answer."]
    logger.info("First request's response after both turns completed: %s", replies)

    await kit.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
