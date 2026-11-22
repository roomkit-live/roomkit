"""A custom AI provider ends each generation in the contract's one vocabulary.

Every vendor reports why a response stopped in its own words. A provider maps
them onto FinishReason, a closed set, and the AI channel reads only that set:
it never learns a vendor's word. Here a provider for a fictional vendor maps
``parse_failure`` to ``malformed_call`` and ``complete`` to ``stop``; the
channel tells the model its call did not run, and the model answers on its
next round. Shows:
- FinishReason: stop, tool_calls, length, content_filter, refusal,
  malformed_call, unexpected_call, error
- A vendor word left unmapped: AIResponse refuses it

Run with:
    uv run python examples/ai_finish_reason.py
"""

from __future__ import annotations

import asyncio

from pydantic import ValidationError
from shared import setup_logging

from roomkit import (
    AIChannel,
    ChannelCategory,
    EventType,
    InboundMessage,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.providers.ai import AIContext, AIProvider, AIResponse, AITool, FinishReason

logger = setup_logging("ai_finish_reason")

LOOKUP = AITool(name="lookup", description="Look an order up", parameters={})

# The fictional vendor's stop words, and the contract's word for each.
ACME_STOP_WORDS = {
    "complete": FinishReason.STOP,
    "calls": FinishReason.TOOL_CALLS,
    "budget_exhausted": FinishReason.LENGTH,
    "parse_failure": FinishReason.MALFORMED_CALL,
}


class AcmeProvider(AIProvider):
    """A provider whose vendor answers in its own words: a call it could not
    parse first, then a whole answer."""

    def __init__(self) -> None:
        self._answers = [("", "parse_failure"), ("Order 42 ships tomorrow.", "complete")]
        self.told: list[str] = []

    @property
    def model_name(self) -> str:
        return "acme-1"

    async def generate(self, context: AIContext) -> AIResponse:
        self.told.append(str(context.messages[-1].content))
        text, word = self._answers.pop(0)
        # The adapter's one job here: the vendor's word in the contract's.
        return AIResponse(content=text, finish_reason=ACME_STOP_WORDS.get(word))


async def main() -> None:
    try:
        AIResponse(content="", finish_reason="parse_failure")
    except ValidationError:
        logger.info("a vendor's own word is refused: the provider maps it first")

    provider = AcmeProvider()
    kit = RoomKit()
    kit.register_channel(WebSocketChannel("member"))
    kit.register_channel(
        AIChannel("agent", provider=provider, tools=[LOOKUP], tool_handler=lambda *_: "ok")
    )
    await kit.create_room(room_id="room")
    await kit.attach_channel("room", "member")
    await kit.attach_channel("room", "agent", category=ChannelCategory.INTELLIGENCE)
    await kit.process_inbound(
        InboundMessage(
            channel_id="member", sender_id="u1", content=TextContent(body="Where is order 42?")
        )
    )

    logger.info("the model was told, after its first round: %s", provider.told[-1])
    for event in await kit.store.list_events("room"):
        if event.type == EventType.MESSAGE and event.source.channel_id == "agent":
            logger.info("agent: %s", event.content.body)
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
