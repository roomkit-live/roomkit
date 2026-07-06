"""Streamed AI segments must pass through BEFORE_BROADCAST sync hooks.

The streaming path used to persist and re-broadcast segments without ever
running BEFORE_BROADCAST (only AFTER_BROADCAST fired). A pre-broadcast guard
on AI output — e.g. PII de-anonymisation — was therefore bypassed in
streaming: the stored row kept the raw text. These tests pin that a hook's
modification lands on the persisted segment and that a block drops it.
"""

from __future__ import annotations

from roomkit.channels.ai import AIChannel
from roomkit.core.framework import RoomKit
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, EventType, HookTrigger
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.hook import HookResult
from roomkit.providers.ai.base import AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel


async def _wire(kit: RoomKit, content: str) -> None:
    provider = MockAIProvider(
        streaming=True,
        ai_responses=[AIResponse(content=content, finish_reason="stop")],
    )
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(AIChannel("ai1", provider=provider))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)


def _ai_messages(events: list[RoomEvent]) -> list[RoomEvent]:
    return [e for e in events if e.type == EventType.MESSAGE and e.source.channel_id == "ai1"]


async def test_streamed_segment_applies_before_broadcast_modification() -> None:
    kit = RoomKit()
    await _wire(kit, content="Hi [PERSON_1], welcome")

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="deanon")
    async def deanon(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if not isinstance(event.content, TextContent) or "[PERSON_1]" not in event.content.body:
            return HookResult.allow()
        restored = event.content.body.replace("[PERSON_1]", "Alice")
        return HookResult.modify(event.model_copy(update={"content": TextContent(body=restored)}))

    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )

    ai_msgs = _ai_messages(await kit.store.list_events("r1"))
    assert ai_msgs, "streamed AI segment was not persisted"
    assert ai_msgs[-1].content.body == "Hi Alice, welcome"
    assert "[PERSON_1]" not in ai_msgs[-1].content.body


async def test_streamed_segment_dropped_when_before_broadcast_blocks() -> None:
    kit = RoomKit()
    await _wire(kit, content="[SECRET] leaked")

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="blocker")
    async def blocker(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if event.source.channel_id == "ai1":
            return HookResult.block("withheld")
        return HookResult.allow()

    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )

    # The blocked segment never enters the timeline as a delivered message.
    assert _ai_messages(await kit.store.list_events("r1")) == []
