"""A caller's response collection never includes another inbound's answer."""

from __future__ import annotations

import asyncio

import pytest

from roomkit import AIChannel, ChannelCategory, RoomKit, WebSocketChannel
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import EventStatus, EventType, HookTrigger
from roomkit.models.event import TextContent
from roomkit.models.hook import HookResult
from roomkit.providers.ai.mock import MockAIProvider


async def _kit(*, streaming: bool) -> RoomKit:
    kit = RoomKit()
    kit.register_channel(WebSocketChannel("user"))
    kit.register_channel(
        AIChannel(
            "assistant",
            MockAIProvider(responses=["first answer", "second answer"], streaming=streaming),
        )
    )
    await kit.create_room(room_id="room")
    await kit.attach_channel("room", "user")
    await kit.attach_channel("room", "assistant", category=ChannelCategory.INTELLIGENCE)
    return kit


def _message(body: str) -> InboundMessage:
    return InboundMessage(channel_id="user", sender_id="person", content=TextContent(body=body))


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("deferred", [False, True])
async def test_response_events_belong_to_their_call(streaming: bool, deferred: bool) -> None:
    kit = await _kit(streaming=streaming)
    try:
        first = await kit.process_inbound(
            _message("first task"), room_id="room", defer_delivery=deferred
        )
        if deferred:
            assert first.delivery is not None
            first = await first.delivery.wait()
        second = await kit.process_inbound(
            _message("second task"), room_id="room", defer_delivery=deferred
        )
        if deferred:
            assert second.delivery is not None
            second = await second.delivery.wait()
        assert [e.content.body for e in first.response_events] == ["first answer"]
        assert [e.content.body for e in second.response_events] == ["second answer"]
        for result in (first, second):
            assert result.event is not None
            for event in result.response_events:
                assert event.id != result.event.id
                assert event.status == EventStatus.DELIVERED
                assert event.type == EventType.MESSAGE
                stored = await kit.store.get_event(event.id)
                assert stored == event
    finally:
        await kit.close()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("blocked", [False, True])
async def test_response_events_obey_broadcast_hooks(streaming: bool, blocked: bool) -> None:
    kit = await _kit(streaming=streaming)

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def gate(event, context):
        if event.source.channel_id != "assistant":
            return HookResult.allow()
        if blocked:
            return HookResult.block("redacted")
        return HookResult.modify(
            event.model_copy(update={"content": TextContent(body="safe answer")})
        )

    try:
        result = await kit.process_inbound(_message("task"), room_id="room")
        assert [e.content.body for e in result.response_events] == (
            [] if blocked else ["safe answer"]
        )
    finally:
        await kit.close()


@pytest.mark.parametrize("streaming", [False, True])
async def test_concurrent_calls_keep_disjoint_responses(streaming: bool) -> None:
    kit = await _kit(streaming=streaming)
    try:
        first, second = await asyncio.gather(
            kit.process_inbound(_message("first task"), room_id="room"),
            kit.process_inbound(_message("second task"), room_id="room"),
        )
        assert [e.content.body for e in first.response_events] == ["first answer"]
        assert [e.content.body for e in second.response_events] == ["second answer"]
        assert {e.id for e in first.response_events}.isdisjoint(
            e.id for e in second.response_events
        )
    finally:
        await kit.close()


@pytest.mark.parametrize("streaming", [False, True])
async def test_a_call_without_an_answer_has_no_other_calls_events(streaming: bool) -> None:
    kit = await _kit(streaming=streaming)
    try:
        answered = await kit.process_inbound(_message("first task"), room_id="room")
        assert answered.response_events
        await kit.detach_channel("room", "assistant")
        unanswered = await kit.process_inbound(_message("no agent"), room_id="room")
        assert not unanswered.response_events
        assert answered.response_events
    finally:
        await kit.close()


@pytest.mark.parametrize("blocked", [False, True])
async def test_detached_response_still_obeys_broadcast_hooks(blocked: bool) -> None:
    started, release = asyncio.Event(), asyncio.Event()

    class DelayedProvider(MockAIProvider):
        async def generate(self, context):
            started.set()
            await release.wait()
            return await super().generate(context)

    kit = await _kit(streaming=False)
    kit.get_channel("assistant")._provider = DelayedProvider(responses=["private answer"])
    inspected = []

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def gate(event, context):
        if event.source.channel_id != "assistant":
            return HookResult.allow()
        inspected.append(event.id)
        if blocked:
            return HookResult.block("redacted")
        return HookResult.modify(
            event.model_copy(update={"content": TextContent(body="safe answer")})
        )

    try:
        result = await kit.process_inbound(_message("task"), room_id="room", defer_delivery=True)
        await asyncio.wait_for(started.wait(), timeout=5)
        await kit.detach_channel("room", "assistant")
        release.set()
        result = await result.delivery.wait()
        assert len(inspected) == 1
        assert [e.content.body for e in result.response_events] == (
            [] if blocked else ["safe answer"]
        )
        events = await kit.store.list_events("room")
        response = next(e for e in events if e.source.channel_id == "assistant")
        assert response.status == (EventStatus.BLOCKED if blocked else EventStatus.DELIVERED)
        if not blocked:
            assert response.content.body == "safe answer"
    finally:
        release.set()
        await kit.close()
