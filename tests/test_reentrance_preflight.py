"""A named call spends its budget before asking its channel (RFC §10.1 step 1)."""

from __future__ import annotations

import asyncio

import pytest

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit, RoomNotFoundError
from roomkit.models import (
    ChannelBinding,
    ChannelOutput,
    InboundMessage,
    InboundResult,
    RoomContext,
    RoomEvent,
    TextContent,
)
from roomkit.telemetry import MockTelemetryProvider, SpanKind, TelemetryConfig
from tests.test_framework import SimpleChannel
from tests.test_reentrance import BUDGET, _body, _message, _messages, _room


class AddressCalls(SimpleChannel):
    def __init__(self, channel_id: str = "late") -> None:
        super().__init__(channel_id)
        self.calls: list[str] = []

    def normalize_address(self, address: str) -> str:
        self.calls.append("normalize")
        return address.lower()

    def conversation_address(self, message: InboundMessage) -> str:
        self.calls.append("conversation")
        return message.sender_id

    def reply_metadata(self, message: InboundMessage) -> dict[str, str]:
        self.calls.append("reply")
        return {}


@pytest.mark.parametrize("work", ["delivery", "locked", "off-lock", "observer"])
async def test_a_named_call_past_its_budget_asks_no_address_callback(work: str) -> None:
    """Passes and hook calls refuse before normalization, without a second charge."""
    telemetry = MockTelemetryProvider()
    kit = RoomKit(
        max_chain_depth=1,
        telemetry=TelemetryConfig(provider=telemetry, record_sensitive_data=True),
    )
    late = AddressCalls()
    finished = asyncio.Event()
    answers: list[InboundResult] = []

    async def spend() -> None:
        for _ in range(BUDGET - 1):
            answers.append(await kit.process_inbound(_message("again"), room_id="r1"))
        # A failed scope check must leave the last unit and ask no callback.
        with pytest.raises(RoomNotFoundError):
            await kit.process_inbound(
                InboundMessage(
                    channel_id="late", sender_id="RAW", content=TextContent(body="scope")
                ),
                room_id="r1",
                organization_id="other",
            )
        answers.append(await kit.process_inbound(_message("again"), room_id="r1"))
        answers.append(
            await kit.process_inbound(
                InboundMessage(channel_id="late", sender_id="RAW", content=TextContent(body="x")),
                room_id="r1",
            )
        )
        finished.set()

    class Spender(SimpleChannel):
        async def deliver(
            self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
        ) -> ChannelOutput:
            if _body(event) == "hello":
                await spend()
            return ChannelOutput.empty()

    await _room(kit, *([Spender("spender")] if work == "delivery" else []))
    kit.register_channel(late)
    before = list(late.calls)
    if work != "delivery":
        trigger = (
            HookTrigger.AFTER_BROADCAST if work == "observer" else HookTrigger.BEFORE_BROADCAST
        )

        @kit.hook(
            trigger,
            execution=HookExecution.ASYNC if work == "observer" else HookExecution.SYNC,
            needs_lock=work != "off-lock",
        )
        async def spend_from_hook(event: RoomEvent, context: RoomContext) -> HookResult:
            if _body(event) == "hello":
                await spend()
            return HookResult.allow()

    try:
        await asyncio.wait_for(kit.process_inbound(_message("hello"), room_id="r1"), 5)
        await asyncio.wait_for(finished.wait(), 5)
        assert len(answers) == BUDGET + 1
        assert all(not answer.blocked and answer.event is not None for answer in answers[:-1])
        refused = answers[-1]
        assert (refused.blocked, refused.reason, refused.event) == (True, "reentry_loop_cap", None)
        assert late.calls == before
        assert await kit.store.get_binding("r1", "late") is None
        assert sorted(_body(event) for event in await _messages(kit)) == ["again"] * BUDGET + [
            "hello"
        ]
        [span] = [
            span
            for span in telemetry.get_spans(SpanKind.INBOUND_PIPELINE)
            if span.name == "framework.inbound" and span.channel_id == "late"
        ]
        assert span.room_id == "r1"
        assert span.attributes["blocked"] is True
        assert span.attributes["sender_id"] == "RAW"
    finally:
        await kit.close()


@pytest.mark.parametrize("work", ["delivery", "observer"])
async def test_two_named_calls_share_the_last_unit_before_address_callbacks(work: str) -> None:
    """Exactly one simultaneous call reaches its channel on the last unit."""
    kit = RoomKit(max_chain_depth=1)
    channels = [AddressCalls("one"), AddressCalls("two")]
    answers: list[InboundResult] = []
    finished = asyncio.Event()

    async def spend() -> None:
        for _ in range(BUDGET - 1):
            answer = await kit.process_inbound(_message("again"), room_id="r1")
            assert not answer.blocked
        barrier = asyncio.Barrier(2)

        async def last(channel_id: str) -> InboundResult:
            await barrier.wait()
            return await kit.process_inbound(
                InboundMessage(
                    channel_id=channel_id, sender_id="RAW", content=TextContent(body="last")
                ),
                room_id="r1",
            )

        answers.extend(await asyncio.gather(*(last(channel.channel_id) for channel in channels)))
        finished.set()

    class Spender(SimpleChannel):
        async def deliver(
            self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
        ) -> ChannelOutput:
            if _body(event) == "hello":
                await spend()
            return ChannelOutput.empty()

    await _room(kit, *channels, *([Spender("spender")] if work == "delivery" else []))
    before = [list(channel.calls) for channel in channels]
    if work == "observer":

        @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC)
        async def spend_from_hook(event: RoomEvent, context: RoomContext) -> None:
            if _body(event) == "hello":
                await spend()

    try:
        await asyncio.wait_for(kit.process_inbound(_message("hello"), room_id="r1"), 5)
        await asyncio.wait_for(finished.wait(), 5)
        assert len(answers) == 2
        assert sorted(answer.blocked for answer in answers) == [False, True]
        for index, answer in enumerate(answers):
            if answer.blocked:
                assert (answer.reason, answer.event) == ("reentry_loop_cap", None)
                assert channels[index].calls == before[index]
            else:
                assert answer.event is not None
                assert channels[index].calls != before[index]
        texts = [_body(event) for event in await _messages(kit)]
        assert texts.count("again") == 9
        assert texts.count("last") == 1
        assert len(texts) == 11
    finally:
        await kit.close()
