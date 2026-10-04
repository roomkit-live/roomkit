"""How a turn ends, the same on every door that runs one (RMK-497, RFC §6.4,
§12.2 step 13s, §23.3).

A turn cancelled from outside records ``cancelled`` as one its reader
stopped does; a delegated turn's failure is reported, scoped and logged as a
room turn's; the error a caller reads, a kit that closes and a reasoning
backend's failure follow one rule whichever door the turn came through.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.agent import Agent
from roomkit.channels.base import Channel
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType, EventType
from roomkit.models.event import RoomEvent, TextContent
from roomkit.providers.ai.base import (
    AIContext,
    AIResponse,
    AITool,
    AIToolCall,
    ProviderError,
    StreamEvent,
    StreamTextDelta,
)
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel


class _Slow(MockAIProvider):
    """Streams the start of an answer, says so, then waits for the rest."""

    def __init__(self) -> None:
        super().__init__(streaming=True)
        self.started = asyncio.Event()

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        for word in ("The ", "partial "):
            yield StreamTextDelta(text=word)
        self.started.set()
        await asyncio.sleep(5)
        yield StreamTextDelta(text="rest.")


async def _kit() -> tuple[RoomKit, _Slow]:
    provider = _Slow()
    kit = RoomKit()
    kit.register_channel(SimpleChannel("tx"))
    kit.register_channel(Agent("worker", provider=provider))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "tx")
    return kit, provider


async def _worker_rows(kit: RoomKit, room_id: str) -> list[Any]:
    return [
        event.metadata.get("loop_end_reason")
        for event in await kit.store.list_events(room_id)
        if event.type == EventType.MESSAGE and event.source.channel_id == "worker"
    ]


async def _room_cancelled(kit: RoomKit, provider: _Slow) -> tuple[str, Any]:
    await kit.attach_channel("r", "worker", category=ChannelCategory.INTELLIGENCE)
    message = InboundMessage(channel_id="tx", sender_id="u", content=TextContent(body="Go."))
    result = await kit.process_inbound(message, defer_delivery=True)
    await provider.started.wait()
    assert result.delivery is not None
    final = await result.delivery.cancel()
    return "r", dict(final.response_metadata).get("turns")


async def _delegate_cancelled(kit: RoomKit, provider: _Slow) -> tuple[str, Any]:
    rooms: list[str] = []

    @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
    async def done(event: Any, context: Any) -> None:
        rooms.append(event.metadata["child_room_id"])

    delegation = asyncio.ensure_future(kit.delegate("r", "worker", "Go.", wait=True))
    await provider.started.wait()
    delegation.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await delegation
    for _ in range(100):
        if rooms:
            break
        await asyncio.sleep(0.01)
    return rooms[0], None


@pytest.mark.parametrize("door", [_room_cancelled, _delegate_cancelled], ids=["room", "delegate"])
async def test_a_turn_cancelled_from_outside_records_cancelled(door: Any) -> None:
    kit, provider = await _kit()

    room_id, turns = await door(kit, provider)
    rows = await _worker_rows(kit, room_id)
    await kit.close()

    assert rows == ["cancelled"]
    if door is _room_cancelled:
        assert turns == {"worker": {"loop_end_reason": "cancelled"}}


LOOKUP = AITool(name="lookup", description="Look up", parameters={"type": "object"})


class _FailsAfterARound(MockAIProvider):
    """Calls a tool, then fails: a streamed turn interrupted after a round."""

    async def generate(self, context: AIContext) -> AIResponse:
        self.calls.append(context)
        if len(self.calls) % 2 == 1:
            call = AIToolCall(id=f"c{len(self.calls)}", name="lookup", arguments={})
            return AIResponse(content="Checking.", finish_reason="tool_calls", tool_calls=[call])
        raise ProviderError("upstream 400", provider="mock", status_code=400)


async def _found(name: str, arguments: dict[str, Any]) -> str:
    return "found"


def _failing_worker() -> Agent:
    provider = _FailsAfterARound(streaming=True)
    return Agent("worker", provider=provider, tools=[LOOKUP], tool_handler=_found)


@pytest.mark.parametrize("shared", [False, True], ids=["trace", "shared"])
async def test_a_delegated_turn_s_failure_is_reported_in_the_turn_s_scope(shared: bool) -> None:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(_failing_worker())
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    errors: list[Any] = []

    @kit.hook(HookTrigger.ON_ERROR, execution=HookExecution.ASYNC)
    async def on_error(event: Any, context: Any) -> None:
        errors.append(event)

    await kit.delegate(
        "r", "worker", "Find it.", wait=True, share_channels=["sms"] if shared else None
    )
    for _ in range(100):
        if errors:
            break
        await asyncio.sleep(0.01)
    await kit.close()

    [error] = errors
    assert (error.metadata["error_category"], error.chain_depth) == ("streaming", 1)
    assert error.correlation_id is not None


class _BufferedFails(Channel):
    """An intelligence channel whose buffered reply carries an error."""

    channel_type = ChannelType.AI
    category = ChannelCategory.INTELLIGENCE

    async def handle_inbound(self, message: Any, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput(responded=True, error=RuntimeError("buffered boom"))


@pytest.mark.parametrize("door", ["inbound", "regenerate"])
async def test_the_buffered_failure_is_the_error_a_caller_reads_on_every_door(door: str) -> None:
    """A buffered agent and a streamed one fail together: the caller reads
    the buffered failure, the cascade's first (RFC §10.1 step 18)."""
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(_BufferedFails("buffered"))
    kit.register_channel(_failing_worker())
    await kit.create_room(room_id="r")
    for channel_id in ("sms", "buffered", "worker"):
        category = ChannelCategory.TRANSPORT if channel_id == "sms" else None
        await kit.attach_channel("r", channel_id, category=category)
    message = InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="Go."))
    result = await kit.process_inbound(message)
    if door == "regenerate":
        result = await kit.regenerate_response("r")
    await kit.close()

    assert repr(result.error) == "RuntimeError('buffered boom')"


class _NoAnswerFails(_BufferedFails):
    """A buffered reply that did not respond and carries why (a runner's shape)."""

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput(responded=False, error=RuntimeError("producer failed"))


@pytest.mark.parametrize("shared", [False, True], ids=["trace", "shared"])
async def test_a_delegated_reply_that_did_not_respond_fails_the_task_with_its_error(
    shared: bool,
) -> None:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(_NoAnswerFails("worker"))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")

    task = await kit.delegate(
        "r", "worker", "Go.", wait=True, share_channels=["sms"] if shared else None
    )
    await kit.close()

    assert task.result is not None
    assert (str(task.result.status), task.result.error) == ("failed", "producer failed")
