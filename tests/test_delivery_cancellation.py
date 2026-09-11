"""A turn owns its work until cancellation cleanup has finished."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from roomkit import Agent, ChannelCategory, InboundMessage, RoomKit, TextContent
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel


class PausedTool:
    def __init__(self, *, slow_cleanup: bool = False, suppress: bool = False) -> None:
        self.started = asyncio.Event()
        self.unwinding = asyncio.Event()
        self.finished = asyncio.Event()
        self.release = asyncio.Event()
        self.cleanup = asyncio.Event()
        if not slow_cleanup:
            self.cleanup.set()
        self.suppress = suppress
        self.calls: list[str] = []
        self.effects: list[str] = []

    async def __call__(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append(name)
        self.started.set()
        try:
            await self.release.wait()
            self.effects.append(name)
        except asyncio.CancelledError:
            if not self.suppress:
                raise
        finally:
            self.unwinding.set()
            await self.cleanup.wait()
            self.finished.set()
        return "result"


def message(text: str = "Work") -> InboundMessage:
    return InboundMessage(channel_id="input", sender_id="user", content=TextContent(body=text))


async def setup(
    tool: PausedTool, *, streaming: bool = False
) -> tuple[RoomKit, Agent, MockAIProvider]:
    provider = MockAIProvider(
        streaming=streaming,
        ai_responses=[
            AIResponse(
                content="",
                tool_calls=[
                    AIToolCall(id="first", name="first", arguments={}),
                ],
            ),
            AIResponse(content="Resumed"),
        ],
    )
    provider.close = AsyncMock()
    ai = Agent(
        "ai",
        provider=provider,
        tool_handler=tool,
        tools=[
            AITool(name=name, description=name, parameters={"type": "object", "properties": {}})
            for name in ("first", "second")
        ],
    )
    kit = RoomKit()
    kit.register_channel(SimpleChannel("input"))
    kit.register_channel(ai)
    for room_id in ("owned", "other"):
        await kit.create_room(room_id=room_id)
        await kit.attach_channel(room_id, "input")
    await kit.attach_channel("owned", "ai", category=ChannelCategory.INTELLIGENCE)
    other = Agent("other-ai", provider=MockAIProvider(responses=["Other room works"]))
    kit.register_channel(other)
    await kit.attach_channel("other", "other-ai", category=ChannelCategory.INTELLIGENCE)
    return kit, ai, provider


@pytest.mark.parametrize("streaming", [False, True])
async def test_caller_cancellation_drains_tool_and_preserves_other_room(streaming: bool) -> None:
    tool = PausedTool()
    kit, ai, provider = await setup(tool, streaming=streaming)
    try:
        caller = asyncio.create_task(kit.process_inbound(message(), room_id="owned"))
        await asyncio.wait_for(tool.started.wait(), 2)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(caller, 2)
        assert tool.finished.is_set()
        assert tool.calls == ["first"]
        assert tool.effects == []
        assert len(provider.calls) == 1
        assert ai.active_turns == 0
        provider.close.assert_not_awaited()
        other = await kit.process_inbound(message(), room_id="other")
        assert other.response_events[0].content.body == "Other room works"
        resumed = await kit.process_inbound(message("Continue"), room_id="owned")
        assert resumed.response_events[-1].content.body == "Resumed"
        assert tool.calls == ["first"]
    finally:
        tool.release.set()
        tool.cleanup.set()
        await kit.close()


async def test_queued_turn_in_same_room_survives_cancelled_cascade() -> None:
    tool = PausedTool(slow_cleanup=True)
    kit, ai, provider = await setup(tool)
    try:
        first = await kit.process_inbound(message(), room_id="owned", defer_delivery=True)
        await tool.started.wait()
        second = await kit.process_inbound(
            message("Independent"), room_id="owned", defer_delivery=True
        )
        assert first.delivery is not None and second.delivery is not None
        cancelled = asyncio.create_task(first.delivery.cancel())
        await tool.unwinding.wait()
        assert not second.delivery.done
        tool.cleanup.set()
        await cancelled
        await asyncio.wait_for(second.delivery.wait(), 2)
        assert second.cancellation_reason is None
        assert second.response_events[-1].content.body == "Resumed"
        assert first.response_events == []
        assert ai.active_turns == 0
        assert len(provider.calls) == 2
    finally:
        tool.cleanup.set()
        await kit.close()


async def test_queued_cancel_does_not_wait_for_unrelated_tool() -> None:
    tool = PausedTool()
    kit, _, _ = await setup(tool)
    try:
        first = await kit.process_inbound(message(), room_id="owned", defer_delivery=True)
        await tool.started.wait()
        second = await kit.process_inbound(
            message("Never execute"), room_id="owned", defer_delivery=True
        )
        assert first.delivery is not None and second.delivery is not None
        await second.delivery.cancel(timeout=0.1)
        assert not tool.unwinding.is_set()
        assert not first.delivery.done
        assert second.delivery.done
        tool.release.set()
        await asyncio.wait_for(first.delivery.wait(), 2)
        third = await kit.process_inbound(message("Next"), room_id="owned")
        assert third.error is None
        assert second.response_events == []
    finally:
        tool.release.set()
        await kit.close()


@pytest.mark.parametrize("failure", ["cancel", "error"])
@pytest.mark.parametrize("deferred", [False, True])
async def test_post_commit_setup_failure_drains_owned_tool(failure: str, deferred: bool) -> None:
    tool = PausedTool()
    kit, ai, _ = await setup(tool)

    async def connect(*args: Any) -> None:
        await tool.started.wait()
        if failure == "error":
            raise RuntimeError("setup failed")
        await asyncio.Future()

    source = kit.get_channel("input")
    assert source is not None
    source.connect_session = connect
    try:
        # On the awaited path setup follows generation; cancellation during
        # generation still must drain it. Deferred setup overlaps the tool.
        caller = asyncio.create_task(
            kit.process_inbound(
                message().model_copy(update={"session": object()}),
                room_id="owned",
                defer_delivery=deferred,
            )
        )
        await tool.started.wait()
        if failure == "cancel":
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(caller, 2)
        else:
            if not deferred:
                tool.release.set()
            with pytest.raises(RuntimeError, match="setup failed"):
                await asyncio.wait_for(caller, 2)
        assert tool.finished.is_set()
        assert ai.active_turns == 0
    finally:
        tool.release.set()
        await kit.close()


async def test_cancel_under_room_lock_is_rejected_without_cancelling_turn() -> None:
    tool = PausedTool()
    kit, _, _ = await setup(tool)
    try:
        result = await kit.process_inbound(message(), room_id="owned", defer_delivery=True)
        assert result.delivery is not None
        await tool.started.wait()
        async with kit._lock_manager.locked("owned"):
            with pytest.raises(RuntimeError, match="lane or room lock"):
                await result.delivery.cancel()
        assert not tool.unwinding.is_set()
        await result.delivery.cancel()
    finally:
        tool.release.set()
        await kit.close()


@pytest.mark.parametrize("streaming", [False, True])
async def test_deferred_cancel_is_terminal_and_releases_all_waiters(streaming: bool) -> None:
    tool = PausedTool(slow_cleanup=True)
    kit, ai, provider = await setup(tool, streaming=streaming)
    try:
        result = await kit.process_inbound(message(), room_id="owned", defer_delivery=True)
        handle = result.delivery
        assert handle is not None
        await asyncio.wait_for(tool.started.wait(), 2)
        waiters = [asyncio.create_task(handle.wait()) for _ in range(2)]
        cancel = asyncio.create_task(handle.cancel(reason="hangup"))
        await asyncio.wait_for(tool.unwinding.wait(), 2)
        assert not handle.done
        assert not cancel.done()
        assert all(not waiter.done() for waiter in waiters)
        tool.cleanup.set()
        assert await asyncio.wait_for(cancel, 2) is result
        assert all(item is result for item in await asyncio.gather(*waiters))
        assert result.cancellation_reason == "hangup"
        assert not result.blocked
        assert result.error is None
        assert handle.done
        assert ai.active_turns == 0
        assert tool.calls == ["first"]
        assert await handle.cancel(reason="again") is result
        assert result.cancellation_reason == "hangup"
        provider.close.assert_not_awaited()
    finally:
        tool.release.set()
        tool.cleanup.set()
        await kit.close()


async def test_second_caller_cancellation_does_not_interrupt_finalizer() -> None:
    tool = PausedTool(slow_cleanup=True)
    kit, ai, _ = await setup(tool)
    try:
        caller = asyncio.create_task(kit.process_inbound(message(), room_id="owned"))
        await tool.started.wait()
        caller.cancel()
        await tool.unwinding.wait()
        caller.cancel()
        await asyncio.sleep(0)
        assert not caller.done()
        assert not tool.finished.is_set()
        tool.cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(caller, 2)
        assert tool.finished.is_set()
        assert ai.active_turns == 0
    finally:
        tool.cleanup.set()
        await kit.close()


async def test_cancel_timeout_reports_incomplete_cleanup_and_can_be_joined() -> None:
    tool = PausedTool(slow_cleanup=True)
    kit, ai, _ = await setup(tool)
    try:
        result = await kit.process_inbound(message(), room_id="owned", defer_delivery=True)
        handle = result.delivery
        assert handle is not None
        await tool.started.wait()
        with pytest.raises(TimeoutError, match="cleanup incomplete"):
            await handle.cancel(timeout=0.01)
        assert not handle.done
        assert not tool.finished.is_set()
        assert ai.active_turns == 1
        async with kit._lock_manager.locked("owned"):
            async with asyncio.timeout(0.1):
                assert await handle.wait() is result
            assert not handle.done
        tool.cleanup.set()
        await asyncio.wait_for(handle.wait(), 2)
        assert handle.done
        assert ai.active_turns == 0
    finally:
        tool.cleanup.set()
        await kit.close()


async def test_cancelling_cancel_waiter_still_finishes_cleanup() -> None:
    tool = PausedTool(slow_cleanup=True)
    kit, ai, _ = await setup(tool)
    try:
        result = await kit.process_inbound(message(), room_id="owned", defer_delivery=True)
        handle = result.delivery
        assert handle is not None
        await tool.started.wait()
        cancel = asyncio.create_task(handle.cancel())
        await tool.unwinding.wait()
        cancel.cancel()
        await asyncio.sleep(0)
        assert not cancel.done()
        tool.cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(cancel, 2)
        await handle.wait()
        assert handle.done
        assert tool.finished.is_set()
        assert ai.active_turns == 0
    finally:
        tool.cleanup.set()
        await kit.close()


async def test_cancelled_waiter_does_not_cancel_deferred_turn() -> None:
    tool = PausedTool()
    kit, _, _ = await setup(tool)
    try:
        result = await kit.process_inbound(message(), room_id="owned", defer_delivery=True)
        handle = result.delivery
        assert handle is not None
        await tool.started.wait()
        waiter = asyncio.create_task(handle.wait())
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not tool.unwinding.is_set()
        assert not handle.done
        tool.release.set()
        await asyncio.wait_for(handle.wait(), 2)
        assert result.cancellation_reason is None
        assert result.response_events[-1].content.body == "Resumed"
    finally:
        tool.release.set()
        await kit.close()


@pytest.mark.parametrize("streaming", [False, True])
async def test_handler_suppressing_cancel_cannot_start_next_tool(streaming: bool) -> None:
    tool = PausedTool(suppress=True)
    kit, _, _ = await setup(tool, streaming=streaming)
    try:
        result = await kit.process_inbound(message(), room_id="owned", defer_delivery=True)
        assert result.delivery is not None
        await tool.started.wait()
        await result.delivery.cancel(timeout=0.1)
        assert tool.calls == ["first"]
        assert tool.effects == []
    finally:
        tool.release.set()
        await kit.close()
