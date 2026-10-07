"""``kit.close()`` ends a turn someone awaits as it ends the same turn on the
neighbouring door (RMK-526, RFC §23.3, §10.1 step 18).

An inline delegation ends ``cancelled``, ON_TASK_COMPLETED fired before the
store is sealed, as a background one does. An awaited room turn returns its
``cancelled`` turn, as a deferred one does, never a ``CancelledError`` its
caller did not ask for. A ``close()`` called from inside work the kit holds
does not wait for that work.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.agent import Agent
from roomkit.core.task_utils import hold_task
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory
from roomkit.models.event import TextContent
from roomkit.providers.ai.base import (
    AIContext,
    AIResponse,
    AIToolCall,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
)
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel


class _SlowWords(MockAIProvider):
    """Streams a long answer word by word, and says when it started."""

    def __init__(self, started: asyncio.Event) -> None:
        super().__init__(streaming=True)
        self.started = started

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        self.calls.append(context)
        yield StreamTextDelta(text="The ")
        self.started.set()
        for i in range(2000):
            await asyncio.sleep(0.01)
            yield StreamTextDelta(text=f"w{i} ")
        yield StreamDone(finish_reason="stop", usage={"input_tokens": 5, "output_tokens": 50})


def _ended_tasks(kit: RoomKit) -> list[tuple[str, Any]]:
    ended: list[tuple[str, Any]] = []

    @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
    async def on_task(event: Any, context: Any) -> None:
        ended.append((str(event.metadata.get("task_status")), event.metadata.get("error")))

    return ended


# -- a delegation under kit.close() --------------------------------------------


@pytest.mark.parametrize("shared", [False, True], ids=["alone", "shared-channel"])
@pytest.mark.parametrize("wait", [True, False], ids=["inline", "background"])
async def test_a_delegation_ends_cancelled_with_its_hook(
    wait: bool, shared: bool, caplog: pytest.LogCaptureFixture
) -> None:
    started = asyncio.Event()
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(Agent("worker", provider=_SlowWords(started)))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    ended = _ended_tasks(kit)
    share = ["sms"] if shared else None

    delegation = asyncio.ensure_future(
        kit.delegate("r", "worker", "Go.", wait=wait, share_channels=share)
    )
    if not wait:
        await delegation
    await started.wait()
    with caplog.at_level(logging.WARNING, logger="roomkit"):
        await kit.close()
        task = await asyncio.wait_for(delegation, timeout=5.0)

    assert task.result is not None
    assert (task.result.status, task.result.error, task.result.output) == (
        "cancelled",
        "cancelled",
        None,
    )
    assert ended == [("cancelled", "cancelled")]
    assert [r.message for r in caplog.records if r.levelno >= logging.WARNING] == []


async def test_a_close_from_a_held_run_does_not_wait_for_it() -> None:
    kit = RoomKit()
    closed = asyncio.Event()

    async def run() -> None:
        await kit.close()
        closed.set()

    hold_task(kit, run())
    await asyncio.wait_for(closed.wait(), timeout=5.0)


async def test_a_close_from_an_inline_workers_tool_does_not_wait_for_it() -> None:
    kit = RoomKit()
    closed = asyncio.Event()

    async def shutdown(name: str, arguments: dict[str, Any]) -> str:
        await kit.close()
        closed.set()
        return "closed"

    tool = {"name": "shutdown", "description": "Shut down.", "parameters": {"type": "object"}}
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="c1", name="shutdown", arguments={})],
            ),
            AIResponse(content="done", finish_reason="stop"),
        ]
    )
    kit.register_channel(Agent("worker", provider=provider, tools=[tool], tool_handler=shutdown))
    kit.register_channel(SimpleChannel("sms"))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")

    delegation = asyncio.ensure_future(kit.delegate("r", "worker", "Go.", wait=True))
    await asyncio.wait_for(closed.wait(), timeout=5.0)
    await asyncio.wait_for(asyncio.gather(delegation, return_exceptions=True), timeout=5.0)


# -- a room turn under kit.close() ----------------------------------------------


@pytest.mark.parametrize("entry", ["process_inbound", "send_event"])
@pytest.mark.parametrize("door", ["awaited", "deferred"])
async def test_a_room_turn_returns_cancelled_to_its_caller(
    door: str, entry: str, caplog: pytest.LogCaptureFixture
) -> None:
    started = asyncio.Event()
    kit = RoomKit()
    kit.register_channel(SimpleChannel("tx"))
    kit.register_channel(Agent("agent", provider=_SlowWords(started)))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "tx")
    await kit.attach_channel("r", "agent", category=ChannelCategory.INTELLIGENCE)
    responses: list[str] = []

    @kit.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC)
    async def on_response(event: Any, context: Any) -> None:
        responses.append(str(event.loop_end_reason))

    if entry == "send_event":
        if door == "deferred":
            pytest.skip("send_event has no deferred door")
        call: Any = asyncio.ensure_future(kit.send_event("r", "tx", TextContent(body="Go.")))
    else:
        message = InboundMessage(channel_id="tx", sender_id="u", content=TextContent(body="Go."))
        call = asyncio.ensure_future(
            kit.process_inbound(message, defer_delivery=door == "deferred")
        )
    await started.wait()
    with caplog.at_level(logging.WARNING, logger="roomkit"):
        await kit.close()
        result = await asyncio.wait_for(call, timeout=5.0)
        if door == "deferred":
            await asyncio.wait_for(result.delivery.wait(), timeout=5.0)

    if entry == "process_inbound":
        turns = dict(result.response_metadata).get("turns", {})
        assert turns["agent"]["loop_end_reason"] == "cancelled"
        assert result.error is None
    assert responses == []
    assert [r.message for r in caplog.records if r.levelno >= logging.WARNING] == []
