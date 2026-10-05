"""A handler that raises while it reports a call leaves it reported once (RMK-507, RFC §9.3).

On every door that hands a call's report to an external handler, a handler
that raises before its report reached ON_TOOL_CALL leaves the report to the
channel, which makes it as the call ended; one that raises after has made it,
and the call is not reported a second time.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from roomkit import (
    ChannelCategory,
    HookExecution,
    HookTrigger,
    InboundMessage,
    RoomKit,
    TextContent,
    ToolCallEvent,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.external import PolicyExternalToolHandler, ToolDecision
from tests.test_framework import SimpleChannel
from tests.test_refused_reports import _acp as acp_call
from tests.test_refused_reports import _acp_cut as acp_cut
from tests.test_refused_reports import _acp_permission as acp_permission
from tests.test_refused_reports import _acp_ran_anyway as acp_ran_anyway


class _RaisesReporting(PolicyExternalToolHandler):
    """Approves, refuses or holds a call, then raises reporting it: before
    its report reached ON_TOOL_CALL, or after (*after*)."""

    def __init__(self, *, after: bool, deny: bool = False, pending: bool = False) -> None:
        super().__init__()
        self.after = after
        self.deny = deny
        self.pending = pending
        self.asked = asyncio.Event()

    async def process_tool_call(
        self, tool_name: str, tool_input: dict[str, Any], **kw: Any
    ) -> ToolDecision:
        self.asked.set()
        if self.pending:
            await asyncio.sleep(30)
        if self.deny:
            return ToolDecision(approved=False, reason="not on this host")
        return await super().process_tool_call(tool_name, tool_input, **kw)

    async def on_tool_result(
        self, tool_name: str, tool_input: dict[str, Any], result: str, **kw: Any
    ) -> None:
        if self.after:
            await super().on_tool_result(tool_name, tool_input, result, **kw)
        raise RuntimeError("audit sink down")

    async def on_tool_refused(
        self, tool_name: str, tool_input: dict[str, Any], result: str, **kw: Any
    ) -> None:
        if self.after:
            await super().on_tool_refused(tool_name, tool_input, result, **kw)
        raise RuntimeError("audit sink down")

    async def on_tool_cancelled(
        self, tool_name: str, tool_input: dict[str, Any], **kw: Any
    ) -> None:
        if self.after:
            await super().on_tool_cancelled(tool_name, tool_input, **kw)
        raise RuntimeError("audit sink down")


def _observe(kit: RoomKit) -> list[ToolCallEvent]:
    observed: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, ctx: Any) -> None:
        observed.append(event)

    return observed


def _outcome(event: ToolCallEvent) -> str:
    if event.refused_but_ran:
        return "refused_but_ran"
    if event.cancelled:
        return "cancelled"
    if event.refused:
        return "refused"
    return "failed" if event.is_error else "served"


async def _external_door(outcome: str, after: bool) -> list[ToolCallEvent]:
    """An AIChannel whose provider's call an external handler decides."""
    handler = _RaisesReporting(
        after=after, deny=outcome == "refused", pending=outcome == "cancelled"
    )
    call = AIToolCall(id="c1", name="remote", arguments={"q": "x"})
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(content="", finish_reason="tool_calls", tool_calls=[call]),
            AIResponse(content="done"),
        ]
    )
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(AIChannel("ai", provider=provider, external_tool_handler=handler))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    await kit.attach_channel("r", "ai", category=ChannelCategory.INTELLIGENCE)
    observed = _observe(kit)
    message = InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="go"))
    turn = asyncio.create_task(kit.process_inbound(message))
    if outcome == "cancelled":
        await asyncio.wait_for(handler.asked.wait(), 3)
        turn.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await turn
    await asyncio.sleep(0.2)
    await kit.close()
    return observed


async def _acp_door(outcome: str, after: bool) -> list[ToolCallEvent]:
    """An ACP agent's call, its end handed to the handler."""
    handler = _RaisesReporting(after=after, deny=outcome in ("refused", "refused_but_ran"))
    if outcome == "served":
        heard = await acp_call(handler, status="completed", raw_output={"content": "read"})
    elif outcome == "failed":
        heard = await acp_call(handler, status="failed", raw_output={"error": "no such file"})
    elif outcome == "refused":
        heard = await acp_permission(handler)
    elif outcome == "cancelled":
        heard = await acp_cut(handler)
    else:
        heard, _ = await acp_ran_anyway(handler)
    return heard.observed


_CASES = [
    ("external", "served"),
    ("external", "refused"),
    ("external", "cancelled"),
    ("acp", "served"),
    ("acp", "failed"),
    ("acp", "refused"),
    ("acp", "cancelled"),
    ("acp", "refused_but_ran"),
]


@pytest.mark.parametrize("after", [False, True], ids=["raises-before", "raises-after"])
@pytest.mark.parametrize(("door", "outcome"), _CASES)
async def test_a_handler_raising_while_reporting_leaves_the_call_reported_once(
    door: str, outcome: str, after: bool
) -> None:
    door_run = _external_door if door == "external" else _acp_door
    observed = await door_run(outcome, after)

    assert [_outcome(event) for event in observed] == [outcome]
