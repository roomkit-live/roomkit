"""A handler that raises while it reports a call leaves it reported once (RMK-507, RFC §9.3).

On every door that hands a call's report to an external handler, a handler
that raises before its report reached ON_TOOL_CALL leaves the report to the
channel, which makes it as the call ended; one that raises after has made it,
and the call is not reported a second time.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

import pytest

from roomkit import (
    ChannelCategory,
    InboundMessage,
    RoomKit,
    TextContent,
    ToolCallEvent,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AIResponse, AIToolCall, ServedCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.external import PolicyExternalToolHandler, ToolDecision
from tests.test_framework import SimpleChannel
from tests.test_refused_reports import _acp as acp_call
from tests.test_refused_reports import _acp_cut as acp_cut
from tests.test_refused_reports import _acp_permission as acp_permission
from tests.test_refused_reports import _acp_ran_anyway as acp_ran_anyway
from tests.test_refused_reports import _Heard as Heard


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


class _RejectsTheCall(_RaisesReporting):
    """Overrides that cannot take the arguments a door hands them: the call
    raises before any report runs."""

    async def on_tool_result(  # type: ignore[override]
        self, tool_name: str, tool_input: dict[str, Any], result: str, *, is_error: bool = False
    ) -> None:
        return None

    async def on_tool_cancelled(  # type: ignore[override]
        self, tool_name: str, tool_input: dict[str, Any], *, tool_call_id: str = ""
    ) -> None:
        return None


def _handler(how: str, outcome: str) -> _RaisesReporting:
    kind = _RejectsTheCall if how == "rejects-the-call" else _RaisesReporting
    return kind(
        after=how == "raises-after",
        deny=outcome in ("refused", "refused_but_ran"),
        pending=outcome == "cancelled",
    )


def _outcome(event: ToolCallEvent) -> str:
    if event.refused_but_ran:
        return "refused_but_ran"
    if event.cancelled:
        return "cancelled"
    if event.refused:
        return "refused"
    return "failed" if event.is_error else "served"


_PROVIDER_RAN = {
    "provider-served": ServedCall(result="3 hits"),
    "provider-failed": ServedCall(result="search backend down", is_error=True),
}


async def _external_door(outcome: str, how: str) -> Heard:
    """An AIChannel whose call an external handler decides, or reports as
    its provider ran it."""
    handler = _handler(how, outcome)
    call = AIToolCall(
        id="c1", name="remote", arguments={"q": "x"}, served=_PROVIDER_RAN.get(outcome)
    )
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
    heard = Heard(kit)
    message = InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="go"))
    turn = asyncio.create_task(kit.process_inbound(message))
    if outcome == "cancelled":
        await asyncio.wait_for(handler.asked.wait(), 3)
        turn.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await turn
    await asyncio.sleep(0.2)
    await kit.close()
    return heard


async def _acp_door(outcome: str, how: str) -> Heard:
    """An ACP agent's call, its end handed to the handler."""
    handler = _handler(how, outcome)
    if outcome == "served":
        return await acp_call(handler, status="completed", raw_output={"content": "read"})
    if outcome == "failed":
        return await acp_call(handler, status="failed", raw_output={"error": "no such file"})
    if outcome == "refused":
        return await acp_permission(handler)
    if outcome == "cancelled":
        return await acp_cut(handler)
    heard, _ = await acp_ran_anyway(handler)
    return heard


_CASES = [
    ("external", "provider-served"),
    ("external", "provider-failed"),
    ("external", "served"),
    ("external", "refused"),
    ("external", "cancelled"),
    ("acp", "served"),
    ("acp", "failed"),
    ("acp", "refused"),
    ("acp", "cancelled"),
    ("acp", "refused_but_ran"),
]
_REJECTING = [
    ("external", "provider-served"),
    ("external", "cancelled"),
    ("acp", "served"),
    ("acp", "cancelled"),
]
_RAISING = ("raises-before", "raises-after")
_RUNS = [
    *[(door, outcome, how) for door, outcome in _CASES for how in _RAISING],
    *[(door, outcome, "rejects-the-call") for door, outcome in _REJECTING],
]


@pytest.mark.parametrize(("door", "outcome", "how"), _RUNS)
async def test_a_handler_raising_while_reporting_leaves_the_call_reported_once(
    door: str, outcome: str, how: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Logged once, the turn goes on, and the call is reported once, as it
    ended: a refusal or a cancellation to the observers alone."""
    door_run = _external_door if door == "external" else _acp_door
    with caplog.at_level(logging.ERROR):
        heard = await door_run(outcome, how)

    expected = outcome.removeprefix("provider-")
    assert [_outcome(event) for event in heard.observed] == [expected]
    assert len(heard.sync) == (0 if expected in ("refused", "cancelled") else 1)
    failures = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert [r.getMessage().startswith("External tool handler failed") for r in failures] == [True]
