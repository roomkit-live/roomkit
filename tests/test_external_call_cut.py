"""A call an external handler decides, cut by its turn (RMK-419, RFC §9.3).

Every call is reported once. A pending call the turn cut is reported
cancelled through the handler's ``on_tool_cancelled``, to ON_TOOL_CALL's
observers alone, whether the handler was still deciding it, had not been
asked yet, or was reporting it. A call the provider already ran keeps its
outcome. A handler that raises while deciding fails the call, as a tool
handler that raises does.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest

from roomkit import (
    ChannelCategory,
    HookExecution,
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomKit,
    TextContent,
    ToolCallEvent,
)
from roomkit.channels.ai import AIChannel
from roomkit.models.enums import EventType
from roomkit.models.event import ToolCallContent
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall, ServedCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.external import PolicyExternalToolHandler, ToolDecision
from tests.test_framework import SimpleChannel
from tests.test_tool_call_reports import _StopAtToolStart

LOOKUP = AITool(name="lookup", description="Look up", parameters={"type": "object"})


class _Handler(PolicyExternalToolHandler):
    """An external handler that records what it is told; its approval can be
    left pending, or fail."""

    def __init__(self, *, pending: bool = False, fails: bool = False) -> None:
        super().__init__()
        self.pending = pending
        self.fails = fails
        self.asked = asyncio.Event()
        self.results: list[tuple[str, bool]] = []
        self.cancelled: list[tuple[str, str]] = []

    async def process_tool_call(
        self, tool_name: str, tool_input: dict[str, Any], **kw: Any
    ) -> ToolDecision:
        self.asked.set()
        if self.fails:
            raise RuntimeError("approval backend down")
        if self.pending:
            await asyncio.sleep(30)
        return await super().process_tool_call(tool_name, tool_input, **kw)

    async def on_tool_result(
        self, tool_name: str, tool_input: dict[str, Any], result: str, **kw: Any
    ) -> None:
        self.results.append((tool_name, kw["is_error"]))
        await super().on_tool_result(tool_name, tool_input, result, **kw)

    async def on_tool_cancelled(
        self, tool_name: str, tool_input: dict[str, Any], **kw: Any
    ) -> None:
        self.cancelled.append((tool_name, kw["tool_call_id"]))
        await super().on_tool_cancelled(tool_name, tool_input, **kw)


class _RaisesOnCut(_Handler):
    async def on_tool_cancelled(
        self, tool_name: str, tool_input: dict[str, Any], **kw: Any
    ) -> None:
        raise RuntimeError("prompt already gone")


def _call(name: str, served: str | None = None, **arguments: Any) -> AIResponse:
    """A response calling *name*; *served*, the result of a call the provider ran."""
    ran = ServedCall(result=served) if served is not None else None
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id="c1", name=name, arguments=arguments, served=ran)],
    )


async def _room(channel: AIChannel, *, stops_at_start: bool = False) -> RoomKit:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    if stops_at_start:
        kit.register_channel(_StopAtToolStart("screen"))
        await kit.attach_channel("r1", "screen")
    await kit.attach_channel("r1", channel.channel_id, category=ChannelCategory.INTELLIGENCE)
    return kit


def _observe(kit: RoomKit) -> tuple[list[ToolCallEvent], list[ToolCallEvent]]:
    """ON_TOOL_CALL's ASYNC observers and SYNC hooks, each with what it saw."""
    observed: list[ToolCallEvent] = []
    judged: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, ctx: Any) -> None:
        observed.append(event)

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="judge")
    async def judge(event: ToolCallEvent, ctx: Any) -> HookResult:
        judged.append(event)
        return HookResult.allow()

    return observed, judged


async def _until(condition: Any) -> None:
    for _ in range(200):
        if condition():
            return
        await asyncio.sleep(0.01)


def _channel(handler: _Handler, responses: list[AIResponse], **kw: Any) -> AIChannel:
    return AIChannel(
        "ai1",
        provider=MockAIProvider(ai_responses=responses, streaming=True),
        external_tool_handler=handler,
        **kw,
    )


async def _cut_while(kit: RoomKit, started: asyncio.Event) -> None:
    turn = asyncio.create_task(
        kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
        )
    )
    await asyncio.wait_for(started.wait(), 5)
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn


async def _say(kit: RoomKit) -> None:
    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )


async def test_a_call_cut_while_its_handler_decides_is_reported_cancelled() -> None:
    handler = _Handler(pending=True)
    kit = await _room(_channel(handler, [_call("Bash"), AIResponse(content="done")]))
    observed, judged = _observe(kit)

    await _cut_while(kit, handler.asked)
    await _until(lambda: bool(observed))
    await asyncio.sleep(0.05)

    assert handler.cancelled == [("Bash", "c1")]
    assert [(e.name, e.is_error, e.cancelled) for e in observed] == [("Bash", True, True)]
    # A cut call never ran outside the channel: no hook may act on it.
    assert judged == []
    await kit.close()


async def test_a_call_cut_before_its_handler_is_asked_is_reported_cancelled() -> None:
    handler = _Handler()
    channel = _channel(handler, [_call("Bash"), AIResponse(content="done")])
    kit = await _room(channel, stops_at_start=True)
    observed, _ = _observe(kit)

    await _say(kit)
    await _until(lambda: bool(observed))
    await asyncio.sleep(0.05)

    assert not handler.asked.is_set()
    assert handler.cancelled == [("Bash", "c1")]
    assert [(e.name, e.cancelled) for e in observed] == [("Bash", True)]
    await kit.close()


@pytest.mark.parametrize("with_handler", [True, False], ids=["handler", "no-handler"])
async def test_a_call_the_provider_ran_keeps_its_outcome_when_cut(with_handler: bool) -> None:
    handler = _Handler()
    ran = _call("Write", path="/tmp/n.md", served="wrote 3 bytes")
    channel = AIChannel(
        "ai1",
        provider=MockAIProvider(ai_responses=[ran, AIResponse(content="done")], streaming=True),
        external_tool_handler=handler if with_handler else None,
    )
    kit = await _room(channel, stops_at_start=True)
    observed, _ = _observe(kit)

    await _say(kit)
    await _until(lambda: bool(observed))
    await asyncio.sleep(0.05)

    assert [(e.name, e.is_error, e.cancelled, str(e.result)) for e in observed] == [
        ("Write", False, False, "wrote 3 bytes")
    ]
    assert handler.cancelled == []
    assert handler.results == ([("Write", False)] if with_handler else [])
    await kit.close()


async def test_a_call_cut_while_its_report_runs_is_reported_once_with_its_outcome() -> None:
    held = asyncio.Event()
    handler = _Handler()
    kit = await _room(_channel(handler, [_call("Bash"), AIResponse(content="done")]))
    observed: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="hold")
    async def hold(event: ToolCallEvent, ctx: Any) -> HookResult:
        if not event.cancelled:
            held.set()
            await asyncio.sleep(30)
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, ctx: Any) -> None:
        observed.append(event)

    await _cut_while(kit, held)
    await _until(lambda: bool(observed))
    await asyncio.sleep(0.05)

    # The model already read the call's outcome: the observers hear that one.
    assert [(e.name, e.is_error, e.cancelled) for e in observed] == [("Bash", False, False)]
    await kit.close()


@pytest.mark.parametrize("with_handler", [True, False], ids=["handler", "no-handler"])
async def test_a_call_the_provider_ran_cut_while_reported_keeps_its_outcome(
    with_handler: bool,
) -> None:
    held = asyncio.Event()
    handler = _Handler()
    ran = _call("Bash", cmd="rm -rf build", served="removed")
    channel = AIChannel(
        "ai1",
        provider=MockAIProvider(ai_responses=[ran, AIResponse(content="done")], streaming=True),
        external_tool_handler=handler if with_handler else None,
    )
    kit = await _room(channel)
    observed: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="hold")
    async def hold(event: ToolCallEvent, ctx: Any) -> HookResult:
        held.set()
        await asyncio.sleep(30)
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, ctx: Any) -> None:
        observed.append(event)

    framework: list[dict[str, Any]] = []

    @kit.on("tool_call")
    async def on_tool_call(event: Any) -> None:
        framework.append(dict(event.data))

    await _cut_while(kit, held)
    await _until(lambda: bool(observed))
    await asyncio.sleep(0.05)

    assert [(e.name, e.is_error, e.cancelled, str(e.result)) for e in observed] == [
        ("Bash", False, False, "removed")
    ]
    # The framework event says what the observers heard: not a failure.
    assert [(f["tool_call_id"], f.get("is_error", False)) for f in framework] == [("c1", False)]
    await kit.close()


async def test_a_call_cut_while_its_observers_run_is_reported_once() -> None:
    observing = asyncio.Event()
    handler = _Handler()
    kit = await _room(_channel(handler, [_call("Bash"), AIResponse(content="done")]))
    observed: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, ctx: Any) -> None:
        observed.append(event)
        observing.set()
        # Still observing when the turn is cancelled.
        await asyncio.sleep(1)

    await _cut_while(kit, observing)
    await asyncio.sleep(0.2)

    assert [(e.name, e.cancelled) for e in observed] == [("Bash", False)]
    await kit.close()


async def test_a_local_call_cut_is_reported_by_the_channel_not_the_handler() -> None:
    started = asyncio.Event()

    async def slow(name: str, arguments: dict[str, Any]) -> str:
        started.set()
        await asyncio.sleep(30)
        return "ok"

    handler = _Handler()
    channel = _channel(
        handler,
        [_call("lookup"), AIResponse(content="done")],
        tools=[LOOKUP],
        tool_handler=slow,
    )
    kit = await _room(channel)
    observed, _ = _observe(kit)

    await _cut_while(kit, started)
    await _until(lambda: bool(observed))

    assert handler.cancelled == []
    assert [(e.name, e.cancelled) for e in observed] == [("lookup", True)]
    await kit.close()


async def test_a_handler_failing_on_the_cut_does_not_disturb_the_turns_end(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The raise is logged, and the call is still reported cancelled, once:
    by the channel, since the handler never reported it."""
    handler = _RaisesOnCut(pending=True)
    kit = await _room(_channel(handler, [_call("Bash"), AIResponse(content="done")]))
    observed, _ = _observe(kit)

    with caplog.at_level(logging.ERROR):
        await _cut_while(kit, handler.asked)
        await _until(lambda: bool(observed))
        await asyncio.sleep(0.05)

    assert "External tool handler failed on the cut call c1" in caplog.text
    assert [(e.name, e.cancelled) for e in observed] == [("Bash", True)]
    await kit.close()


async def test_a_handler_that_raises_while_deciding_fails_the_call() -> None:
    handler = _Handler(fails=True)
    kit = await _room(_channel(handler, [_call("Bash"), AIResponse(content="done")]))
    observed, _ = _observe(kit)

    await _say(kit)
    await _until(lambda: bool(observed))

    # The tool's failure, never the exception's message (RFC §9.3).
    ends = [
        (e.content.outcome, e.content.result)
        for e in await kit.store.list_events("r1")
        if e.type == EventType.TOOL_CALL_END and isinstance(e.content, ToolCallContent)
    ]
    assert ends == [("failed", json.dumps({"error": "Tool 'Bash' failed (RuntimeError)"}))]
    assert [(e.name, e.is_error, e.cancelled) for e in observed] == [("Bash", True, False)]
    await kit.close()


class _OtherChannelHandler(PolicyExternalToolHandler):
    """Another channel's handler that reports a call of its own under an id the
    current turn also announced."""


async def test_another_channels_report_never_claims_the_turns_call() -> None:
    other = _OtherChannelHandler()
    started = asyncio.Event()

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        # Under the turn's context, another channel reports its own call "c1".
        await other.on_tool_result("Bash", {}, "ran", tool_call_id="c1", room_id="r1")
        started.set()
        return "found"

    channel = AIChannel(
        "ai1",
        provider=MockAIProvider(
            ai_responses=[_call("lookup"), AIResponse(content="done")], streaming=True
        ),
        tools=[LOOKUP],
        tool_handler=lookup,
    )
    kit = await _room(channel)
    kit.register_channel(
        AIChannel("other", provider=MockAIProvider(), external_tool_handler=other)
    )
    observed, _ = _observe(kit)

    await _say(kit)
    await _until(lambda: len(observed) == 2)

    assert sorted((e.channel_id, e.name) for e in observed) == [
        ("ai1", "lookup"),
        ("other", "Bash"),
    ]
    await kit.close()
