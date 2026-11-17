"""ON_TOOL_CALL and a call's structured copy.

A tool may publish a structured copy of its result for UI surfaces (MCP
structuredContent); the tool-call event carries it, the model never reads it.
A SYNC hook sees it and may replace or clear it, a hook that rewrites only the
result keeps it, and a blocked or failed call carries none: the event of a
withheld call must not publish what was withheld (RMK-262).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import AsyncMock

import pytest

from roomkit import (
    ChannelCategory,
    HookExecution,
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomKit,
    TextContent,
    ToolCallContent,
    ToolCallEvent,
    ToolCallVerdict,
)
from roomkit.channels.ai import AIChannel
from roomkit.models import ToolCallVerdict as ModelVerdict
from roomkit.models.enums import ChannelType, EventType
from roomkit.providers.ai.base import AIContext, AIMessage, AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools import current_tool_call
from tests.test_framework import SimpleChannel
from tests.tool_loop_modes import run_tool_loop

_COPY = {"contact": "Jane Doe", "salary": 120_000}


def test_standalone_verdict_defaults_and_frozen_fields() -> None:
    assert ModelVerdict is ToolCallVerdict
    verdict = ToolCallVerdict()
    assert verdict.result is None
    assert not verdict.blocked
    assert not verdict.replaces_structured
    assert verdict.structured_content is None
    assert verdict.error_detail is None
    with pytest.raises(FrozenInstanceError):
        verdict.blocked = True  # type: ignore[misc]


def test_verdict_keeps_its_list_and_the_shared_structured_object() -> None:
    obj = {"original": 1}
    parts: list[Any] = [obj]
    verdict = ToolCallVerdict(result=parts, structured_content=obj)
    parts.append("late")
    obj.update({"second": 2, "original": 3})
    assert verdict.result is parts
    assert verdict.structured_content is obj
    assert verdict.result[0] is verdict.structured_content
    assert verdict.result == [{"original": 3, "second": 2}, "late"]
    assert list(verdict.structured_content) == ["original", "second"]


async def _publishing_handler(name: str, args: dict[str, Any]) -> str:
    call = current_tool_call()
    assert call is not None
    call.structured_content = dict(_COPY)
    return '{"contact": "Jane Doe", "salary": 120000}'


async def _kit(*, streaming: bool) -> tuple[RoomKit, MockAIProvider]:
    provider = MockAIProvider(
        streaming=streaming,
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="c1", name="hr_lookup", arguments={})],
            ),
            AIResponse(content="done", finish_reason="stop"),
        ],
    )
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(
        AIChannel(
            "ai1",
            provider=provider,
            tool_handler=_publishing_handler,
            tools=[AITool(name="hr_lookup", description="HR lookup")],
        )
    )
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    return kit, provider


async def _tool_end(kit: RoomKit) -> ToolCallContent:
    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )
    ends = [e for e in await kit.store.list_events("r1") if e.type == EventType.TOOL_CALL_END]
    assert len(ends) == 1
    assert isinstance(ends[0].content, ToolCallContent)
    return ends[0].content


async def test_a_blocked_call_is_failed_and_carries_no_structured_copy(streaming: bool) -> None:
    kit, provider = await _kit(streaming=streaming)

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="withhold")
    async def withhold(event: ToolCallEvent, ctx: Any) -> HookResult:
        return HookResult.block("restricted: not for this room")

    end = await _tool_end(kit)

    assert end.status == "failed"
    assert end.structured_content is None
    tool_message = next(m for m in provider.calls[-1].messages if m.role == "tool")
    assert "restricted: not for this room" in tool_message.content[0].result


async def test_the_hook_sees_the_copy_and_may_replace_it(streaming: bool) -> None:
    kit, _ = await _kit(streaming=streaming)
    seen: list[Any] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="redact")
    async def redact(event: ToolCallEvent, ctx: Any) -> HookResult:
        seen.append(event.structured_content)
        copy = {**(event.structured_content or {}), "contact": "[PERSON_1]"}
        return HookResult(action="allow", metadata={"structured_content": copy})

    end = await _tool_end(kit)

    assert seen == [_COPY]
    assert end.structured_content == {"contact": "[PERSON_1]", "salary": 120_000}
    assert end.status == "completed"


async def test_the_hook_may_clear_the_copy(streaming: bool) -> None:
    kit, _ = await _kit(streaming=streaming)

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="clear")
    async def clear(event: ToolCallEvent, ctx: Any) -> HookResult:
        return HookResult(action="allow", metadata={"structured_content": None})

    end = await _tool_end(kit)

    assert end.structured_content is None
    assert end.status == "completed"


async def test_a_result_rewrite_alone_keeps_the_copy(streaming: bool) -> None:
    """Re-tokenising text is not withholding the payload a widget renders."""
    kit, _ = await _kit(streaming=streaming)

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="tokenise")
    async def tokenise(event: ToolCallEvent, ctx: Any) -> HookResult:
        text = str(event.result).replace("Jane Doe", "[PERSON_1]")
        return HookResult(action="allow", metadata={"result": text})

    end = await _tool_end(kit)

    assert "[PERSON_1]" in str(end.result)
    assert end.structured_content == _COPY


async def test_a_call_that_fails_after_its_copy_was_captured_keeps_none(streaming: bool) -> None:
    """The copy is read once the handler returns; a failure past that point
    (here the ON_TOOL_CALL dispatch itself) must not publish it."""
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="c1", name="hr_lookup", arguments={})],
            ),
            AIResponse(content="done", finish_reason="stop"),
        ]
    )
    ch = AIChannel("ai1", provider=provider, tool_handler=_publishing_handler)
    ch._tool_call_hook = AsyncMock(side_effect=RuntimeError("hook dispatch broke"))

    await run_tool_loop(ch, AIContext(messages=[AIMessage(role="user", content="go")]))

    tool_message = next(m for m in provider.calls[-1].messages if m.role == "tool")
    part = tool_message.content[0]
    assert part.is_error
    assert part.structured_content is None


async def test_a_copy_that_is_not_a_mapping_is_dropped_and_the_turn_goes_on(
    streaming: bool,
) -> None:
    kit, provider = await _kit(streaming=streaming)

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="broken")
    async def broken(event: ToolCallEvent, ctx: Any) -> HookResult:
        return HookResult(action="allow", metadata={"structured_content": ["not", "a", "dict"]})

    end = await _tool_end(kit)

    assert end.structured_content is None
    assert end.status == "completed"
    assert provider.calls[-1].messages[-1].role == "tool"


async def test_a_block_without_a_reason_never_serves_the_original(streaming: bool) -> None:
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="c1", name="hr_lookup", arguments={})],
            ),
            AIResponse(content="done", finish_reason="stop"),
        ]
    )
    ch = AIChannel("ai1", provider=provider, tool_handler=_publishing_handler)
    ch._tool_call_hook = AsyncMock(return_value=ToolCallVerdict(blocked=True))

    await run_tool_loop(ch, AIContext(messages=[AIMessage(role="user", content="go")]))

    part = next(m for m in provider.calls[-1].messages if m.role == "tool").content[0]
    assert part.is_error
    assert "Jane Doe" not in str(part.result)
    assert part.structured_content is None


async def test_a_fail_closed_hook_that_cannot_run_withholds_the_result() -> None:
    """The room's context would not build, so no ON_TOOL_CALL hook runs: a
    fail-closed one withholds, as its own failure would."""
    kit, _ = await _kit(streaming=True)

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="pii", fail_closed=True)
    async def pii(event: ToolCallEvent, ctx: Any) -> HookResult:
        return HookResult.allow()

    callback = kit._build_tool_call_hook("ai1")
    kit._build_context = AsyncMock(side_effect=RuntimeError("store down"))  # type: ignore[method-assign]
    event = ToolCallEvent(
        channel_id="ai1",
        channel_type=ChannelType.AI,
        tool_call_id="c1",
        name="hr_lookup",
        arguments={},
        result="Jane Doe",
        room_id="r1",
    )

    verdict = await callback(event)

    assert verdict is not None and verdict.blocked
    assert "hook_error:pii" in str(verdict.result)


async def test_without_a_fail_closed_hook_the_result_stands() -> None:
    """The room's context would not build for a hook that does not fail
    closed: the result stands."""
    kit, _ = await _kit(streaming=True)

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="audit")
    async def audit(event: ToolCallEvent, ctx: Any) -> HookResult:
        return HookResult.allow()

    callback = kit._build_tool_call_hook("ai1")
    kit._build_context = AsyncMock(side_effect=RuntimeError("store down"))  # type: ignore[method-assign]
    event = ToolCallEvent(
        channel_id="ai1",
        channel_type=ChannelType.AI,
        tool_call_id="c1",
        name="hr_lookup",
        arguments={},
        result="ok",
        room_id="r1",
    )

    assert await callback(event) is None
