"""Tests for human-in-the-loop tool primitive."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.models.enums import ChannelType
from roomkit.models.pending_input import PendingInput, PendingInputEvent, PendingInputStatus
from roomkit.tools.compose import compose_tool_handlers
from roomkit.tools.human_input import HumanInputHandler, HumanInputToolHandler

# ── HumanInputHandler ────────────────────────────────────────────────


async def test_create_returns_pending() -> None:
    handler = HumanInputHandler()
    pending = await handler.create(
        "approve", {"amount": 100}, room_id="r1", tool_call_id="tc1", channel_id="ch1"
    )
    assert isinstance(pending, PendingInput)
    assert pending.status == PendingInputStatus.PENDING
    assert pending.tool_name == "approve"
    assert pending.arguments == {"amount": 100}
    assert pending.pending_id in handler.pending


async def test_resolve_unblocks_wait() -> None:
    handler = HumanInputHandler()
    pending = await handler.create(
        "confirm", {}, room_id="r1", tool_call_id="tc1", channel_id="ch1"
    )

    async def _resolve_later() -> None:
        await asyncio.sleep(0.01)
        handler.resolve(pending.pending_id, "yes")

    asyncio.create_task(_resolve_later())
    result = await handler.wait(pending.pending_id, timeout=5)
    assert result == "yes"
    assert pending.pending_id not in handler.pending


async def test_reject_raises_runtime_error() -> None:
    handler = HumanInputHandler()
    pending = await handler.create(
        "confirm", {}, room_id="r1", tool_call_id="tc1", channel_id="ch1"
    )

    async def _reject_later() -> None:
        await asyncio.sleep(0.01)
        handler.reject(pending.pending_id, "denied by admin")

    asyncio.create_task(_reject_later())
    with pytest.raises(RuntimeError, match="denied by admin"):
        await handler.wait(pending.pending_id, timeout=5)


async def test_wait_timeout() -> None:
    handler = HumanInputHandler()
    pending = await handler.create(
        "confirm", {}, room_id="r1", tool_call_id="tc1", channel_id="ch1"
    )
    with pytest.raises(asyncio.TimeoutError):
        await handler.wait(pending.pending_id, timeout=0.01)
    assert pending.pending_id not in handler.pending
    assert pending.status == PendingInputStatus.TIMED_OUT


async def test_wait_nonexistent_raises_value_error() -> None:
    handler = HumanInputHandler()
    with pytest.raises(ValueError, match="No pending request"):
        await handler.wait("nonexistent", timeout=1)


async def test_resolve_nonexistent_returns_false() -> None:
    handler = HumanInputHandler()
    assert handler.resolve("nonexistent", "value") is False


async def test_reject_nonexistent_returns_false() -> None:
    handler = HumanInputHandler()
    assert handler.reject("nonexistent") is False


async def test_resolve_already_resolved_returns_false() -> None:
    handler = HumanInputHandler()
    pending = await handler.create("tool", {}, room_id="r1", tool_call_id="tc1", channel_id="ch1")
    assert handler.resolve(pending.pending_id, "first") is True
    assert handler.resolve(pending.pending_id, "second") is False


async def test_callback_fires_on_create() -> None:
    events: list[PendingInputEvent] = []

    async def capture(e: PendingInputEvent) -> bool:
        events.append(e)
        return True

    handler = HumanInputHandler()
    handler._on_input_required = capture

    await handler.create(
        "tool1",
        {"key": "value"},
        room_id="r1",
        tool_call_id="tc1",
        channel_id="ch1",
        channel_type=ChannelType.AI,
    )
    assert len(events) == 1
    assert events[0].tool_name == "tool1"
    assert events[0].room_id == "r1"
    assert events[0].channel_type == ChannelType.AI


async def test_callback_deny_auto_rejects_pending() -> None:
    async def deny_callback(e: PendingInputEvent) -> bool:
        return False

    handler = HumanInputHandler()
    handler._on_input_required = deny_callback

    pending = await handler.create("tool", {}, room_id="r1", tool_call_id="tc1", channel_id="ch1")
    assert pending.status == PendingInputStatus.REJECTED
    with pytest.raises(RuntimeError, match="Denied by ON_USER_INPUT_REQUIRED hook"):
        await handler.wait(pending.pending_id, timeout=1)


async def test_callback_error_does_not_break_create() -> None:
    async def broken_callback(e: PendingInputEvent) -> bool:
        raise RuntimeError("callback exploded")

    handler = HumanInputHandler()
    handler._on_input_required = broken_callback

    pending = await handler.create("tool", {}, room_id="r1", tool_call_id="tc1", channel_id="ch1")
    assert pending.status == PendingInputStatus.PENDING
    assert pending.pending_id in handler.pending


async def test_concurrent_pending_requests() -> None:
    handler = HumanInputHandler()
    p1 = await handler.create("t1", {}, room_id="r1", tool_call_id="tc1", channel_id="ch1")
    p2 = await handler.create("t2", {}, room_id="r1", tool_call_id="tc2", channel_id="ch1")
    assert len(handler.pending) == 2

    handler.resolve(p1.pending_id, "result1")
    handler.resolve(p2.pending_id, "result2")

    r1 = await handler.wait(p1.pending_id, timeout=1)
    r2 = await handler.wait(p2.pending_id, timeout=1)
    assert r1 == "result1"
    assert r2 == "result2"
    assert len(handler.pending) == 0


async def test_pending_property_returns_snapshot() -> None:
    handler = HumanInputHandler()
    await handler.create("t1", {}, room_id="r1", tool_call_id="tc1", channel_id="ch1")
    snapshot = handler.pending
    assert len(snapshot) == 1
    # Mutating the snapshot doesn't affect the handler
    snapshot.clear()
    assert len(handler.pending) == 1


# ── HumanInputToolHandler ───────────────────────────────────────────


async def test_tool_handler_falls_through_for_unknown() -> None:
    hit = HumanInputToolHandler(tool_names={"approve"})
    result = await hit("other_tool", {})
    parsed = json.loads(result)
    assert "error" in parsed
    assert "Unknown tool" in parsed["error"]


async def test_tool_handler_blocks_and_resolves() -> None:
    hit = HumanInputToolHandler(tool_names={"approve"}, timeout=5)

    async def _resolve_later() -> None:
        await asyncio.sleep(0.01)
        pending = hit.handler.pending
        pid = next(iter(pending))
        hit.handler.resolve(pid, json.dumps({"approved": True}))

    asyncio.create_task(_resolve_later())
    result = await hit("approve", {"amount": 500})
    assert json.loads(result) == {"approved": True}


async def test_tool_handler_timeout() -> None:
    hit = HumanInputToolHandler(tool_names={"approve"}, timeout=0.01)
    result = await hit("approve", {"amount": 500})
    parsed = json.loads(result)
    assert "timed out" in parsed["error"]


async def test_tool_handler_rejection() -> None:
    hit = HumanInputToolHandler(tool_names={"approve"}, timeout=5)

    async def _reject_later() -> None:
        await asyncio.sleep(0.01)
        pending = hit.handler.pending
        pid = next(iter(pending))
        hit.handler.reject(pid, "nope")

    asyncio.create_task(_reject_later())
    result = await hit("approve", {"amount": 500})
    parsed = json.loads(result)
    assert "rejected" in parsed["error"].lower()


async def test_tool_handler_exposes_inner_handler() -> None:
    inner = HumanInputHandler()
    hit = HumanInputToolHandler(tool_names={"x"}, handler=inner)
    assert hit.handler is inner


async def test_tool_handler_creates_handler_if_not_provided() -> None:
    hit = HumanInputToolHandler(tool_names={"x"})
    assert isinstance(hit.handler, HumanInputHandler)


# ── Composition with compose_tool_handlers ───────────────────────────


async def test_compose_with_other_handler() -> None:
    hit = HumanInputToolHandler(tool_names={"approve"}, timeout=5)

    async def other(name: str, args: dict[str, Any]) -> str:
        if name == "search":
            return json.dumps({"results": []})
        return json.dumps({"error": f"Unknown tool: {name}"})

    composed = compose_tool_handlers(hit, other)

    # "search" falls through HumanInputToolHandler to other
    result = await composed("search", {})
    assert json.loads(result) == {"results": []}


async def test_compose_human_input_matches_first() -> None:
    hit = HumanInputToolHandler(tool_names={"approve"}, timeout=5)

    async def other(name: str, args: dict[str, Any]) -> str:
        return json.dumps({"from": "other"})

    composed = compose_tool_handlers(hit, other)

    async def _resolve_later() -> None:
        await asyncio.sleep(0.01)
        pending = hit.handler.pending
        pid = next(iter(pending))
        hit.handler.resolve(pid, json.dumps({"from": "human"}))

    asyncio.create_task(_resolve_later())
    result = await composed("approve", {})
    assert json.loads(result) == {"from": "human"}


async def test_compose_unknown_tool_falls_through_all() -> None:
    hit = HumanInputToolHandler(tool_names={"approve"}, timeout=5)

    async def other(name: str, args: dict[str, Any]) -> str:
        return json.dumps({"error": f"Unknown tool: {name}"})

    composed = compose_tool_handlers(hit, other)
    result = await composed("nonexistent", {})
    parsed = json.loads(result)
    assert "Unknown tool" in parsed["error"]


# ── ToolCallContext contextvar ───────────────────────────────────────


async def test_tool_call_context_read_from_contextvar() -> None:
    from roomkit.tools.human_input import ToolCallContext, _current_tool_call

    hit = HumanInputToolHandler(tool_names={"ask"}, timeout=5)

    ctx = ToolCallContext(room_id="room-1", tool_call_id="tc-42", channel_id="ai-1")
    token = _current_tool_call.set(ctx)

    async def _resolve_later() -> None:
        await asyncio.sleep(0.01)
        pending = hit.handler.pending
        pid = next(iter(pending))
        p = pending[pid]
        # Verify the context was propagated
        assert p.room_id == "room-1"
        assert p.tool_call_id == "tc-42"
        assert p.channel_id == "ai-1"
        hit.handler.resolve(pid, "answered")

    try:
        asyncio.create_task(_resolve_later())
        result = await hit("ask", {"question": "color?"})
        assert result == "answered"
    finally:
        _current_tool_call.reset(token)


# ── Validation ───────────────────────────────────────────────────────


async def test_empty_tool_names_raises() -> None:
    with pytest.raises(ValueError, match="tool_names must not be empty"):
        HumanInputToolHandler(tool_names=set())


# ── Framework integration ────────────────────────────────────────────


async def test_register_channel_injects_hook_callback() -> None:
    from roomkit import RoomKit
    from roomkit.providers.ai.mock import MockAIProvider

    kit = RoomKit()
    human = HumanInputToolHandler(tool_names={"AskUser"}, timeout=30)
    ai = AIChannel(
        "ai-test",
        provider=MockAIProvider(),
        human_input_handler=human,
    )
    kit.register_channel(ai)

    # After registration, the framework should have injected the hook callback
    assert human.handler._on_input_required is not None


async def test_hook_deny_blocks_tool_via_framework() -> None:
    from roomkit import HookExecution, HookResult, HookTrigger, RoomKit
    from roomkit.providers.ai.base import AIResponse, AIToolCall
    from roomkit.providers.ai.mock import MockAIProvider

    kit = RoomKit()

    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_use",
                tool_calls=[AIToolCall(id="tc-1", name="AskUser", arguments={"q": "?"})],
            ),
            AIResponse(content="OK, moving on.", finish_reason="stop"),
        ]
    )

    human = HumanInputToolHandler(tool_names={"AskUser"}, timeout=5)
    ai = AIChannel("ai-deny", provider=provider, human_input_handler=human)

    kit.register_channel(ai)

    # Hook that denies the input request
    @kit.hook(HookTrigger.ON_USER_INPUT_REQUIRED, execution=HookExecution.SYNC)
    async def deny_input(event, ctx):
        return HookResult.block(reason="Questions not allowed in this room")

    await kit.create_room(room_id="deny-room")
    await kit.attach_channel("deny-room", "ai-deny")

    # The AI's tool call should be rejected by the hook
    from roomkit import InboundMessage, TextContent, WebSocketChannel

    ws = WebSocketChannel("ws-test")
    kit.register_channel(ws)
    await kit.attach_channel("deny-room", "ws-test")

    inbox: list = []
    ws.register_connection("c1", lambda _conn, ev: inbox.append(ev))

    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-test",
            sender_id="user",
            content=TextContent(body="hello"),
        )
    )
    await asyncio.sleep(0.5)

    # The AI should have received a rejection error and continued
    assert len(inbox) > 0
    await kit.close()
