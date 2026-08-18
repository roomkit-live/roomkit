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
    fired = asyncio.Event()

    async def capture(e: PendingInputEvent) -> bool:
        events.append(e)
        fired.set()
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
    await asyncio.wait_for(fired.wait(), timeout=1)
    assert len(events) == 1
    assert events[0].tool_name == "tool1"
    assert events[0].room_id == "r1"
    assert events[0].channel_type == ChannelType.AI


async def test_notification_does_not_gate_the_answer() -> None:
    """A human answering mid-notification answers a request already listening."""
    answered = asyncio.Event()

    async def slow_notify(e: PendingInputEvent) -> bool:
        # Only returns once the answer has been collected — if create()
        # awaited this, nothing would ever reach wait().
        await answered.wait()
        return True

    handler = HumanInputHandler()
    handler._on_input_required = slow_notify

    pending = await handler.create("tool", {}, room_id="r1", tool_call_id="tc1", channel_id="ch1")
    waiter = asyncio.create_task(handler.wait(pending.pending_id, timeout=5))
    await asyncio.sleep(0)  # let wait() reach the event

    handler.resolve(pending.pending_id, "Dark")
    assert await asyncio.wait_for(waiter, timeout=1) == "Dark"

    answered.set()  # release the notification task


async def test_callback_deny_rejects_the_request() -> None:
    async def deny_callback(e: PendingInputEvent) -> bool:
        return False

    handler = HumanInputHandler()
    handler._on_input_required = deny_callback

    pending = await handler.create("tool", {}, room_id="r1", tool_call_id="tc1", channel_id="ch1")
    with pytest.raises(RuntimeError, match="Denied by ON_USER_INPUT_REQUIRED hook"):
        await handler.wait(pending.pending_id, timeout=1)


async def test_callback_error_leaves_the_request_answerable() -> None:
    exploded = asyncio.Event()

    async def broken_callback(e: PendingInputEvent) -> bool:
        exploded.set()
        raise RuntimeError("callback exploded")

    handler = HumanInputHandler()
    handler._on_input_required = broken_callback

    pending = await handler.create("tool", {}, room_id="r1", tool_call_id="tc1", channel_id="ch1")
    await asyncio.wait_for(exploded.wait(), timeout=1)
    assert pending.status == PendingInputStatus.PENDING
    assert pending.pending_id in handler.pending

    handler.resolve(pending.pending_id, "still works")
    assert await handler.wait(pending.pending_id, timeout=1) == "still works"


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


async def test_close_cancels_only_one_channels_notifications_and_requests() -> None:
    started = {"ch1": asyncio.Event(), "ch2": asyncio.Event()}
    cancelled: set[str] = set()

    async def hanging_callback(event: PendingInputEvent) -> bool:
        started[event.channel_id].set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.add(event.channel_id)
        return True

    handler = HumanInputHandler()
    handler._on_input_required = hanging_callback
    first = await handler.create("ask", {}, channel_id="ch1")
    second = await handler.create("ask", {}, channel_id="ch2")
    await asyncio.gather(started["ch1"].wait(), started["ch2"].wait())

    await handler.close(channel_id="ch1")

    assert cancelled == {"ch1"}
    with pytest.raises(RuntimeError, match="handler closed"):
        await handler.wait(first.pending_id)
    assert second.pending_id in handler.pending

    handler.resolve(second.pending_id, "ok")
    assert await handler.wait(second.pending_id) == "ok"
    await handler.close()
    assert cancelled == {"ch1", "ch2"}


async def test_close_prevents_new_requests_in_the_closed_scope() -> None:
    handler = HumanInputHandler()

    await handler.close(channel_id="ch1")

    with pytest.raises(RuntimeError, match="closed for channel ch1"):
        await handler.create("ask", {}, channel_id="ch1")

    other = await handler.create("ask", {}, channel_id="ch2")
    assert other.pending_id in handler.pending

    await handler.close()
    with pytest.raises(RuntimeError, match="handler is closed"):
        await handler.create("ask", {}, channel_id="ch3")


async def test_registering_reopens_a_closed_channel_scope() -> None:
    async def callback(event: PendingInputEvent) -> bool:
        return True

    handler = HumanInputHandler()
    await handler.close(channel_id="ch1")
    with pytest.raises(RuntimeError, match="closed for channel ch1"):
        await handler.create("ask", {}, channel_id="ch1")

    # A channel object registering under that id is a new owner, not late
    # work from the one that closed.
    handler._set_on_input_required("ch1", callback)

    reopened = await handler.create("ask", {}, channel_id="ch1")
    assert reopened.pending_id in handler.pending


async def test_displaced_owner_close_spares_the_live_channel() -> None:
    async def callback(event: PendingInputEvent) -> bool:
        return True

    handler = HumanInputHandler()
    displaced = handler._set_on_input_required("ch1", callback)
    live = handler._set_on_input_required("ch1", callback)
    assert displaced != live

    pending = await handler.create("ask", {}, channel_id="ch1")
    await handler.close(channel_id="ch1", registration=displaced)

    assert "ch1" in handler._on_input_required_by_channel
    assert pending.status == PendingInputStatus.PENDING
    still_armable = await handler.create("ask", {}, channel_id="ch1")
    assert still_armable.pending_id in handler.pending

    # The owner that does hold the id still closes it.
    await handler.close(channel_id="ch1", registration=live)
    with pytest.raises(RuntimeError, match="closed for channel ch1"):
        await handler.create("ask", {}, channel_id="ch1")


async def test_ai_channel_close_stops_its_human_input_notifications() -> None:
    from roomkit.providers.ai.mock import MockAIProvider

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def hanging_callback(event: PendingInputEvent) -> bool:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return True

    human = HumanInputToolHandler(tool_names={"ask"})
    human.handler._on_input_required = hanging_callback
    channel = AIChannel("ai-close", provider=MockAIProvider(), human_input_handler=human)
    pending = await human.handler.create("ask", {}, channel_id="ai-close")
    await started.wait()

    await channel.close()

    await asyncio.wait_for(cancelled.wait(), timeout=1)
    with pytest.raises(RuntimeError, match="handler closed"):
        await human.handler.wait(pending.pending_id)


async def test_pending_property_returns_snapshot() -> None:
    handler = HumanInputHandler()
    await handler.create("t1", {}, room_id="r1", tool_call_id="tc1", channel_id="ch1")
    snapshot = handler.pending
    assert len(snapshot) == 1
    # Mutating the snapshot doesn't affect the handler
    snapshot.clear()
    assert len(handler.pending) == 1


# ── Retention of consumed outcomes ──────────────────────────────────


async def test_wait_replays_a_consumed_answer() -> None:
    handler = HumanInputHandler()
    pending = await handler.create("tool", {}, room_id="r1", tool_call_id="tc1", channel_id="ch1")
    handler.resolve(pending.pending_id, "Dark")

    assert await handler.wait(pending.pending_id, timeout=1) == "Dark"
    assert pending.pending_id not in handler.pending
    # A second waiter reads the answer, not a ValueError.
    assert await handler.wait(pending.pending_id, timeout=1) == "Dark"


async def test_wait_reads_an_answer_the_host_dropped() -> None:
    """The incident: a host doing its own bookkeeping drops the request on
    resolve(), and the tool reaches wait() afterwards."""
    handler = HumanInputHandler()
    pending = await handler.create("tool", {}, room_id="r1", tool_call_id="tc1", channel_id="ch1")
    handler.resolve(pending.pending_id, "Dark")
    handler._pending.pop(pending.pending_id, None)

    assert await handler.wait(pending.pending_id, timeout=1) == "Dark"


async def test_wait_replays_a_rejection() -> None:
    handler = HumanInputHandler()
    pending = await handler.create("tool", {}, room_id="r1", tool_call_id="tc1", channel_id="ch1")
    handler.reject(pending.pending_id, "denied by admin")

    with pytest.raises(RuntimeError, match="denied by admin"):
        await handler.wait(pending.pending_id, timeout=1)
    with pytest.raises(RuntimeError, match="denied by admin"):
        await handler.wait(pending.pending_id, timeout=1)


async def test_wait_replays_a_timeout() -> None:
    handler = HumanInputHandler()
    pending = await handler.create("tool", {}, room_id="r1", tool_call_id="tc1", channel_id="ch1")

    with pytest.raises(asyncio.TimeoutError):
        await handler.wait(pending.pending_id, timeout=0.01)
    with pytest.raises(asyncio.TimeoutError):
        await handler.wait(pending.pending_id, timeout=0.01)


async def test_retention_is_bounded() -> None:
    handler = HumanInputHandler(retention=1)
    first = await handler.create("t1", {}, room_id="r1", tool_call_id="tc1", channel_id="ch1")
    second = await handler.create("t2", {}, room_id="r1", tool_call_id="tc2", channel_id="ch1")
    handler.resolve(first.pending_id, "one")
    handler.resolve(second.pending_id, "two")

    assert await handler.wait(first.pending_id, timeout=1) == "one"
    assert await handler.wait(second.pending_id, timeout=1) == "two"

    # The newest outcome is still readable; the oldest has been evicted.
    assert await handler.wait(second.pending_id, timeout=1) == "two"
    with pytest.raises(ValueError, match="No pending request"):
        await handler.wait(first.pending_id, timeout=1)


async def test_retention_can_be_switched_off() -> None:
    handler = HumanInputHandler(retention=0)
    pending = await handler.create("tool", {}, room_id="r1", tool_call_id="tc1", channel_id="ch1")
    handler.resolve(pending.pending_id, "Dark")

    assert await handler.wait(pending.pending_id, timeout=1) == "Dark"
    with pytest.raises(ValueError, match="No pending request"):
        await handler.wait(pending.pending_id, timeout=1)


# ── Detached requests ───────────────────────────────────────────────


async def test_create_detached_marks_the_request() -> None:
    handler = HumanInputHandler()
    attached = await handler.create("t1", {}, room_id="r1", tool_call_id="tc1", channel_id="ch1")
    detached = await handler.create_detached(
        "t2", {}, room_id="r1", tool_call_id="tc2", channel_id="ch1"
    )
    assert attached.detached is False
    assert detached.detached is True


async def test_release_drops_the_request_but_keeps_its_answer() -> None:
    handler = HumanInputHandler()
    pending = await handler.create_detached(
        "tool", {}, room_id="r1", tool_call_id="tc1", channel_id="ch1"
    )
    handler.resolve(pending.pending_id, "Dark")

    assert handler.release(pending.pending_id) is True
    assert pending.pending_id not in handler.pending
    assert await handler.wait(pending.pending_id, timeout=1) == "Dark"


async def test_release_rejects_a_request_still_unanswered() -> None:
    handler = HumanInputHandler()
    pending = await handler.create_detached(
        "tool", {}, room_id="r1", tool_call_id="tc1", channel_id="ch1"
    )
    waiter = asyncio.create_task(handler.wait(pending.pending_id, timeout=5))
    await asyncio.sleep(0)

    assert handler.release(pending.pending_id) is True
    with pytest.raises(RuntimeError, match="Released before an answer arrived"):
        await asyncio.wait_for(waiter, timeout=1)


async def test_release_unknown_returns_false() -> None:
    handler = HumanInputHandler()
    assert handler.release("nonexistent") is False


async def test_release_is_idempotent() -> None:
    handler = HumanInputHandler()
    pending = await handler.create_detached(
        "tool", {}, room_id="r1", tool_call_id="tc1", channel_id="ch1"
    )
    handler.resolve(pending.pending_id, "Dark")
    assert handler.release(pending.pending_id) is True
    assert handler.release(pending.pending_id) is False


# ── Who the request is for ──────────────────────────────────────────


async def test_create_carries_the_actor_to_the_request_and_the_event() -> None:
    handler = HumanInputHandler()
    events: list[PendingInputEvent] = []

    async def capture(event: PendingInputEvent) -> bool:
        events.append(event)
        return True

    handler._on_input_required = capture
    pending = await handler.create(
        "ask", {}, room_id="r1", tool_call_id="tc1", channel_id="ch1", actor_id="alice"
    )
    await asyncio.sleep(0.01)

    assert pending.actor_id == "alice"
    assert [e.actor_id for e in events] == ["alice"]


async def test_a_request_with_no_actor_says_so() -> None:
    """A creator running its own tool loop may not know who asked. ``None``
    is the honest answer — the alternative is a notification layer inventing
    a recipient."""
    handler = HumanInputHandler()
    pending = await handler.create_detached("ask", {}, room_id="r1", channel_id="ch1")
    assert pending.actor_id is None


async def test_request_names_the_speaker_through_the_framework() -> None:
    """Two people, one agent, one channel object: each request names the
    person whose turn raised it."""
    from roomkit import (
        HookExecution,
        HookResult,
        HookTrigger,
        InboundMessage,
        RoomKit,
        TextContent,
        WebSocketChannel,
    )
    from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
    from roomkit.providers.ai.mock import MockAIProvider

    def turn() -> list[AIResponse]:
        return [
            AIResponse(
                content="",
                finish_reason="tool_use",
                tool_calls=[AIToolCall(id="tc", name="AskUser", arguments={"q": "?"})],
            ),
            AIResponse(content="Thanks.", finish_reason="stop"),
        ]

    kit = RoomKit()
    human = HumanInputToolHandler(
        tool_names={"AskUser"},
        timeout=5,
        # Without a definition the tool is absent from the turn's toolset and
        # the loop drops the call before any handler sees it.
        tool_definitions=[
            AITool(
                name="AskUser",
                description="Ask the human.",
                parameters={"type": "object", "properties": {}},
            )
        ],
    )
    ai = AIChannel(
        "ai-ask",
        provider=MockAIProvider(ai_responses=turn() * 2),
        human_input_handler=human,
    )
    ws = WebSocketChannel("ws-ask")
    kit.register_channel(ai)
    kit.register_channel(ws)

    seen: list[Any] = []

    @kit.hook(HookTrigger.ON_USER_INPUT_REQUIRED, execution=HookExecution.SYNC)
    async def capture(event: Any, ctx: Any) -> HookResult:
        seen.append(event)
        human.handler.resolve(event.pending_id, json.dumps({"answer": "yes"}))
        return HookResult(action="allow")

    await kit.create_room(room_id="ask-room")
    await kit.attach_channel("ask-room", "ai-ask")
    await kit.attach_channel("ask-room", "ws-ask")
    # Someone has to drain the AI's stream for the tool loop to advance.
    ws.register_connection("c1", lambda _conn, _ev: None, room_id="ask-room")

    for who in ("alice", "bob"):
        await kit.process_inbound(
            InboundMessage(channel_id="ws-ask", sender_id=who, content=TextContent(body="hi"))
        )
        await asyncio.sleep(0.3)

    assert [e.actor_id for e in seen] == ["alice", "bob"]
    await kit.close()


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
    from roomkit.tools.context import ToolCallContext, _current_tool_call

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
    assert "ai-test" in human.handler._on_input_required_by_channel


async def test_shared_handler_routes_framework_callbacks_by_channel() -> None:
    from roomkit import RoomKit
    from roomkit.providers.ai.mock import MockAIProvider

    shared = HumanInputHandler()
    first_human = HumanInputToolHandler(tool_names={"AskUser"}, handler=shared)
    second_human = HumanInputToolHandler(tool_names={"AskUser"}, handler=shared)
    first_kit = RoomKit()
    second_kit = RoomKit()
    first_kit.register_channel(
        AIChannel("ai-first", provider=MockAIProvider(), human_input_handler=first_human)
    )
    second_kit.register_channel(
        AIChannel("ai-second", provider=MockAIProvider(), human_input_handler=second_human)
    )
    await first_kit.create_room(room_id="room-first")
    await second_kit.create_room(room_id="room-second")

    first_events: list[Any] = []
    second_events: list[Any] = []
    first_fired = asyncio.Event()
    second_fired = asyncio.Event()

    @first_kit.on("user_input_required")
    async def on_first(event: Any) -> None:
        first_events.append(event)
        first_fired.set()

    @second_kit.on("user_input_required")
    async def on_second(event: Any) -> None:
        second_events.append(event)
        second_fired.set()

    first = await shared.create_detached(
        "AskUser", {}, room_id="room-first", channel_id="ai-first"
    )
    second = await shared.create_detached(
        "AskUser", {}, room_id="room-second", channel_id="ai-second"
    )

    await asyncio.gather(
        asyncio.wait_for(first_fired.wait(), timeout=1),
        asyncio.wait_for(second_fired.wait(), timeout=1),
    )

    assert [event.channel_id for event in first_events] == ["ai-first"]
    assert [event.channel_id for event in second_events] == ["ai-second"]

    shared.release(first.pending_id)
    shared.release(second.pending_id)
    await first_kit.close()
    await second_kit.close()


async def test_rebuilt_channel_survives_its_predecessors_teardown() -> None:
    """A displaced channel must not strand the id its replacement now serves.

    A host that rebuilds the channel serving an agent — the same agent
    attached to a second room — registers the new object over the old one and
    closes the old one afterwards, once its in-flight turns are done. With a
    handler shared by both, that late close used to mark the id closed for
    good: the live channel lost its callback and the next rebuild raised.
    """
    from roomkit import RoomKit
    from roomkit.providers.ai.mock import MockAIProvider

    shared = HumanInputHandler()
    kit = RoomKit()

    def build() -> AIChannel:
        return AIChannel(
            "agent:a1",
            provider=MockAIProvider(),
            human_input_handler=HumanInputToolHandler(tool_names={"AskUser"}, handler=shared),
        )

    displaced = build()
    kit.register_channel(displaced)
    kit.unregister_channel("agent:a1")
    kit.register_channel(build())

    await displaced.close()

    assert "agent:a1" in shared._on_input_required_by_channel
    pending = await shared.create_detached("AskUser", {}, channel_id="agent:a1")
    assert pending.pending_id in shared.pending
    shared.release(pending.pending_id)

    kit.unregister_channel("agent:a1")
    kit.register_channel(build())

    await kit.close()


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
    ws.register_connection("c1", lambda _conn, ev: inbox.append(ev), room_id="deny-room")

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
