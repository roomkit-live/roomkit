"""Tool-call ephemeral events fire on the realtime bus alongside thinking.

Regression tests for the bug where ``_publish_tool_event`` existed on
``AIEventsMixin`` but had no call sites: subscribers saw the model's
reasoning stream live (THINKING_*) while tool calls were invisible until
a page reload. All three generation paths must publish TOOL_CALL_START /
TOOL_CALL_END: the streaming internal tool loop, the streaming external
handler path, and the non-streaming tool loop.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

from roomkit import HookExecution, HookResult, HookTrigger, RoomContext, ToolCallEvent
from roomkit.channels._ai_streaming import _ToolCallDeltaCoalescer
from roomkit.channels.ai import AIChannel
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.models.steering import Cancel
from roomkit.models.streaming import LoopEndMarker
from roomkit.providers.ai.base import (
    AIContext,
    AIResponse,
    AITool,
    AIToolCall,
    StreamDone,
    StreamToolCall,
    StreamToolCallDelta,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.realtime.base import EphemeralEvent, EphemeralEventType
from roomkit.tools.external import PolicyExternalToolHandler
from tests.conftest import make_event
from tests.test_framework import SimpleChannel

_TOOLS = [{"name": "search", "description": "Search"}]


async def _run_turn(kit: RoomKit, ai: AIChannel) -> list[EphemeralEvent]:
    """Wire a room, subscribe to its realtime bus, run one inbound turn."""
    sms = SimpleChannel("sms1")
    kit.register_channel(sms)
    kit.register_channel(ai)

    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel(
        "r1", "ai1", category=ChannelCategory.INTELLIGENCE, metadata={"tools": _TOOLS}
    )

    received: list[EphemeralEvent] = []

    async def on_event(ev: EphemeralEvent) -> None:
        received.append(ev)

    await kit.realtime.subscribe_to_room("r1", on_event)

    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )

    # InMemoryRealtime dispatches subscriber callbacks via background tasks;
    # yield once so they run before we inspect the list.
    await asyncio.sleep(0.05)
    return received


def _tool_events(
    received: list[EphemeralEvent],
) -> tuple[list[EphemeralEvent], list[EphemeralEvent]]:
    starts = [e for e in received if e.type == EphemeralEventType.TOOL_CALL_START]
    ends = [e for e in received if e.type == EphemeralEventType.TOOL_CALL_END]
    return starts, ends


async def test_streaming_tool_loop_publishes_tool_events() -> None:
    """Internal-handler streaming loop: one START + one END per round."""

    async def tool_handler(name: str, args: dict[str, Any]) -> str:
        return f"result of {name}"

    provider = MockAIProvider(
        streaming=True,
        ai_responses=[
            AIResponse(
                content="",
                thinking="round one reasoning",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="tc1", name="search", arguments={"q": "x"})],
            ),
            AIResponse(content="Done.", thinking="round two reasoning", finish_reason="stop"),
        ],
    )
    kit = RoomKit()
    ai = AIChannel("ai1", provider=provider, tool_handler=tool_handler, thinking_budget=4096)

    received = await _run_turn(kit, ai)
    starts, ends = _tool_events(received)

    assert len(starts) == 1
    assert starts[0].data["tool_calls"] == [
        {"id": "tc1", "name": "search", "arguments": {"q": "x"}}
    ]
    assert starts[0].data["round"] == 0
    assert starts[0].channel_id == "ai1"

    assert len(ends) == 1
    assert ends[0].data["tool_calls"] == [
        {"id": "tc1", "name": "search", "result": "result of search"}
    ]
    assert ends[0].data["round"] == 0
    assert isinstance(ends[0].data["duration_ms"], int)

    # The bug scenario: reasoning AND tool events both reach subscribers.
    assert any(e.type == EphemeralEventType.THINKING_START for e in received)

    await kit.close()


async def test_streaming_tool_end_result_preview_capped() -> None:
    """END payloads carry a bounded result preview (500 chars)."""

    async def tool_handler(name: str, args: dict[str, Any]) -> str:
        return "x" * 600

    provider = MockAIProvider(
        streaming=True,
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="tc1", name="search", arguments={})],
            ),
            AIResponse(content="Done.", finish_reason="stop"),
        ],
    )
    kit = RoomKit()
    ai = AIChannel("ai1", provider=provider, tool_handler=tool_handler)

    received = await _run_turn(kit, ai)
    _, ends = _tool_events(received)

    assert len(ends) == 1
    assert len(ends[0].data["tool_calls"][0]["result"]) == 500

    await kit.close()


async def test_non_streaming_tool_loop_publishes_tool_events() -> None:
    """Non-streaming tool loop publishes the same START/END pairs."""

    async def tool_handler(name: str, args: dict[str, Any]) -> str:
        return "42"

    provider = MockAIProvider(
        streaming=False,
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="tc1", name="calc", arguments={"n": 1})],
            ),
            AIResponse(content="The answer is 42.", finish_reason="stop"),
        ],
    )
    kit = RoomKit()
    ai = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=tool_handler,
        tools=[AITool(name="calc", description="Calculate", parameters={})],
    )

    received = await _run_turn(kit, ai)
    starts, ends = _tool_events(received)

    assert len(starts) == 1
    assert starts[0].data["tool_calls"][0]["name"] == "calc"
    assert starts[0].data["round"] == 0
    assert len(ends) == 1
    assert ends[0].data["tool_calls"][0]["result"] == "42"

    await kit.close()


async def test_external_handler_streaming_publishes_tool_events() -> None:
    """External-handler path (provider executed the tool) publishes START/END."""
    provider = MockAIProvider(
        streaming=True,
        ai_responses=[
            AIResponse(
                content="I ran the tool.",
                finish_reason="stop",
                tool_calls=[
                    AIToolCall(
                        id="tc1",
                        name="Bash",
                        arguments={"cmd": "ls", "_result": "file.txt"},
                    )
                ],
            ),
        ],
    )
    kit = RoomKit()
    handler = PolicyExternalToolHandler()
    process_tool_call = AsyncMock(wraps=handler.process_tool_call)
    handler.process_tool_call = process_tool_call  # type: ignore[method-assign]
    ai = AIChannel("ai1", provider=provider, external_tool_handler=handler)

    received = await _run_turn(kit, ai)
    starts, ends = _tool_events(received)

    assert len(starts) == 1
    # `_result` is stripped from the arguments before publishing.
    assert starts[0].data["tool_calls"] == [
        {"id": "tc1", "name": "Bash", "arguments": {"cmd": "ls"}}
    ]
    assert len(ends) == 1
    assert ends[0].data["tool_calls"] == [{"id": "tc1", "name": "Bash", "result": "file.txt"}]
    # The provider embedded ``_result``, so the side effect already happened.
    # A retroactive BEFORE_TOOL_USE decision would be misleading and unsafe.
    process_tool_call.assert_not_awaited()

    await kit.close()


async def test_provider_executed_tool_never_fires_retroactive_before_hook() -> None:
    """An embedded result is observable, but can no longer be authorized."""
    provider = MockAIProvider(
        streaming=True,
        ai_responses=[
            AIResponse(
                content="done",
                finish_reason="stop",
                tool_calls=[
                    AIToolCall(
                        id="tc1",
                        name="Write",
                        arguments={"path": "/tmp/out", "_result": "written"},
                    )
                ],
            )
        ],
    )
    kit = RoomKit()
    ai = AIChannel("ai1", provider=provider)
    before_events: list[ToolCallEvent] = []
    observed_events: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC)
    async def before(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        before_events.append(event)
        return HookResult.block("too late")

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC)
    async def observe(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        observed_events.append(event)
        return HookResult.allow()

    await _run_turn(kit, ai)

    assert before_events == []
    assert len(observed_events) == 1
    assert observed_events[0].arguments == {"path": "/tmp/out"}
    assert observed_events[0].result == "written"

    await kit.close()


async def test_no_streaming_target_error_fires_on_error() -> None:
    """A provider failure on the no-streaming-targets path (a PII-locked / edge
    agent whose stream send fn was withheld) must still fire ON_ERROR so the
    error reaches the ON_ERROR hooks that classify + surface it — not propagate
    raw and vanish. Mirrors the streaming branch's error contract."""
    from collections.abc import AsyncIterator

    from roomkit.core.hooks import HookRegistration
    from roomkit.models.enums import HookExecution, HookTrigger
    from roomkit.providers.ai.base import StreamEvent

    class _RaisingProvider(MockAIProvider):
        async def generate_structured_stream(
            self, context: AIContext
        ) -> AsyncIterator[StreamEvent]:
            raise RuntimeError("exceeds the available context size")
            yield  # pragma: no cover - keep this an async generator

    kit = RoomKit()
    ai = AIChannel("ai1", provider=_RaisingProvider(streaming=True))

    errors: list[RoomEvent] = []

    async def on_error(event: RoomEvent, _ctx: RoomContext) -> None:
        errors.append(event)

    kit.hook_engine.register(
        HookRegistration(
            trigger=HookTrigger.ON_ERROR,
            execution=HookExecution.ASYNC,
            fn=on_error,
            name="test_capture_error",
        )
    )

    # With the fix the turn completes (the else branch catches + surfaces);
    # without it, process_inbound would propagate the raw error and this raises.
    await _run_turn(kit, ai)

    assert len(errors) == 1
    meta = errors[0].metadata or {}
    assert meta.get("error_category") == "streaming"
    assert "exceeds the available context size" in str(meta.get("error", ""))

    await kit.close()


# --- TOOL_CALL_DELTA: the composition of a call's arguments ----------------
#
# A model calling a tool spends the whole composition of its arguments
# producing tokens the provider hands over fragment by fragment. Until the
# call is complete nothing reached the bus, so a five-kilobyte argument was
# eight minutes of "working" with no name, no size, and no way to tell a
# model still generating from one that had hung.


def _delta_events(received: list[EphemeralEvent]) -> list[EphemeralEvent]:
    return [e for e in received if e.type == EphemeralEventType.TOOL_CALL_DELTA]


async def test_composition_deltas_reach_the_bus_before_the_call_completes() -> None:
    """TOOL_CALL_DELTA lands before TOOL_CALL_START, and START/END are unchanged."""

    async def tool_handler(name: str, args: dict[str, Any]) -> str:
        return f"result of {name}"

    provider = MockAIProvider(
        streaming=True,
        tool_call_delta_chunks=4,
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="tc1", name="search", arguments={"q": "cats"})],
            ),
            AIResponse(content="Done.", finish_reason="stop"),
        ],
    )
    kit = RoomKit()
    # A zero-millisecond window publishes every fragment, so the assertions
    # below read the composition rather than one coalesced snapshot.
    ai = AIChannel("ai1", provider=provider, tool_handler=tool_handler, thinking_coalesce_ms=0)

    received = await _run_turn(kit, ai)
    deltas = _delta_events(received)
    starts, ends = _tool_events(received)

    assert deltas, "the composition of the arguments was never published"
    assert received.index(deltas[0]) < received.index(starts[0])
    assert deltas[0].data["round"] == 0
    assert deltas[0].channel_id == "ai1"

    # The name is there from the first frame; the size grows and never shrinks.
    assert all(d.data["tool_calls"][0]["name"] == "search" for d in deltas)
    sizes = [d.data["tool_calls"][0]["arguments_chars"] for d in deltas]
    assert sizes == sorted(sizes)
    assert sizes[-1] == len(json.dumps({"q": "cats"}))

    # START and END are exactly what they were before composition events existed.
    assert len(starts) == 1
    assert starts[0].data["tool_calls"] == [
        {"id": "tc1", "name": "search", "arguments": {"q": "cats"}}
    ]
    assert len(ends) == 1
    assert ends[0].data["tool_calls"] == [
        {"id": "tc1", "name": "search", "result": "result of search"}
    ]

    await kit.close()


async def test_composition_delta_never_carries_the_argument_content() -> None:
    """The payload is a name and a size. Arguments can be megabytes, or personal."""
    secret = "ROSEBUD-" + "x" * 4000

    async def tool_handler(name: str, args: dict[str, Any]) -> str:
        return "ok"

    provider = MockAIProvider(
        streaming=True,
        tool_call_delta_chunks=8,
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="tc1", name="publish", arguments={"svg": secret})],
            ),
            AIResponse(content="Done.", finish_reason="stop"),
        ],
    )
    kit = RoomKit()
    ai = AIChannel("ai1", provider=provider, tool_handler=tool_handler, thinking_coalesce_ms=0)

    received = await _run_turn(kit, ai)
    deltas = _delta_events(received)

    assert deltas
    for delta in deltas:
        payload = json.dumps(delta.data)
        assert "ROSEBUD" not in payload
        assert set(delta.data["tool_calls"][0]) == {"id", "name", "arguments_chars"}
    # The complete arguments still arrive once, at the end, where they belong.
    starts, _ = _tool_events(received)
    assert starts[0].data["tool_calls"][0]["arguments"] == {"svg": secret}

    await kit.close()


async def test_first_fragment_publishes_without_waiting_for_the_window() -> None:
    """The tool's name is the signal — holding it for a window would defeat the point."""

    async def tool_handler(name: str, args: dict[str, Any]) -> str:
        return "ok"

    provider = MockAIProvider(
        streaming=True,
        tool_call_delta_chunks=6,
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="tc1", name="search", arguments={"q": "cats"})],
            ),
            AIResponse(content="Done.", finish_reason="stop"),
        ],
    )
    kit = RoomKit()
    # A window nothing can cross: only the immediate first-fragment publish fires.
    ai = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=tool_handler,
        thinking_coalesce_ms=1e9,
        thinking_coalesce_chars=100_000,
    )

    received = await _run_turn(kit, ai)
    deltas = _delta_events(received)

    assert len(deltas) == 1
    assert deltas[0].data["tool_calls"][0]["name"] == "search"

    await kit.close()


async def test_reasoning_window_closes_when_the_model_starts_composing() -> None:
    """THINKING_END fires on the first fragment, as it does on the first text delta.

    A round that reasons and then calls a tool with no text left THINKING_START
    open for the whole composition: the UI showed "thinking" while the model was
    already producing.
    """

    async def tool_handler(name: str, args: dict[str, Any]) -> str:
        return "ok"

    provider = MockAIProvider(
        streaming=True,
        tool_call_delta_chunks=4,
        ai_responses=[
            AIResponse(
                content="",
                thinking="I should search for that",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="tc1", name="search", arguments={"q": "cats"})],
            ),
            AIResponse(content="Done.", finish_reason="stop"),
        ],
    )
    kit = RoomKit()
    ai = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=tool_handler,
        thinking_budget=4096,
        thinking_coalesce_ms=0,
    )

    received = await _run_turn(kit, ai)
    deltas = _delta_events(received)
    thinking_ends = [e for e in received if e.type == EphemeralEventType.THINKING_END]

    assert deltas and thinking_ends
    assert received.index(thinking_ends[0]) < received.index(deltas[0])
    assert thinking_ends[0].data["thinking"] == "I should search for that"

    await kit.close()


async def test_cancelling_mid_composition_ends_the_loop_at_the_next_fragment() -> None:
    """Cancellation no longer waits out the composition.

    ``cancel_event`` is checked between two events of the stream. While a
    provider accumulated a call's arguments it yielded nothing, so a cancel
    landed only once the complete call arrived — minutes, for a large argument.
    """
    channel: dict[str, AIChannel] = {}

    class _ComposingProvider(MockAIProvider):
        """Composes 50 fragments, cancelling the loop after the third."""

        def __init__(self) -> None:
            super().__init__(streaming=True)
            self.pulled = 0

        async def generate_structured_stream(self, context: AIContext) -> Any:
            for _ in range(50):
                self.pulled += 1
                yield StreamToolCallDelta(id="tc1", name="publish", arguments_delta="x" * 64)
                if self.pulled == 3:
                    channel["ai"].steer(Cancel())
            yield StreamToolCall(id="tc1", name="publish", arguments={"svg": "x" * 3200})
            yield StreamDone(finish_reason="tool_calls")

    async def tool_handler(name: str, args: dict[str, Any]) -> str:
        return "ok"

    provider = _ComposingProvider()
    kit = RoomKit()
    ai = AIChannel("ai1", provider=provider, tool_handler=tool_handler)
    channel["ai"] = ai

    received = await _run_turn(kit, ai)
    starts, _ = _tool_events(received)

    # The loop took one more fragment, saw the cancel, and returned.
    assert provider.pulled == 4
    assert starts == []

    await kit.close()


async def test_cancelling_mid_composition_names_its_exit() -> None:
    """The loop says why it stopped — any other exit looks the same from outside.

    ``process_inbound`` consumes the marker stream internally, so this drives
    ``on_event`` directly to read what the loop said on its way out.
    """
    holder: dict[str, AIChannel] = {}

    class _ComposingProvider(MockAIProvider):
        def __init__(self) -> None:
            super().__init__(streaming=True)
            self.pulled = 0

        async def generate_structured_stream(self, context: AIContext) -> Any:
            for _ in range(50):
                self.pulled += 1
                yield StreamToolCallDelta(id="tc1", name="publish", arguments_delta="x" * 64)
                if self.pulled == 3:
                    holder["ai"].steer(Cancel())
            yield StreamToolCall(id="tc1", name="publish", arguments={"svg": "x" * 3200})
            yield StreamDone(finish_reason="tool_calls")

    async def tool_handler(name: str, args: dict[str, Any]) -> str:
        return "ok"

    provider = _ComposingProvider()
    ai = AIChannel("ai1", provider=provider, tool_handler=tool_handler)
    holder["ai"] = ai

    output = await ai.on_event(
        make_event(room_id="r1", body="go", channel_id="sms1"),
        ChannelBinding(
            channel_id="ai1",
            room_id="r1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
            metadata={"tools": _TOOLS},
        ),
        RoomContext(room=Room(id="r1")),
    )
    reasons = []
    if output.response_stream is not None:
        async for item in output.response_stream:
            if isinstance(item, LoopEndMarker):
                reasons.append(item.reason)

    assert reasons == ["cancelled"]
    assert provider.pulled == 4


# --- the coalescer itself --------------------------------------------------


async def test_composition_coalescer_publishes_the_first_fragment_at_once() -> None:
    published: list[tuple[Any, ...]] = []

    async def publish(*args: Any, **kwargs: Any) -> None:
        published.append(args)

    coalescer = _ToolCallDeltaCoalescer(publish, "r1", 0, flush_ms=1e9, flush_chars=100_000)
    await coalescer.add("tc1", "search", 10)

    assert len(published) == 1
    assert published[0][0] == EphemeralEventType.TOOL_CALL_DELTA
    assert published[0][2] == [{"id": "tc1", "name": "search", "arguments_chars": 10}]


async def test_composition_coalescer_accumulates_and_batches_by_size() -> None:
    published: list[list[dict[str, Any]]] = []

    async def publish(_type: Any, _room: Any, calls: list[dict[str, Any]], _round: int) -> None:
        published.append(calls)

    coalescer = _ToolCallDeltaCoalescer(publish, "r1", 0, flush_ms=1e9, flush_chars=100)
    for _ in range(10):
        await coalescer.add("tc1", "search", 30)

    # One publish for the first fragment, then one per 100 accumulated chars.
    # The last 30 characters stay unpublished on purpose: a residual under the
    # threshold loses nothing, since TOOL_CALL_START carries the whole call.
    assert [p[0]["arguments_chars"] for p in published] == [30, 150, 270]


async def test_composition_coalescer_carries_every_call_in_flight() -> None:
    """Providers interleave parallel calls; a frame is the round's snapshot."""
    published: list[list[dict[str, Any]]] = []

    async def publish(_type: Any, _room: Any, calls: list[dict[str, Any]], _round: int) -> None:
        published.append(calls)

    coalescer = _ToolCallDeltaCoalescer(publish, "r1", 0, flush_ms=0, flush_chars=1)
    await coalescer.add("tc1", "search", 10)
    await coalescer.add("tc2", "fetch", 20)
    await coalescer.add("tc1", "search", 5)

    assert published[-1] == [
        {"id": "tc1", "name": "search", "arguments_chars": 15},
        {"id": "tc2", "name": "fetch", "arguments_chars": 20},
    ]


async def test_composition_coalescer_flush_without_a_call_is_a_noop() -> None:
    published: list[Any] = []

    async def publish(*args: Any, **kwargs: Any) -> None:
        published.append(args)

    coalescer = _ToolCallDeltaCoalescer(publish, "r1", 0, flush_ms=1e9, flush_chars=16)
    await coalescer.flush()

    assert published == []
