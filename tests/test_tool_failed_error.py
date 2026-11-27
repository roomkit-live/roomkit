"""The line between a refused and a failed call reads the same everywhere
(RMK-459, RFC §9.3).

A handler that ran and failed raises :class:`ToolFailedError` to hand its
words to the model: the call is failed, never refused, on the text and the
realtime doors alike, and an MCP tool whose result says ``isError`` is one. A
reasoning backend's relay keeps the outcome its loop gave a call and what
failed. A call whose observers' context will not build still emits its
``tool_call`` framework event.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from types import SimpleNamespace
from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit, ToolCallEvent, ToolFailedError
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelType
from roomkit.orchestration.tool_audit import ToolAuditEntry, ToolAuditor, audit_tool_handler
from roomkit.providers.ai.base import AIResponse, AITool
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.realtime import reasoning
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conftest import make_event
from tests.test_mcp_tool_provider import (
    SEARCH_TOOL,
    _make_provider_connected,
    _result,
    _text,
)
from tests.test_toolset_edges import _calling
from tests.tool_loop_modes import respond

WORDS = "The disk is full; try again in a minute."
PARAMS = {"type": "object", "properties": {"query": {"type": "string"}}}


async def _fails(name: str, arguments: dict[str, Any]) -> str:
    raise ToolFailedError(WORDS)


def _mcp_handler() -> Any:
    provider = _make_provider_connected(
        [SEARCH_TOOL],
        call_tool_side_effect=lambda name, args: _result([_text(WORDS)], is_error=True),
    )
    return provider.as_tool_handler()


def _observe(kit: RoomKit) -> list[ToolCallEvent]:
    seen: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, ctx: Any) -> None:
        seen.append(event)

    return seen


async def _text_call(handler: Any, streaming: bool) -> tuple[str, list[ToolCallEvent]]:
    provider = MockAIProvider(
        ai_responses=[_calling("search", query="x"), AIResponse(content="done")],
        streaming=streaming,
    )
    channel = AIChannel(
        "ai1",
        provider=provider,
        tools=[AITool(name="search", description="search", parameters=PARAMS)],
        tool_handler=handler,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ai1")
    seen = _observe(kit)
    binding = ChannelBinding(channel_id="ai1", room_id="r1", channel_type=ChannelType.AI)
    await respond(
        channel, make_event(room_id="r1", body="go"), binding, await kit._build_context("r1")
    )
    await asyncio.sleep(0.05)
    [read] = [
        str(part.result)
        for message in provider.calls[1].messages
        if message.role == "tool"
        for part in message.content
    ]
    await kit.close()
    return read, seen


async def _realtime_call(handler: Any) -> tuple[str, list[ToolCallEvent]]:
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[{"name": "search", "parameters": PARAMS}],
        tool_handler=handler,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    seen = _observe(kit)
    session = await channel.start_session("r1", "u", "ws")
    await provider.simulate_tool_call(session, "c1", "search", {"query": "x"})
    for _ in range(100):
        if provider.tool_results and seen:
            break
        await asyncio.sleep(0.01)
    await kit.close()
    return provider.tool_results[0][2], seen


def _failed(seen: list[ToolCallEvent]) -> list[tuple[bool, bool, str | None]]:
    return [(e.is_error, e.refused, e.error_detail) for e in seen]


async def test_a_handler_s_failure_is_read_in_its_words_on_the_text_door(
    streaming: bool,
) -> None:
    read, seen = await _text_call(_fails, streaming)

    assert read == WORDS
    assert _failed(seen) == [(True, False, WORDS)]


async def test_a_handler_s_failure_is_read_in_its_words_on_the_realtime_door() -> None:
    read, seen = await _realtime_call(_fails)

    assert read == WORDS
    assert _failed(seen) == [(True, False, WORDS)]


async def test_an_mcp_is_error_is_a_failure_on_every_door(streaming: bool) -> None:
    text_read, text_seen = await _text_call(_mcp_handler(), streaming)
    realtime_read, realtime_seen = await _realtime_call(_mcp_handler())

    assert text_read == realtime_read == WORDS
    assert _failed(text_seen) == _failed(realtime_seen) == [(True, False, WORDS)]


async def test_a_backend_s_relay_keeps_the_outcome_and_what_failed() -> None:
    relayed: list[dict[str, Any]] = []

    async def report_refusal(name: str, arguments: dict[str, Any], body: str, **kw: Any) -> None:
        relayed.append(kw)

    backend = reasoning.AgentReasoningBackend.__new__(reasoning.AgentReasoningBackend)
    token = reasoning._DELEGATION.set(SimpleNamespace(report_refusal=report_refusal))
    try:
        await backend._report_loop_refusal(
            ToolCallEvent(
                channel_id="ai",
                channel_type=ChannelType.AI,
                tool_call_id="c1",
                name="search",
                arguments={},
                result='{"error": "failed"}',
                room_id="r1",
                is_error=True,
                refused=False,
                error_detail="boom",
            )
        )
    finally:
        reasoning._DELEGATION.reset(token)

    assert relayed == [{"cancelled": False, "refused": False, "detail": "boom"}]


async def test_the_channel_reports_a_relayed_failure_as_a_failure() -> None:
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt", provider=provider, transport=MockRealtimeTransport(), tools=[]
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    seen = _observe(kit)
    session = await channel.start_session("r1", "u", "ws")

    body = json.dumps({"error": "failed"})
    await channel._report_backend_refusal(
        session, "d1", "search", {}, body, refused=False, detail="boom"
    )
    await asyncio.sleep(0.05)
    await kit.close()

    assert [(e.refused, e.cancelled, e.error_detail) for e in seen] == [(False, False, "boom")]


async def test_a_call_whose_context_will_not_build_still_emits_its_framework_event() -> None:
    kit = RoomKit()
    await kit.create_room(room_id="r1")
    _observe(kit)
    emitted: list[dict[str, Any]] = []

    @kit.on("tool_call")
    async def on_tool_call(event: Any) -> None:
        emitted.append(event.data)

    async def no_context(*args: Any, **kwargs: Any) -> None:
        return None

    kit._hook_context = no_context  # type: ignore[method-assign]
    event = ToolCallEvent(
        channel_id="ai1",
        channel_type=ChannelType.AI,
        tool_call_id="c1",
        name="search",
        arguments={},
        result='{"error": "refused"}',
        room_id="r1",
        is_error=True,
        refused=True,
    )

    await kit._observe_failed_tool_call(event, "ai1")
    await asyncio.sleep(0.05)
    await kit.close()

    assert [(d["tool_name"], d.get("refused")) for d in emitted] == [("search", True)]


async def test_a_reported_call_whose_context_will_not_build_still_emits_its_event() -> None:
    """The same for a call an external handler ran, reported (RFC §9.3)."""
    kit = RoomKit()
    await kit.create_room(room_id="r1")
    _observe(kit)
    emitted: list[dict[str, Any]] = []

    @kit.on("tool_call")
    async def on_tool_call(event: Any) -> None:
        emitted.append(event.data)

    async def no_context(*args: Any, **kwargs: Any) -> None:
        return None

    kit._hook_context = no_context  # type: ignore[method-assign]
    event = ToolCallEvent(
        channel_id="ai1",
        channel_type=ChannelType.AI,
        tool_call_id="c1",
        name="search",
        arguments={},
        result="found",
        room_id="r1",
    )

    await kit._report_tool_call(event, "ai1")
    await asyncio.sleep(0.05)
    await kit.close()

    assert [d["tool_name"] for d in emitted] == ["search"]


class _Auditor(ToolAuditor):
    def __init__(self) -> None:
        self._entries: list[ToolAuditEntry] = []

    def record(self, entry: ToolAuditEntry) -> None:
        self._entries.append(entry)

    def summary(self) -> str:
        return ""

    @property
    def entries(self) -> list[ToolAuditEntry]:
        return self._entries


async def test_an_mcp_is_error_audits_alike_through_both_wrappers() -> None:
    """``as_tool_handler`` raises it, ``call_tool`` renders it: ``failed``
    either way."""
    provider = _make_provider_connected(
        [SEARCH_TOOL],
        call_tool_side_effect=lambda name, args: _result([_text(WORDS)], is_error=True),
    )
    auditor = _Auditor()
    raised = audit_tool_handler(provider.as_tool_handler(), auditor, "agent")
    rendered = audit_tool_handler(
        lambda name, arguments: provider.call_tool(name, arguments), auditor, "agent"
    )

    with pytest.raises(ToolFailedError):
        await raised("search", {"query": "x"})
    await rendered("search", {"query": "x"})

    assert [e.status for e in auditor.entries] == ["failed", "failed"]


async def test_a_relayed_refusal_with_the_default_arguments_stays_a_refusal() -> None:
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt", provider=provider, transport=MockRealtimeTransport(), tools=[]
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    seen = _observe(kit)
    session = await channel.start_session("r1", "u", "ws")

    await channel._report_backend_refusal(session, "d1", "search", {}, '{"error": "cut"}')
    await asyncio.sleep(0.05)
    await kit.close()

    assert [(e.refused, e.cancelled, e.error_detail) for e in seen] == [(True, False, None)]


async def test_a_reporter_written_to_the_old_signature_still_hears_a_refusal() -> None:
    relayed: list[tuple[str, bool]] = []

    async def report_refusal(
        name: str, arguments: dict[str, Any], body: str, *, cancelled: bool = False
    ) -> None:
        relayed.append((name, cancelled))

    backend = reasoning.AgentReasoningBackend.__new__(reasoning.AgentReasoningBackend)
    token = reasoning._DELEGATION.set(SimpleNamespace(report_refusal=report_refusal))
    try:
        await backend._report_loop_refusal(
            ToolCallEvent(
                channel_id="ai",
                channel_type=ChannelType.AI,
                tool_call_id="c1",
                name="search",
                arguments={},
                result='{"error": "cut"}',
                room_id="r1",
                is_error=True,
                refused=True,
            )
        )
    finally:
        reasoning._DELEGATION.reset(token)

    assert relayed == [("search", False)]


async def test_a_call_already_reported_emits_no_second_event_when_its_context_fails() -> None:
    kit = RoomKit()
    await kit.create_room(room_id="r1")
    _observe(kit)
    emitted: list[Any] = []

    @kit.on("tool_call")
    async def on_tool_call(event: Any) -> None:
        emitted.append(event.data)

    async def no_context(*args: Any, **kwargs: Any) -> None:
        return None

    kit._hook_context = no_context  # type: ignore[method-assign]
    event = ToolCallEvent(
        channel_id="ai1",
        channel_type=ChannelType.AI,
        tool_call_id="c1",
        name="search",
        arguments={},
        result="found",
        room_id="r1",
        is_error=True,
        refused=True,
    )

    await kit._observe_failed_tool_call(event, "ai1", claim=lambda: False)
    await kit._report_tool_call(replace_refused(event), "ai1", claim=lambda: False)
    await asyncio.sleep(0.05)
    await kit.close()

    assert emitted == []


def replace_refused(event: ToolCallEvent) -> ToolCallEvent:
    """*event* as a served call an external handler ran."""
    return dataclasses.replace(event, is_error=False, refused=False)
