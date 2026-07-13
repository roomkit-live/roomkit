"""Structured-result delegation: a delegated worker hands its work back via the
``submit_result`` tool, and a deterministic completion guard re-prompts (then
fails on the worker's behalf) if it never does.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from roomkit.core.mixins.delegation import _run_with_structured_result
from roomkit.models.enums import ChannelType, EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent, ToolCallContent
from roomkit.models.room import Room
from roomkit.orchestration.result import is_submit_result, normalize_result, orchestration_fail


def _text_event(body: str) -> RoomEvent:
    return RoomEvent(
        room_id="parent::task-1",
        source=EventSource(channel_id="agent:w1", channel_type=ChannelType.AI),
        type=EventType.MESSAGE,
        content=TextContent(body=body),
    )


def _make_kit(agent_id: str, submit_on_attempt: int | None, payload: dict[str, Any] | None = None):
    """Mock kit whose broadcast simulates the worker calling submit_result on the
    given 1-based attempt (None = never). Returns (kit, channel, attempts-counter)."""
    kit = MagicMock()
    kit.get_room = AsyncMock(
        return_value=Room(id="parent::task-1", metadata={"task_agent_id": agent_id})
    )
    kit.store.list_bindings = AsyncMock(return_value=[])
    kit.store.list_events = AsyncMock(return_value=[])
    kit.store.add_event_auto_index = AsyncMock(side_effect=lambda _rid, ev: ev)
    kit.store.commit_event = AsyncMock(side_effect=lambda _rid, ev: ev)

    channel = SimpleNamespace(_injected_tools=[], tool_handler=None, role="Researcher")
    kit.channels = {agent_id: channel}
    counter = {"n": 0}

    async def _broadcast(_event, _binding, _context):
        counter["n"] += 1
        if submit_on_attempt is not None and counter["n"] == submit_on_attempt:
            await channel.tool_handler(
                "submit_result",
                payload or {"status": "completed", "summary": "done", "data": {"x": 1}},
            )
        out = SimpleNamespace(responded=True, response_events=[_text_event("raw text")])
        return SimpleNamespace(outputs={"w1": out}, streaming_responses=[])

    kit._get_router = MagicMock(
        return_value=SimpleNamespace(broadcast=AsyncMock(side_effect=_broadcast))
    )
    return kit, channel, counter


class TestStructuredResultGuard:
    async def test_returns_structured_payload_when_worker_submits(self) -> None:
        kit, channel, counter = _make_kit("agent:w1", submit_on_attempt=1)
        out = await _run_with_structured_result(
            kit, "parent::task-1", "do it", max_result_retries=3
        )
        payload = json.loads(out)
        assert payload["status"] == "completed"
        assert payload["data"] == {"x": 1}
        assert counter["n"] == 1  # no retries needed
        # The tool was cleaned up afterwards (no leak onto the shared channel).
        assert channel._injected_tools == []
        assert channel.tool_handler is None

    async def test_reprompts_until_worker_submits(self) -> None:
        kit, _channel, counter = _make_kit("agent:w1", submit_on_attempt=3)
        out = await _run_with_structured_result(
            kit, "parent::task-1", "do it", max_result_retries=3
        )
        payload = json.loads(out)
        assert payload["status"] == "completed"
        assert counter["n"] == 3  # two nudges, submitted on the third turn

    async def test_orchestration_fail_when_never_submits(self) -> None:
        kit, _channel, counter = _make_kit("agent:w1", submit_on_attempt=None)
        out = await _run_with_structured_result(
            kit, "parent::task-1", "do it", max_result_retries=2
        )
        payload = json.loads(out)
        assert payload["status"] == "failed"
        assert payload["by"] == "orchestration"  # mechanism-level failure, not the worker's
        assert payload["reason"] == "no_structured_result_after_3_attempts"
        assert payload["role"] == "Researcher"
        assert payload["last_output"] == "raw text"  # explanatory context
        assert counter["n"] == 3  # initial + 2 retries


def _make_cc_kit(events: list[RoomEvent]):
    """Mock kit for a claude_code worker: it calls the gateway-exposed
    submit_result (so the wrapped tool_handler is NEVER invoked), and the call is
    persisted as a TOOL_CALL event that the scan must pick up."""
    kit = MagicMock()
    kit.get_room = AsyncMock(
        return_value=Room(id="parent::task-1", metadata={"task_agent_id": "agent:w1"})
    )
    kit.store.list_bindings = AsyncMock(return_value=[])
    kit.store.list_events = AsyncMock(return_value=events)
    kit.store.add_event_auto_index = AsyncMock(side_effect=lambda _rid, ev: ev)
    kit.store.commit_event = AsyncMock(side_effect=lambda _rid, ev: ev)
    channel = SimpleNamespace(_injected_tools=[], tool_handler=None, role="Researcher")
    kit.channels = {"agent:w1": channel}

    async def _broadcast(_event, _binding, _context):
        # The gateway handled submit_result; tool_handler is not called here.
        out = SimpleNamespace(responded=True, response_events=[_text_event("done")])
        return SimpleNamespace(outputs={"w1": out}, streaming_responses=[])

    kit._get_router = MagicMock(
        return_value=SimpleNamespace(broadcast=AsyncMock(side_effect=_broadcast))
    )
    return kit


def _tool_call_event(tool_name: str, arguments: dict[str, Any]) -> RoomEvent:
    return RoomEvent(
        room_id="parent::task-1",
        source=EventSource(channel_id="agent:w1", channel_type=ChannelType.AI),
        type=EventType.TOOL_CALL_END,
        content=ToolCallContent(
            tool_name=tool_name, tool_id="tc-1", arguments=arguments, status="completed"
        ),
    )


class TestStructuredResultViaTrace:
    """The claude_code path: submit_result arrives as a persisted tool call
    (prefixed by the gateway), captured by scanning the trace — not via the
    wrapped tool_handler."""

    async def test_captures_prefixed_gateway_call_from_trace(self) -> None:
        events = [
            _tool_call_event(
                "mcp__luge-integrations__submit_result",
                {"status": "completed", "summary": "shipped", "data": {"y": 2}},
            )
        ]
        kit = _make_cc_kit(events)
        out = await _run_with_structured_result(
            kit, "parent::task-1", "do it", max_result_retries=3
        )
        payload = json.loads(out)
        assert payload["status"] == "completed"
        assert payload["summary"] == "shipped"
        assert payload["data"] == {"y": 2}

    async def test_fails_when_trace_has_no_submit_result(self) -> None:
        events = [_tool_call_event("mcp__luge-integrations__web_search", {"q": "x"})]
        kit = _make_cc_kit(events)
        out = await _run_with_structured_result(
            kit, "parent::task-1", "do it", max_result_retries=1
        )
        payload = json.loads(out)
        assert payload["status"] == "failed"
        assert payload["by"] == "orchestration"


class TestResultHelpers:
    def test_is_submit_result_matches_bare_and_prefixed(self) -> None:
        assert is_submit_result("submit_result")
        assert is_submit_result("mcp__luge-integrations__submit_result")
        assert not is_submit_result("web_search")
        assert not is_submit_result("submit_result_extra")

    def test_normalize_fills_defaults(self) -> None:
        r = normalize_result({"status": "completed", "summary": "s"})
        assert r == {
            "status": "completed",
            "summary": "s",
            "data": {},
            "deliverables": [],
            "reason": "",
        }

    def test_normalize_coerces_bad_types(self) -> None:
        r = normalize_result({"summary": "s", "data": "not-a-dict", "deliverables": "nope"})
        assert r["status"] == "completed"  # defaulted
        assert r["data"] == {}
        assert r["deliverables"] == []

    def test_orchestration_fail_shape(self) -> None:
        f = orchestration_fail(role="Analyst", last_output="partial", attempts=3)
        assert f["status"] == "failed"
        assert f["by"] == "orchestration"
        assert f["role"] == "Analyst"
        assert f["last_output"] == "partial"
        assert "3" in f["reason"]
