"""Structured-result delegation: a delegated worker hands its work back via the
``submit_result`` tool, and a deterministic completion guard re-prompts (then
fails on the worker's behalf) if it never does.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit import HookExecution, HookResult, HookTrigger
from roomkit.channels._tool_registry import ChannelRegistry
from roomkit.channels.agent import Agent
from roomkit.core.event_router import BroadcastResult
from roomkit.core.framework import RoomKit
from roomkit.core.mixins._child_execution import _scan_for_submitted_result
from roomkit.core.mixins.delegation import _run_with_structured_result
from roomkit.models.channel import ChannelOutput
from roomkit.models.enums import ChannelType, EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent, ToolCallContent
from roomkit.models.room import Room
from roomkit.models.steering import Cancel, InjectMessage
from roomkit.models.tool_call import ToolCallEvent, ToolRoundEvent
from roomkit.orchestration.result import is_submit_result, normalize_result, orchestration_fail
from roomkit.orchestration.strategies.supervisor.prompts import SUBMIT_VERDICT
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.context import current_tool_room_id
from roomkit.turns._registry_tools import OwnerToolStore
from tests.tool_room import owner_tools_in, room_tool_names


def _text_event(body: str) -> RoomEvent:
    return RoomEvent(
        room_id="parent::task-1",
        source=EventSource(channel_id="agent:w1", channel_type=ChannelType.AI),
        type=EventType.MESSAGE,
        content=TextContent(body=body),
    )


def _make_kit(
    agent_id: str,
    submit_on_attempt: int | None,
    payload: dict[str, Any] | None = None,
    tool_name: str = "submit_result",
):
    """Mock kit whose broadcast simulates the worker calling *tool_name* on the
    given 1-based attempt (None = never). Returns (kit, channel, counter), the
    counter holding the attempts and the message each one received."""
    kit = MagicMock()
    kit.get_room = AsyncMock(
        return_value=Room(id="parent::task-1", metadata={"task_agent_id": agent_id})
    )
    kit.store.list_bindings = AsyncMock(return_value=[])
    # The child room's trace: the worker's served call lands there, as the
    # loop stores it.
    trace: list[RoomEvent] = []
    kit.store.list_events = AsyncMock(side_effect=lambda *_a, **_k: list(trace))
    kit.store.get_private_state = AsyncMock(return_value=None)
    kit.store.add_event_auto_index = AsyncMock(side_effect=lambda _rid, ev: ev)
    kit.store.commit_event = AsyncMock(side_effect=lambda _rid, ev: ev)
    kit._commit_indexed = AsyncMock(side_effect=lambda _rid, ev: ev)
    kit._commit_blocked_events = AsyncMock()
    kit._persist_side_effects = AsyncMock()
    kit._report_intelligence_errors = AsyncMock()

    channel = SimpleNamespace(_registry=ChannelRegistry(agent_id, list), role="Researcher")
    kit.channels = {agent_id: channel}
    # The kit keeps the result tool it sets up for the agent (RFC §10.6.2).
    kit._owner_tools = OwnerToolStore(kit.channels.get)
    counter: dict[str, Any] = {"n": 0, "messages": []}

    async def _broadcast(event, _binding, _context):
        counter["n"] += 1
        counter["messages"].append(getattr(event.content, "body", ""))
        if submit_on_attempt is not None and counter["n"] == submit_on_attempt:
            # The agent's tool loop in the child room serves the result tool,
            # given to its turn by the kit.
            with owner_tools_in(kit, agent_id, "parent::task-1"):
                entry = channel._registry.lookup(tool_name, "parent::task-1")
            arguments = payload or {"status": "completed", "summary": "done", "data": {"x": 1}}
            await entry.serve(arguments)
            trace.append(
                _tool_call_event(tool_name, arguments, channel_id=agent_id, outcome="served")
            )
        out = ChannelOutput(responded=True, response_events=[_text_event("raw text")])
        return BroadcastResult(outputs={agent_id: out})

    kit._get_router = MagicMock(
        return_value=SimpleNamespace(broadcast=AsyncMock(side_effect=_broadcast))
    )
    return kit, channel, counter


class TestStructuredResultGuard:
    async def test_returns_structured_payload_when_worker_submits(self) -> None:
        kit, _channel, counter = _make_kit("agent:w1", submit_on_attempt=1)
        out = await _run_with_structured_result(
            kit, "parent::task-1", "do it", max_result_retries=3
        )
        payload = json.loads(out)
        assert payload["status"] == "completed"
        assert payload["data"] == {"x": 1}
        assert counter["n"] == 1  # no retries needed
        # The tool was declared in the child room only, and removed afterwards.
        assert not room_tool_names(kit, "agent:w1", "parent::task-1")

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
    kit.store.get_private_state = AsyncMock(return_value=None)
    kit.store.add_event_auto_index = AsyncMock(side_effect=lambda _rid, ev: ev)
    kit.store.commit_event = AsyncMock(side_effect=lambda _rid, ev: ev)
    kit._commit_indexed = AsyncMock(side_effect=lambda _rid, ev: ev)
    kit._commit_blocked_events = AsyncMock()
    kit._persist_side_effects = AsyncMock()
    kit._report_intelligence_errors = AsyncMock()
    channel = SimpleNamespace(_registry=ChannelRegistry("agent:w1", list), role="Researcher")
    kit.channels = {"agent:w1": channel}
    kit._owner_tools = OwnerToolStore(kit.channels.get)

    async def _broadcast(_event, _binding, _context):
        # The gateway handled submit_result; tool_handler is not called here.
        out = ChannelOutput(responded=True, response_events=[_text_event("done")])
        return BroadcastResult(outputs={"agent:w1": out})

    kit._get_router = MagicMock(
        return_value=SimpleNamespace(broadcast=AsyncMock(side_effect=_broadcast))
    )
    return kit


def _tool_call_event(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    channel_id: str = "agent:w1",
    outcome: Any = None,
) -> RoomEvent:
    return RoomEvent(
        room_id="parent::task-1",
        source=EventSource(channel_id=channel_id, channel_type=ChannelType.AI),
        type=EventType.TOOL_CALL_END,
        content=ToolCallContent(
            tool_name=tool_name,
            tool_id="tc-1",
            arguments=arguments,
            status="failed" if outcome not in (None, "served") else "completed",
            outcome=outcome,
        ),
    )


class TestStructuredResultViaTrace:
    """The claude_code path: submit_result arrives as a persisted tool call
    (prefixed by the gateway), captured by scanning the trace — not via the
    wrapped tool_handler."""

    async def test_captures_prefixed_gateway_call_from_trace(self) -> None:
        events = [
            _tool_call_event(
                "mcp__gateway__submit_result",
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
        events = [_tool_call_event("mcp__gateway__web_search", {"q": "x"})]
        kit = _make_cc_kit(events)
        out = await _run_with_structured_result(
            kit, "parent::task-1", "do it", max_result_retries=1
        )
        payload = json.loads(out)
        assert payload["status"] == "failed"
        assert payload["by"] == "orchestration"


class TestOnlyTheWorkersServedResultCounts:
    """A refused result call, or one another channel made, is no result (RMK-396)."""

    @pytest.mark.parametrize(
        ("channel_id", "outcome"),
        [("agent:w1", "refused"), ("agent:w1", "failed"), ("agent:other", "served")],
    )
    async def test_the_scan_ignores_it(self, channel_id: str, outcome: str) -> None:
        submitted = {"status": "completed", "summary": "not mine", "data": {}}
        kit = _make_cc_kit(
            [_tool_call_event("submit_result", submitted, channel_id=channel_id, outcome=outcome)]
        )

        assert await _scan_for_submitted_result(kit, "parent::task-1", "agent:w1") is None

    async def test_the_scan_takes_the_workers_served_call(self) -> None:
        submitted = {"status": "completed", "summary": "mine", "data": {}}
        kit = _make_cc_kit([_tool_call_event("submit_result", submitted, outcome="served")])

        found = await _scan_for_submitted_result(kit, "parent::task-1", "agent:w1")

        assert found is not None and found["summary"] == "mine"


class TestResultHelpers:
    def test_is_submit_result_matches_bare_and_prefixed(self) -> None:
        assert is_submit_result("submit_result")
        assert is_submit_result("mcp__gateway__submit_result")
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


class TestAnotherResultTool:
    """The guard forces whichever tool it is given; the supervised flow's verdict
    goes through ``submit_verdict`` this way (RMK-246)."""

    async def test_the_given_tool_is_injected_and_its_call_captured(self) -> None:
        kit, _channel, _counter = _make_kit(
            "agent:boss",
            submit_on_attempt=1,
            payload={"approved": True, "feedback": "", "next_task": "write it up"},
            tool_name="submit_verdict",
        )
        seen: list[list[str]] = []
        original_broadcast = kit._get_router.return_value.broadcast.side_effect

        async def _spy(event, binding, context):
            seen.append(room_tool_names(kit, "agent:boss", "parent::task-1"))
            return await original_broadcast(event, binding, context)

        kit._get_router.return_value.broadcast.side_effect = _spy

        out = await _run_with_structured_result(
            kit, "parent::task-1", "judge it", max_result_retries=2, result_tool=SUBMIT_VERDICT
        )

        assert json.loads(out) == {"approved": True, "feedback": "", "next_task": "write it up"}
        assert seen == [["submit_verdict"]]
        assert not room_tool_names(kit, "agent:boss", "parent::task-1")

    async def test_its_reminder_and_its_missing_payload_are_used(self) -> None:
        kit, _channel, counter = _make_kit(
            "agent:boss", submit_on_attempt=None, tool_name="submit_verdict"
        )

        out = await _run_with_structured_result(
            kit, "parent::task-1", "judge it", max_result_retries=1, result_tool=SUBMIT_VERDICT
        )

        verdict = json.loads(out)
        assert verdict["approved"] is False
        assert verdict["next_task"] is None
        assert counter["messages"][1] == SUBMIT_VERDICT.reminder

    async def test_a_prefixed_gateway_call_is_found_in_the_trace(self) -> None:
        kit = _make_cc_kit(
            [
                _tool_call_event(
                    "mcp__gateway__submit_verdict",
                    {"approved": False, "feedback": "add sources", "next_task": ""},
                )
            ]
        )

        found = await _scan_for_submitted_result(kit, "parent::task-1", "agent:w1", SUBMIT_VERDICT)

        assert found == {"approved": False, "feedback": "add sources", "next_task": None}


class _RolesSeen(list):
    """A provider's requests as the model was shown them: the roles of their
    messages when asked (the loop goes on appending to its context)."""

    def append(self, context: Any) -> None:
        super().append([message.role for message in context.messages])


class TestTheWorkersResultCall:
    """The result tool served in the loop is read once ON_TOOL_CALL judged the
    call: a call a hook blocked is no result, and its turn goes on (RMK-396).
    A served call ends the worker's turn: the model is asked nothing after its
    round, unless the round gave it something to read, a steered message or
    an AFTER_TOOL_ROUND one; a Cancel steered in then cancels it (RMK-357,
    RFC §23.3, §21.3)."""

    async def _delegate(
        self,
        *,
        block: bool = False,
        streaming: bool = True,
        steer: InjectMessage | Cancel | None = None,
        after_round: str | None = None,
    ) -> tuple[Any, list[list[str]]]:
        submitted = {"status": "completed", "summary": "the worker's result", "data": {}}
        calls = [AIToolCall(id="s1", name="submit_result", arguments=submitted)]
        if steer is not None:
            calls.append(AIToolCall(id="w1", name="wait_a_bit", arguments={}))
        provider = MockAIProvider(
            streaming=streaming,
            ai_responses=[AIResponse(content="", tool_calls=calls)]
            + [AIResponse(content="ok")] * 6,
        )
        provider.calls = _RolesSeen()
        worker: Agent

        async def wait_a_bit(name: str, arguments: dict[str, Any]) -> str:
            assert steer is not None
            worker.steer(steer, room_id=current_tool_room_id())
            return "waited"

        worker = Agent(
            "worker",
            provider=provider,
            tools=[AITool(name="wait_a_bit", description="Wait.", parameters={"type": "object"})],
            tool_handler=wait_a_bit,
        )
        kit = RoomKit()
        kit.register_channel(worker)
        await kit.create_room(room_id="parent")
        if block:

            @kit.hook(HookTrigger.ON_TOOL_CALL, HookExecution.SYNC)
            async def refuse(event: ToolCallEvent, ctx: Any) -> HookResult:
                return HookResult.block("not now")

        if after_round is not None:

            @kit.hook(HookTrigger.AFTER_TOOL_ROUND)
            async def remind(event: ToolRoundEvent, ctx: Any) -> HookResult:
                event.messages.append(after_round)
                return HookResult.allow()

        task = await kit.delegate(
            "parent", "worker", "do it", wait=True, require_structured_result=True
        )
        await kit.close()
        assert task.result is not None
        return task.result, list(provider.calls)

    async def test_a_blocked_call_is_no_result_and_its_turn_goes_on(self) -> None:
        result, calls = await self._delegate(block=True)
        assert "the worker's result" not in json.dumps(result.model_dump(mode="json"))
        # The same turn asks again, the blocked call's result in hand, rather
        # than the delegation reminding a turn that ended.
        assert calls[1][-1] == "tool"

    @pytest.mark.parametrize("streaming", [False, True], ids=["generates", "streams"])
    async def test_a_served_call_is_the_result_and_ends_the_turn(self, streaming: bool) -> None:
        result, calls = await self._delegate(streaming=streaming)
        assert json.loads(result.output or "{}")["summary"] == "the worker's result"
        assert len(calls) == 1

    @pytest.mark.parametrize("given", ["steered", "after_tool_round"])
    async def test_a_message_its_round_gives_the_model_is_read(self, given: str) -> None:
        if given == "steered":
            _, calls = await self._delegate(steer=InjectMessage(role="user", content="One more."))
        else:
            _, calls = await self._delegate(after_round="One more.")
        assert [roles[-1] for roles in calls] == ["user", "user"]

    async def test_a_cancel_steered_in_its_round_cancels_the_turn(self) -> None:
        result, calls = await self._delegate(steer=Cancel(reason="stop"))
        assert len(calls) == 1
        assert result.metadata["turns"]["worker"]["loop_end_reason"] == "cancelled"
