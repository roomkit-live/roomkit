"""A live sandbox catalogue is consulted at each command boundary."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from roomkit import AIChannel, HookResult, HookTrigger, SandboxResult, ToolCallEvent
from roomkit.models.streaming import ToolCallEndMarker
from roomkit.models.tool_call import AIGenerationEvent
from roomkit.providers.ai import AIContext, AIResponse, AIToolCall, MockAIProvider
from tests.conftest import make_event
from tests.test_external_call_cut import _observe, _room, _say
from tests.test_sandbox_contract import RecordingSandbox, answer
from tests.test_tool_policy_exemptions import _binding
from tests.test_tool_search_text import _ctx


class ChangingSandbox(RecordingSandbox):
    def __init__(self, mutation: str) -> None:
        super().__init__()
        self.mutation = mutation
        self.force_failure = False

    def tool_definitions(self) -> list[dict[str, Any]]:
        if self.force_failure or (self.calls and self.mutation == "failed"):
            raise ConnectionError("catalogue changed")
        if self.calls:
            return []
        return super().tool_definitions()

    async def execute(
        self, command: str, arguments: dict[str, Any] | None = None
    ) -> SandboxResult:
        self.calls.append((command, arguments))
        return SandboxResult(exit_code=0, output="sandbox")


@pytest.mark.parametrize("mutation", ["removed", "failed"])
async def test_each_command_reads_the_live_catalogue(mutation: str, streaming: bool) -> None:
    sandbox = ChangingSandbox(mutation)
    host_calls: list[tuple[str, dict[str, Any]]] = []

    async def host(name: str, arguments: dict[str, Any]) -> str:
        host_calls.append((name, arguments))
        return "host"

    provider = MockAIProvider(
        streaming=streaming,
        ai_responses=[
            AIResponse(
                content="",
                tool_calls=[AIToolCall(id="one", name="sandbox_bash", arguments={"n": 1})],
            ),
            AIResponse(
                content="",
                tool_calls=[AIToolCall(id="two", name="sandbox_bash", arguments={"n": 2})],
            ),
            AIResponse(content="done"),
        ],
    )
    channel = AIChannel("ai1", provider=provider, sandbox=sandbox, tool_handler=host)
    if mutation == "failed":
        output = await channel.on_event(make_event(), _binding(), _ctx())
        assert output.response_stream is not None
        ends: list[ToolCallEndMarker] = []
        with pytest.raises(ConnectionError, match="^catalogue changed$"):
            async for item in output.response_stream:
                if isinstance(item, ToolCallEndMarker):
                    ends.append(item)
        await channel.close()
        assert sandbox.calls == [("bash", {"n": 1})]
        assert host_calls == []
        assert len(provider.calls) == 2
        assert [end.tool_id for end in ends] == ["one"]
        return
    run = await answer(channel)
    await channel.close()
    assert sandbox.calls == [("bash", {"n": 1})]
    assert host_calls == [("sandbox_bash", {"n": 2})]
    assert len(provider.calls) == 3
    assert len(run.calls) == 2
    assert not run.calls[0].failed
    assert run.calls[1].result == "host"
    assert not run.calls[1].failed
    parts = [
        part
        for message in provider.calls[2].messages
        for part in message.content
        if getattr(part, "tool_call_id", None) == "two"
    ]
    assert len(parts) == 1
    assert parts[0].outcome == "served"


class PendingCatalogue(ChangingSandbox):
    def __init__(self) -> None:
        super().__init__("failed")
        self.entered = False
        self.ended = False

    async def execute(
        self, command: str, arguments: dict[str, Any] | None = None
    ) -> SandboxResult:
        self.calls.append((command, arguments))
        self.entered = True
        try:
            await asyncio.Event().wait()
        finally:
            self.ended = True
        return SandboxResult(exit_code=0)


async def test_a_catalogue_gate_failure_joins_a_pending_sibling(streaming: bool) -> None:
    sandbox = PendingCatalogue()
    provider = MockAIProvider(
        streaming=streaming,
        ai_responses=[
            AIResponse(
                content="",
                tool_calls=[
                    AIToolCall(id="one", name="sandbox_bash", arguments={"n": 1}),
                    AIToolCall(id="two", name="sandbox_bash", arguments={"n": 2}),
                ],
            ),
        ],
    )
    channel = AIChannel("ai1", provider=provider, sandbox=sandbox)
    with pytest.raises(ConnectionError, match="^catalogue changed$"):
        await asyncio.wait_for(answer(channel), 1)
    assert sandbox.entered and sandbox.ended
    assert sandbox.calls == [("bash", {"n": 1})]
    assert len(provider.calls) == 1
    await channel.close()


class FailingOnGeneration(MockAIProvider):
    def __init__(self, sandbox: ChangingSandbox, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.sandbox = sandbox

    async def generate(self, context: AIContext) -> AIResponse:
        self.sandbox.force_failure = True
        return await super().generate(context)


async def test_withdrawal_refuses_before_a_live_catalogue_failure(streaming: bool) -> None:
    sandbox = ChangingSandbox("removed")
    provider = FailingOnGeneration(
        sandbox,
        streaming=streaming,
        ai_responses=[
            AIResponse(
                content="", tool_calls=[AIToolCall(id="one", name="sandbox_bash", arguments={})]
            ),
            AIResponse(content="done"),
        ],
    )
    channel = AIChannel("ai1", provider=provider, sandbox=sandbox)
    kit = await _room(channel)
    observed, judged = _observe(kit)

    @kit.hook(HookTrigger.BEFORE_AI_GENERATION)
    async def withdraw(event: AIGenerationEvent, _: Any) -> HookResult:
        event.ai_context = event.ai_context.model_copy(update={"tools": []})
        return HookResult.modify(event)

    await asyncio.wait_for(_say(kit), 2)
    await asyncio.wait_for(kit.close(), 2)
    assert sandbox.calls == []
    assert len(provider.calls) == 2
    assert judged == []
    assert len(observed) == 1
    assert observed[0].refused and not observed[0].cancelled
    assert (
        observed[0].result == '{"error": "Tool \'sandbox_bash\' is not available in this turn."}'
    )


async def test_a_catalogue_gate_failure_reports_both_announced_calls_once(streaming: bool) -> None:
    sandbox = PendingCatalogue()
    provider = MockAIProvider(
        streaming=streaming,
        ai_responses=[
            AIResponse(
                content="",
                tool_calls=[
                    AIToolCall(id="one", name="sandbox_bash", arguments={"n": 1}),
                    AIToolCall(id="two", name="sandbox_bash", arguments={"n": 2}),
                ],
            ),
        ],
    )
    kit = await _room(AIChannel("ai1", provider=provider, sandbox=sandbox))
    observed, judged = _observe(kit)

    @kit.hook(HookTrigger.BEFORE_TOOL_USE)
    async def rewrite(event: ToolCallEvent, _: Any) -> HookResult:
        return HookResult.modify(replace(event, arguments={"n": 101}))

    await asyncio.wait_for(_say(kit), 2)
    await asyncio.wait_for(kit.close(), 2)
    assert sandbox.entered and sandbox.ended
    assert sandbox.calls == [("bash", {"n": 101})]
    assert len(provider.calls) == 1
    assert judged == []
    assert sorted(event.tool_call_id for event in observed) == ["one", "two"]
    assert all(event.cancelled and event.is_error for event in observed)
    assert all(event.error_detail is None for event in observed)
    reports = {event.tool_call_id: event for event in observed}
    assert reports["one"].arguments == {"n": 101}
    assert reports["two"].arguments == {"n": 2}
    assert all(
        event.result
        == (
            '{"error": "Tool call cancelled", "tool": "sandbox_bash", '
            '"hint": "The turn ended before its result."}'
        )
        for event in observed
    )


class ReservedCatalogue(RecordingSandbox):
    def __init__(self) -> None:
        super().__init__()
        self.armed = False
        self.reads = 0

    def tool_definitions(self) -> list[dict[str, Any]]:
        if self.armed:
            self.reads += 1
            if self.reads > 2:
                raise ConnectionError("sandbox dispatch must not run")
        return super().tool_definitions()


class ReservedGeneration(MockAIProvider):
    def __init__(self, sandbox: ReservedCatalogue, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.sandbox = sandbox

    async def generate(self, context: AIContext) -> AIResponse:
        self.sandbox.armed = True
        return await super().generate(context)


async def test_a_reserved_tool_returns_before_sandbox_dispatch(streaming: bool) -> None:
    sandbox = ReservedCatalogue()
    provider = ReservedGeneration(
        sandbox,
        streaming=streaming,
        ai_responses=[
            AIResponse(
                content="",
                tool_calls=[
                    AIToolCall(id="search", name="find_tools", arguments={"query": "bash"})
                ],
            ),
            AIResponse(content="done"),
        ],
    )
    channel = AIChannel(
        "ai1", provider=provider, sandbox=sandbox, tool_search=True, max_tool_rounds=1
    )
    run = await answer(channel)
    await channel.close()
    assert sandbox.calls == []
    assert sandbox.reads == 2
    assert len(provider.calls) == 2
    assert len(run.calls) == 1
    assert not run.calls[0].failed
    assert "sandbox_bash" in str(run.calls[0].result)
