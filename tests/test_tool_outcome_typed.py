"""A handler declines a call with a typed signal, read the same everywhere (RFC §21.4, RMK-305).

``UnservedToolCallError`` is how a handler says a tool is not its own; the
earlier ``{"error": "Unknown tool: ..."}`` envelope, as text or as a mapping,
is read as the same signal by one adapter. A composition hands a declined call
to its next handler, and every channel, a conference and the calls a realtime
session recovers from speech included, reads the call as served by nothing.
Every part the model reads is built from a typed outcome, so a cancelled or
failed call carries ``is_error`` on every path.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

import pytest

from roomkit import (
    ConferenceRealtimeConfig,
    HookExecution,
    HookResult,
    HookTrigger,
    RoomKit,
    ToolCallEvent,
)
from roomkit.channels._dangling_recovery import _build_patched_list
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.core.exceptions import UnservedToolCallError
from roomkit.models.context import RoomContext
from roomkit.providers.ai.base import AIMessage, AIToolCallPart
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.sandbox.executor import SandboxExecutor
from roomkit.sandbox.models import SandboxResult
from roomkit.tools._outcome import OutcomeKind, ToolOutcome
from roomkit.tools.compose import compose_tool_handlers
from roomkit.tools.result import declined_answer
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.realtime.reasoning import ReasoningBackend, ReasoningOutput, ReasoningRequest
from tests.conference.test_conference_realtime import ROOM, realtime_kit, until
from tests.test_tool_policy_exemptions import _calls, _tool_payload, _turn

LOOKUP = {
    "name": "lookup",
    "description": "Looks something up",
    "parameters": {"type": "object", "properties": {}},
}


async def _declines(name: str, arguments: dict[str, Any]) -> str:
    raise UnservedToolCallError(f"{name} is not mine")


async def _envelope_as_mapping(name: str, arguments: dict[str, Any]) -> Any:
    return {"error": f"Unknown tool: {name}"}


async def _serves(name: str, arguments: dict[str, Any]) -> str:
    return '{"served_by": "second"}'


# -- The adapter and the composition --------------------------------------------


class TestDeclining:
    @pytest.mark.parametrize(
        "answer", ['{"error": "Unknown tool: x"}', {"error": "unknown tool x"}]
    )
    def test_the_old_envelope_reads_as_the_typed_signal(self, answer: Any) -> None:
        with pytest.raises(UnservedToolCallError):
            declined_answer(answer, "x")

    def test_any_other_answer_passes(self) -> None:
        assert declined_answer('{"error": "quota exceeded"}', "x") == '{"error": "quota exceeded"}'

    @pytest.mark.parametrize("first", [_declines, _envelope_as_mapping], ids=["raises", "mapping"])
    async def test_a_composition_hands_a_declined_call_to_the_next_handler(
        self, first: Any
    ) -> None:
        composed = compose_tool_handlers(first, _serves)

        assert await composed("lookup", {}) == '{"served_by": "second"}'

    async def test_the_last_handlers_decline_is_the_compositions(self) -> None:
        composed = compose_tool_handlers(_envelope_as_mapping, _declines)

        with pytest.raises(UnservedToolCallError):
            await composed("lookup", {})


# -- Every path reads a declined call as served by nothing -----------------------


async def _conference(provider: MockRealtimeProvider, handler: Any = None) -> tuple[RoomKit, Any]:
    async def conference_handler(room_id: str, name: str, arguments: dict[str, Any]) -> Any:
        return await (handler or _declines)(name, arguments)

    kit, channel, _, _ = await realtime_kit(
        provider=provider,
        config=ConferenceRealtimeConfig(
            provider=provider, tools=[LOOKUP], tool_handler=conference_handler
        ),
    )
    session = await channel._realtime.ensure_session(ROOM)  # noqa: SLF001
    assert session is not None
    return kit, session


@pytest.mark.parametrize("handler", [_declines, _envelope_as_mapping], ids=["raises", "envelope"])
async def test_a_conference_reports_a_declined_call_as_failed(handler: Any) -> None:
    provider = MockRealtimeProvider()
    kit, session = await _conference(provider, handler)
    observed: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, ctx: RoomContext) -> None:
        observed.append(event)

    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await until(lambda: bool(provider.tool_results) and bool(observed))

    assert "No handler for tool lookup" in json.loads(provider.tool_results[0][2])["error"]
    assert [(e.name, e.is_error) for e in observed] == [("lookup", True)]
    await kit.close()


async def test_a_conference_hook_may_serve_a_call_its_handler_declined() -> None:
    provider = MockRealtimeProvider()
    kit, session = await _conference(provider)

    @kit.hook(HookTrigger.ON_TOOL_CALL, name="fallback")
    async def fallback(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        if event.result is None:
            return HookResult.modify(replace(event, result='{"served_by": "hook"}'))
        return HookResult.allow()

    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await until(lambda: bool(provider.tool_results))

    assert json.loads(provider.tool_results[0][2]) == {"served_by": "hook"}
    await kit.close()


@pytest.mark.parametrize("handler", [_declines, _envelope_as_mapping], ids=["raises", "envelope"])
async def test_a_call_recovered_from_speech_that_its_handler_declined_failed(
    handler: Any,
) -> None:
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[LOOKUP],
        tool_handler=handler,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    session = await channel.start_session("r1", "u1", "ws")

    await provider.simulate_transcription(session, "call:lookup{}", "assistant")
    await until(lambda: bool(provider.injected_texts))

    injected = provider.injected_texts[0][1]
    assert "failed" in injected
    assert "No handler for tool lookup" in injected
    await kit.close()


async def _realtime(handler: Any, **kwargs: Any) -> tuple[RoomKit, MockRealtimeProvider, Any]:
    provider = MockRealtimeProvider(full_duplex="reasoning_backend" in kwargs)
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[LOOKUP],
        tool_handler=handler,
        **kwargs,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    return kit, provider, await channel.start_session("r1", "u1", "ws")


async def test_a_realtime_call_its_handler_declined_failed() -> None:
    kit, provider, session = await _realtime(_declines)
    observed: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, ctx: RoomContext) -> None:
        observed.append(event)

    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await until(lambda: bool(provider.tool_results) and bool(observed))

    assert "No handler for tool lookup" in json.loads(provider.tool_results[0][2])["error"]
    assert [(e.name, e.is_error) for e in observed] == [("lookup", True)]
    await kit.close()


class _Backend(ReasoningBackend):
    """Calls ``lookup`` once, then answers."""

    def __init__(self) -> None:
        self.results: list[str] = []

    async def run(self, request: ReasoningRequest) -> AsyncIterator[ReasoningOutput]:
        assert request.execute_tool is not None
        self.results.append(await request.execute_tool("lookup", {}))
        yield ReasoningOutput("done", is_final=True)


async def test_a_reasoning_backend_call_its_handler_declined_failed() -> None:
    backend = _Backend()
    kit, provider, session = await _realtime(_declines, reasoning_backend=backend)

    await provider.simulate_delegation(session, "d1", "integrator")
    await until(lambda: bool(backend.results))

    assert "No handler for tool lookup" in json.loads(backend.results[0])["error"]
    await kit.close()


class _FailingCommand(SandboxExecutor):
    """A command whose error reads like the "not mine" envelope."""

    async def execute(
        self, command: str, arguments: dict[str, Any] | None = None
    ) -> SandboxResult:
        return SandboxResult(exit_code=1, error="Unknown tool: frobnicate")

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "sandbox_bash",
                "description": "Run a shell command in the sandbox.",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
            }
        ]


async def test_a_channel_tool_that_reads_like_the_envelope_is_served() -> None:
    """Only the host's answer may decline a call: a command's own error is its
    result (RFC §21.4)."""
    provider = MockAIProvider(ai_responses=_calls(("sandbox_bash", {"command": "frobnicate"})))
    channel = AIChannel("ai1", provider=provider, sandbox=_FailingCommand())
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")

    await _turn(channel)

    assert _tool_payload(provider.calls[1], "sandbox_bash")["error"] == "Unknown tool: frobnicate"
    await kit.close()


# -- Every part is built from a typed outcome -----------------------------------


@pytest.mark.parametrize(
    ("kind", "is_error"),
    [(OutcomeKind.SERVED, False)]
    + [(k, True) for k in OutcomeKind if k is not OutcomeKind.SERVED],
)
def test_a_part_carries_its_outcome(kind: OutcomeKind, is_error: bool) -> None:
    part = ToolOutcome(kind, "body").as_part("c1", "lookup", references=["x"])

    assert (part.is_error, part.result, part.references) == (is_error, "body", ["x"])
    assert part.outcome == kind.value


def test_a_dangling_call_patched_as_cancelled_reads_as_failed() -> None:
    call = AIToolCallPart(id="c1", name="lookup", arguments={})
    patched = _build_patched_list([AIMessage(role="assistant", content=[call])], set())

    result = patched[-1].content[0]
    assert result.tool_call_id == "c1"
    assert result.is_error


async def test_a_conference_without_a_handler_reads_a_call_as_unserved() -> None:
    """Nothing serves it: unserved, a failure the hooks may still serve, as on
    every channel (RFC §9.3, §21.4), never a refusal."""
    provider = MockRealtimeProvider()
    kit, channel, _, _ = await realtime_kit(
        provider=provider, config=ConferenceRealtimeConfig(provider=provider)
    )
    session = await channel._realtime.ensure_session(ROOM)  # noqa: SLF001
    observed: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, ctx: RoomContext) -> None:
        observed.append(event)

    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await until(lambda: bool(provider.tool_results) and bool(observed))
    await kit.close()

    assert json.loads(provider.tool_results[0][2]) == {"error": "No handler for tool lookup"}
    assert [(e.is_error, e.refused) for e in observed] == [(True, False)]


async def test_a_conference_hook_may_serve_a_call_no_handler_serves() -> None:
    provider = MockRealtimeProvider()
    kit, channel, _, _ = await realtime_kit(
        provider=provider, config=ConferenceRealtimeConfig(provider=provider)
    )
    session = await channel._realtime.ensure_session(ROOM)  # noqa: SLF001

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="serve")
    async def serve(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        return HookResult.modify(replace(event, result='{"found": true}'))

    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await until(lambda: bool(provider.tool_results))
    await kit.close()

    assert json.loads(provider.tool_results[0][2]) == {"found": True}
