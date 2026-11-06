"""A tool handler that never answers costs its call, never the turn (RFC §21.6, RMK-366).

Every wait on a handler goes through one bound, on every path a call takes:
both text loops, a speech-to-speech channel's provider calls, the calls it
recovers from speech, its reasoning backend's calls, its skill scripts and a
pipeline agent's own tools, and a conference's provider calls. Past the bound
the handler is cancelled and the call fails as a raise (RFC §9.3). A tool that
waits on another agent or a person, or carries a timeout of its own, keeps its
own bound.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from roomkit import AIChannelTurnConfig, ConferenceRealtimeConfig, ToolTimeoutError
from roomkit.channels._tool_registry import orchestration_tool, schema_tool
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.orchestration.pipeline import ConversationPipeline, PipelineStage
from roomkit.orchestration.state import ConversationState, set_conversation_state
from roomkit.providers.ai.base import AITool
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.sandbox.executor import SandboxExecutor
from roomkit.sandbox.models import SandboxResult
from roomkit.skills.executor import ScriptExecutor
from roomkit.skills.models import ScriptResult
from roomkit.skills.registry import SkillRegistry
from roomkit.tools.human_input import HumanInputToolHandler
from roomkit.tools.timeout import ToolTimeouts, answer_within
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.realtime.reasoning import ReasoningBackend, ReasoningOutput, ReasoningRequest
from tests.conference.test_conference_realtime import ROOM, realtime_kit, until
from tests.test_tool_policy_exemptions import _calls, _tool_payload, _turn
from tests.tool_room import owner_tools_in

SLOW = {
    "name": "slow",
    "description": "Answers slowly",
    "parameters": {"type": "object", "properties": {}},
}
BOUND = 0.02
LONGER = 0.06  # three times BOUND: a call that is not cut answers after this


class _Hung:
    """A handler that takes *delay* seconds to answer, and records a cancel."""

    def __init__(self, delay: float = 60.0) -> None:
        self.delay = delay
        self.cancelled = False

    async def wait(self) -> None:
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise

    async def handler(self, name: str, arguments: dict[str, Any]) -> str:
        await self.wait()
        return '{"ok": true}'

    async def conference(self, room_id: str, name: str, arguments: dict[str, Any]) -> str:
        return await self.handler(name, arguments)


def _timed_out(body: str) -> bool:
    return "ToolTimeoutError" in json.loads(body)["error"]


# -- The bound itself -------------------------------------------------------


class TestToolTimeouts:
    def test_a_call_takes_the_default(self) -> None:
        assert ToolTimeouts(10.0).for_call("lookup") == 10.0

    def test_a_tool_bound_overrides_the_default(self) -> None:
        timeouts = ToolTimeouts(10.0, {"report": 120.0, "export": None})
        assert timeouts.for_call("report") == 120.0
        assert timeouts.for_call("export") is None

    def test_a_tool_that_waits_keeps_its_own_bound_unless_named(self) -> None:
        assert ToolTimeouts(10.0).for_call("delegate", waits=True) is None
        assert ToolTimeouts(10.0, {"delegate": 30.0}).for_call("delegate", waits=True) == 30.0

    def test_overridden_bounds_keep_the_default_and_every_name_not_given(self) -> None:
        timeouts = ToolTimeouts(10.0, {"report": 120.0, "export": None})
        turn = timeouts.overridden({"export": 30.0, "device": None})
        assert turn.for_call("lookup") == 10.0
        assert turn.for_call("report") == 120.0
        assert turn.for_call("export") == 30.0
        assert turn.for_call("device") is None

    @pytest.mark.parametrize("bounds", [(0.0, {}), (-1.0, {}), (10.0, {"slow": 0.0})])
    def test_a_bound_that_is_not_positive_is_refused(
        self, bounds: tuple[float, dict[str, float | None]]
    ) -> None:
        with pytest.raises(ValueError, match="must be positive or None"):
            ToolTimeouts(*bounds)


class TestAnswerWithin:
    async def test_an_expired_bound_cancels_the_handler_and_raises(self) -> None:
        hung = _Hung()

        with pytest.raises(ToolTimeoutError, match="'slow' did not answer within 0.02 s"):
            await answer_within(BOUND, "slow", hung.handler("slow", {}))

        assert hung.cancelled

    async def test_a_handler_timeout_is_its_own_failure(self) -> None:
        async def times_out() -> str:
            raise TimeoutError("upstream API")

        with pytest.raises(TimeoutError, match="upstream API") as raised:
            await answer_within(10.0, "slow", times_out())
        assert not isinstance(raised.value, ToolTimeoutError)

    async def test_no_bound_waits_for_the_answer(self) -> None:
        assert await answer_within(None, "slow", _Hung(delay=0.01).handler("slow", {})) == (
            '{"ok": true}'
        )


# -- Text: both generation loops -------------------------------------------


def _text_channel(provider: MockAIProvider, hung: _Hung, **kwargs: Any) -> AIChannel:
    return AIChannel(
        "ai1", provider=provider, tool_handler=hung.handler, tool_timeout_seconds=BOUND, **kwargs
    )


def _calling(name: str, streaming: bool, **arguments: Any) -> MockAIProvider:
    return MockAIProvider(ai_responses=_calls((name, arguments)), streaming=streaming)


async def test_a_hung_text_tool_fails_its_call_and_the_turn_goes_on(streaming: bool) -> None:
    provider, hung = _calling("slow", streaming), _Hung()

    await _turn(_text_channel(provider, hung), [SLOW])

    assert hung.cancelled
    assert "ToolTimeoutError" in _tool_payload(provider.calls[1], "slow")["error"]
    assert len(provider.calls) == 2  # the model answered after the failed call


async def test_a_text_tool_bound_lets_a_slow_tool_finish(streaming: bool) -> None:
    provider, hung = _calling("slow", streaming), _Hung(delay=LONGER)

    await _turn(_text_channel(provider, hung, tool_timeouts={"slow": None}), [SLOW])

    assert not hung.cancelled
    assert _tool_payload(provider.calls[1], "slow") == {"ok": True}


def _configured(
    provider: MockAIProvider, hung: _Hung, *turns: dict[str, float | None] | None
) -> AIChannel:
    """A channel bound at :data:`BOUND` whose config provider declares SLOW
    and its handler for each turn, with *turns*' bounds, one per turn."""
    given = iter(turns)

    async def resolve(binding: ChannelBinding, context: RoomContext) -> AIChannelTurnConfig:
        return AIChannelTurnConfig(
            tools=[AITool(**SLOW)], tool_handler=hung.handler, tool_timeouts=next(given)
        )

    return AIChannel("ai1", provider=provider, tool_timeout_seconds=BOUND, config_provider=resolve)


async def test_a_turns_configuration_bounds_the_tools_it_declares(streaming: bool) -> None:
    """A tool the turn declares may be one the channel was never built with:
    its bound travels with it, beside the handler that serves it."""
    provider, hung = _calling("slow", streaming), _Hung(delay=LONGER)

    await _turn(_configured(provider, hung, {"slow": None}))

    assert not hung.cancelled
    assert _tool_payload(provider.calls[1], "slow") == {"ok": True}


async def test_a_turns_bounds_are_never_the_next_turns(streaming: bool) -> None:
    """The bounds are the turn's: one that gives none reads the channel's."""
    provider = MockAIProvider(
        ai_responses=_calls(("slow", {})) + _calls(("slow", {})), streaming=streaming
    )
    hung = _Hung(delay=LONGER)
    channel = _configured(provider, hung, {"slow": None}, None)

    await _turn(channel)
    assert not hung.cancelled
    await _turn(channel)

    assert hung.cancelled
    assert "ToolTimeoutError" in _tool_payload(provider.calls[3], "slow")["error"]


async def test_a_turns_bound_that_is_not_positive_fails_the_turn() -> None:
    provider, hung = _calling("slow", False), _Hung()

    with pytest.raises(ValueError, match="must be positive or None"):
        await _turn(_configured(provider, hung, {"slow": 0.0}))

    assert provider.calls == []  # refused before the model is asked


async def test_a_human_input_tool_keeps_its_own_bound(streaming: bool) -> None:
    ask = AITool(name="ask_user", description="Ask the user", parameters={"type": "object"})
    human = HumanInputToolHandler({"ask_user"}, timeout=LONGER, tool_definitions=[ask])
    provider = _calling("ask_user", streaming)
    channel = _text_channel(provider, _Hung(), human_input_handler=human)

    started = time.monotonic()
    await _turn(channel)

    assert time.monotonic() - started >= LONGER  # its own timeout, not the channel's
    assert "ToolTimeoutError" not in _tool_payload(provider.calls[1], "ask_user")["error"]


async def test_a_waiting_orchestration_tool_keeps_its_own_bound(streaming: bool) -> None:
    provider = _calling("delegate_task", streaming)
    channel = _text_channel(provider, _Hung())
    worker = _Hung(delay=LONGER)

    async def delegate(arguments: dict[str, Any]) -> str:
        await worker.wait()
        return '{"delegated": true}'

    tool = schema_tool({"name": "delegate_task", "parameters": {"type": "object"}})
    # Set up by the room owner, which gives it to the channel's turn.
    kit = RoomKit()
    kit._owner_tools.of("ai1").register(
        orchestration_tool(tool, delegate, waits=True), room_id="r1", owner=object()
    )

    with owner_tools_in(kit, "ai1", "r1"):
        await _turn(channel)

    assert not worker.cancelled
    assert _tool_payload(provider.calls[1], "delegate_task") == {"delegated": True}


class _SlowSandbox(SandboxExecutor):
    """A sandbox whose commands take LONGER, under their own ``timeout``."""

    async def execute(
        self, command: str, arguments: dict[str, Any] | None = None
    ) -> SandboxResult:
        await asyncio.sleep(LONGER)
        return SandboxResult(exit_code=0, output="built")

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "sandbox_bash",
                "description": "Run a shell command in the sandbox.",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
            }
        ]


async def test_a_sandbox_command_keeps_its_own_timeout(streaming: bool) -> None:
    provider = _calling("sandbox_bash", streaming, command="make")
    channel = _text_channel(provider, _Hung(), sandbox=_SlowSandbox())

    await _turn(channel)

    assert "built" in json.dumps(_tool_payload(provider.calls[1], "sandbox_bash"))


# -- Speech-to-speech: every entry of a session ------------------------------


async def _realtime(
    hung: _Hung, **kwargs: Any
) -> tuple[RoomKit, MockRealtimeProvider, VoiceSession]:
    provider = MockRealtimeProvider(full_duplex="reasoning_backend" in kwargs)
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[SLOW],
        tool_handler=hung.handler,
        tool_timeout_seconds=BOUND,
        **kwargs,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    return kit, provider, await channel.start_session("r1", "u1", "ws")


async def test_a_hung_realtime_tool_fails_its_call() -> None:
    hung = _Hung()
    kit, provider, session = await _realtime(hung)

    await provider.simulate_tool_call(session, "c1", "slow", {})
    await until(lambda: bool(provider.tool_results))

    assert hung.cancelled
    assert _timed_out(provider.tool_results[0][2])
    await kit.close()


async def test_a_realtime_tool_bound_lets_a_slow_tool_finish() -> None:
    hung = _Hung(delay=LONGER)
    kit, provider, session = await _realtime(hung, tool_timeouts={"slow": None})

    await provider.simulate_tool_call(session, "c1", "slow", {})
    await until(lambda: bool(provider.tool_results))

    assert not hung.cancelled
    assert json.loads(provider.tool_results[0][2]) == {"ok": True}
    await kit.close()


async def test_a_hung_tool_recovered_from_speech_fails_its_call() -> None:
    hung = _Hung()
    kit, provider, session = await _realtime(hung)

    await provider.simulate_transcription(session, "call:slow{}", "assistant")
    await until(lambda: any("ToolTimeoutError" in t for _s, t, _r in provider.injected_texts))

    assert hung.cancelled
    await kit.close()


class _Backend(ReasoningBackend):
    """Calls the slow tool once, then answers."""

    def __init__(self) -> None:
        self.results: list[str] = []

    async def run(self, request: ReasoningRequest) -> AsyncIterator[ReasoningOutput]:
        assert request.execute_tool is not None
        self.results.append(await request.execute_tool("slow", {}))
        yield ReasoningOutput("done", is_final=True)


async def test_a_hung_tool_a_reasoning_backend_calls_fails_its_call() -> None:
    hung, backend = _Hung(), _Backend()
    kit, provider, session = await _realtime(hung, reasoning_backend=backend)

    await provider.simulate_delegation(session, "d1", "integrator")
    await until(lambda: bool(backend.results))

    assert hung.cancelled
    assert _timed_out(backend.results[0])
    await kit.close()


class _HungScripts(ScriptExecutor):
    def __init__(self) -> None:
        self.hung = _Hung()

    async def execute(
        self, skill: Any, script_name: str, arguments: dict[str, str] | None = None
    ) -> ScriptResult:
        await self.hung.wait()
        return ScriptResult(exit_code=0, stdout="ran")


def _skills(root: Path) -> SkillRegistry:
    skill = root / "tools"
    (skill / "scripts").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: tools\ndescription: Tools\n---\nUse it.")
    (skill / "scripts" / "run.sh").write_text("echo hi\n")
    registry = SkillRegistry()
    registry.discover(root)
    return registry


async def test_a_hung_realtime_skill_script_fails_its_call(tmp_path: Path) -> None:
    scripts = _HungScripts()
    kit, provider, session = await _realtime(
        _Hung(), skills=_skills(tmp_path), script_executor=scripts
    )

    await provider.simulate_tool_call(
        session, "c1", "run_skill_script", {"skill_name": "tools", "script_name": "run.sh"}
    )
    await until(lambda: bool(provider.tool_results))

    assert scripts.hung.cancelled
    assert _timed_out(provider.tool_results[0][2])
    await kit.close()


async def test_a_pipeline_agents_own_tool_is_bounded() -> None:
    hung, provider = _Hung(), MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rtv", provider=provider, transport=MockRealtimeTransport(), tool_timeout_seconds=BOUND
    )
    kit = RoomKit()
    kit.register_channel(channel)
    agent = Agent(
        "agent-a", role="A", voice="v", system_prompt="x", tools=[SLOW], tool_handler=hung.handler
    )
    pipeline = ConversationPipeline(stages=[PipelineStage(phase="a", agent_id="agent-a")])
    pipeline.install(kit, [agent], voice_channel_id="rtv")
    room = await kit.create_room()
    state = ConversationState(active_agent_id="agent-a", phase="a")
    await kit.store.update_room(set_conversation_state(room, state))
    await kit.attach_channel(room.id, "rtv")
    session = await channel.start_session(room.id, "u1", "ws")

    await provider.simulate_tool_call(session, "c1", "slow", {})
    await until(lambda: bool(provider.tool_results))

    assert hung.cancelled
    assert _timed_out(provider.tool_results[0][2])
    await kit.close()


# -- Conference ----------------------------------------------------------------


async def _conference(hung: _Hung, **bounds: Any) -> tuple[RoomKit, MockRealtimeProvider, Any]:
    provider = MockRealtimeProvider()
    kit, channel, _, _ = await realtime_kit(
        provider=provider,
        config=ConferenceRealtimeConfig(
            provider=provider, tools=[SLOW], tool_handler=hung.conference, **bounds
        ),
    )
    session = await channel._realtime.ensure_session(ROOM)  # noqa: SLF001
    assert session is not None
    return kit, provider, session


async def test_a_hung_conference_tool_fails_its_call() -> None:
    hung = _Hung()
    kit, provider, session = await _conference(hung, tool_timeout_seconds=BOUND)

    await provider.simulate_tool_call(session, "c1", "slow", {})
    await until(lambda: bool(provider.tool_results))

    assert hung.cancelled
    assert _timed_out(provider.tool_results[0][2])
    await kit.close()


async def test_a_conference_tool_bound_lets_a_slow_tool_finish() -> None:
    hung = _Hung(delay=LONGER)
    kit, provider, session = await _conference(
        hung, tool_timeout_seconds=BOUND, tool_timeouts={"slow": 1.0}
    )

    await provider.simulate_tool_call(session, "c1", "slow", {})
    await until(lambda: bool(provider.tool_results))

    assert not hung.cancelled
    assert json.loads(provider.tool_results[0][2]) == {"ok": True}
    await kit.close()


def test_a_conference_bound_that_is_not_positive_is_refused() -> None:
    with pytest.raises(ValueError, match="must be positive or None"):
        ConferenceRealtimeConfig(provider=MockRealtimeProvider(), tool_timeout_seconds=0)
