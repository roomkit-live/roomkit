"""Code a served skill would cause to run, approved by the host (RFC §24.7)."""

from __future__ import annotations

from typing import Any

import pytest

from roomkit.channels._served_execution import ServedExecutionGate
from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.sandbox.executor import SandboxExecutor
from roomkit.sandbox.models import SandboxResult
from roomkit.skills import SkillExecutionRequest, SkillRegistry
from tests.conftest import make_event
from tests.mcp_skill_fakes import FakeSkillServer, ServedSkill, skill_md
from tests.tool_loop_modes import LoopRun, respond

PDF = ServedSkill(
    "pdf-processing",
    {"SKILL.md": skill_md("pdf-processing", "Extract and fill PDF documents", "Run it.")},
)
NAME = "hf:pdf-processing"
BASH = {
    "name": "sandbox_bash",
    "description": "Run a shell command in the sandbox.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
}


class Sandbox(SandboxExecutor):
    def __init__(self) -> None:
        self.ran: list[str] = []

    async def execute(
        self, command: str, arguments: dict[str, Any] | None = None
    ) -> SandboxResult:
        self.ran.append(command)
        return SandboxResult(exit_code=0, output="ran")

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [BASH]


class Approver:
    def __init__(self, answer: bool | Exception = True) -> None:
        self.answer = answer
        self.asked: list[SkillExecutionRequest] = []

    async def __call__(self, request: SkillExecutionRequest) -> bool:
        self.asked.append(request)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _call(tool: str, **arguments: Any) -> AIResponse:
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=f"tc-{tool}-{len(arguments)}", name=tool, arguments=arguments)],
    )


def _say(text: str) -> AIResponse:
    return AIResponse(content=text, finish_reason="stop")


async def _channel(
    script: list[AIResponse], streaming: bool, **registration: Any
) -> tuple[AIChannel, Sandbox, list[tuple[str, dict[str, Any]]]]:
    registry = SkillRegistry()
    await registry.add_mcp_server(FakeSkillServer([PDF]), "hf", **registration)
    sandbox = Sandbox()
    host_calls: list[tuple[str, dict[str, Any]]] = []

    async def host(name: str, arguments: dict[str, Any]) -> str:
        host_calls.append((name, arguments))
        return "host ran"

    channel = AIChannel(
        "ai1",
        provider=MockAIProvider(ai_responses=script, streaming=streaming),
        skills=registry,
        sandbox=sandbox,
        tool_handler=host,
    )
    return channel, sandbox, host_calls


async def _turn(channel: AIChannel, tools: list[AITool] | None = None) -> LoopRun:
    binding = ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": [t.model_dump() for t in tools or []]},
    )
    context = RoomContext(room=Room(id="r1"))
    return await respond(channel, make_event(body="go", channel_id="sms1"), binding, context)


def _result(run: LoopRun, tool: str) -> str:
    call = next(c for c in run.calls if c.name == tool)
    return f"{call.result} {call.error or ''}"


async def test_a_sandbox_command_is_refused_while_a_served_skill_is_active(
    streaming: bool,
) -> None:
    script = [_call("activate_skill", name=NAME), _call("sandbox_bash", command="ls"), _say("ok")]
    channel, sandbox, _ = await _channel(script, streaming)
    run = await _turn(channel)
    assert "has not approved" in _result(run, "sandbox_bash")
    assert sandbox.ran == []


async def test_without_a_served_skill_active_the_sandbox_runs(streaming: bool) -> None:
    channel, sandbox, _ = await _channel(
        [_call("sandbox_bash", command="ls"), _say("ok")], streaming
    )
    await _turn(channel)
    assert sandbox.ran == ["bash"]


async def test_an_approval_lets_it_run_and_holds_for_the_fingerprint(streaming: bool) -> None:
    approver = Approver(True)
    script = [
        _call("activate_skill", name=NAME),
        _call("sandbox_bash", command="ls"),
        _call("sandbox_bash", command="pwd", cwd="/"),
        _say("ok"),
    ]
    channel, sandbox, _ = await _channel(script, streaming, approve_execution=approver)
    await _turn(channel)
    assert sandbox.ran == ["bash", "bash"]
    assert len(approver.asked) == 1
    request = approver.asked[0]
    assert (request.name, request.label, request.uri) == (NAME, "hf", PDF.uri)
    assert request.fingerprint is not None and request.fingerprint.startswith("sha256:")
    assert (request.room_id, request.tool) == ("r1", "sandbox_bash")


@pytest.mark.parametrize("answer", [False, RuntimeError("down")])
async def test_a_denial_or_a_failing_callback_refuses(
    answer: bool | Exception, streaming: bool
) -> None:
    script = [_call("activate_skill", name=NAME), _call("sandbox_bash", command="ls"), _say("ok")]
    channel, sandbox, _ = await _channel(script, streaming, approve_execution=Approver(answer))
    run = await _turn(channel)
    assert "has not approved" in _result(run, "sandbox_bash")
    assert sandbox.ran == []


async def test_a_host_tool_named_as_running_code_is_gated_and_others_are_not(
    streaming: bool,
) -> None:
    script = [
        _call("activate_skill", name=NAME),
        _call("run_python", code="print(1)"),
        _call("weather", city="Paris"),
        _say("ok"),
    ]
    channel, _, host_calls = await _channel(script, streaming, execution_tools=["run_*"])
    tools = [
        AITool(name="run_python", description="Run Python", parameters={"type": "object"}),
        AITool(name="weather", description="Weather", parameters={"type": "object"}),
    ]
    run = await _turn(channel, tools)
    assert "has not approved" in _result(run, "run_python")
    assert [name for name, _ in host_calls] == ["weather"]


class TestGateUnit:
    async def _registry(self, **registration: Any) -> SkillRegistry:
        registry = SkillRegistry()
        await registry.add_mcp_server(FakeSkillServer([PDF]), "hf", **registration)
        await registry.load_skill(NAME)
        return registry

    async def test_an_approval_is_asked_again_for_another_fingerprint(self) -> None:
        approver = Approver(True)
        registry = await self._registry(approve_execution=approver)
        gate = ServedExecutionGate()
        assert (
            await gate.refusal(
                registry, {NAME}, "r1", "sandbox_bash", sandbox_tools={"sandbox_bash"}
            )
            is None
        )
        assert (
            await gate.refusal(
                registry, {NAME}, "r1", "sandbox_bash", sandbox_tools={"sandbox_bash"}
            )
            is None
        )
        assert len(approver.asked) == 1
        # Another conversation is asked on its own.
        await gate.refusal(registry, {NAME}, "r2", "sandbox_bash", sandbox_tools={"sandbox_bash"})
        assert len(approver.asked) == 2

    async def test_a_dynamic_skill_is_asked_every_time_without_a_fingerprint(self) -> None:
        live = ServedSkill("live", {"SKILL.md": skill_md("live", "d")}, dynamic=True)
        approver = Approver(True)
        registry = SkillRegistry()
        await registry.add_mcp_server(
            FakeSkillServer([live]), "hf", allow_dynamic=True, approve_execution=approver
        )
        await registry.load_skill("hf:live")
        gate = ServedExecutionGate()
        for _ in range(2):
            await gate.refusal(
                registry, {"hf:live"}, "r1", "sandbox_bash", sandbox_tools={"sandbox_bash"}
            )
        assert [r.fingerprint for r in approver.asked] == [None, None]

    async def test_a_skill_s_script_counts_as_running_code(self) -> None:
        registry = await self._registry()
        refusal = await ServedExecutionGate().refusal(
            registry, {NAME, "local"}, "r1", "run_skill_script"
        )
        assert refusal is not None and NAME in refusal

    async def test_a_tool_that_runs_no_code_is_never_gated(self) -> None:
        registry = await self._registry()
        gate = ServedExecutionGate()
        assert await gate.refusal(registry, {NAME}, "r1", "read_skill_reference") is None


class TestGateNeverFailsOpen:
    """Found by an automated security review of the first gate commit."""

    async def test_an_active_served_skill_the_registry_no_longer_holds_is_refused(self) -> None:
        approver = Approver(True)
        server = FakeSkillServer([PDF])
        registry = SkillRegistry()
        await registry.add_mcp_server(server, "hf", approve_execution=approver)
        await registry.load_skill(NAME)
        server.skills = []
        await registry.add_mcp_server(server, "hf", approve_execution=approver)
        refusal = await ServedExecutionGate().refusal(
            registry, {NAME}, "r1", "sandbox_bash", sandbox_tools={"sandbox_bash"}
        )
        assert refusal is not None and approver.asked == []

    async def test_without_a_conversation_no_approval_is_kept(self) -> None:
        approver = Approver(True)
        registry = SkillRegistry()
        await registry.add_mcp_server(FakeSkillServer([PDF]), "hf", approve_execution=approver)
        await registry.load_skill(NAME)
        gate = ServedExecutionGate()
        for _ in range(2):
            await gate.refusal(
                registry, {NAME}, None, "sandbox_bash", sandbox_tools={"sandbox_bash"}
            )
        assert len(approver.asked) == 2

    async def test_an_approval_never_covers_the_same_entry_from_another_source(self) -> None:
        """Two turns of one room may give registries of their own (RFC §24.3),
        each with its server serving the same entry: one source's approval is
        that source's, and the other is asked."""
        approving, refusing = Approver(True), Approver(False)
        first, second = SkillRegistry(), SkillRegistry()
        await first.add_mcp_server(FakeSkillServer([PDF]), "hf", approve_execution=approving)
        await second.add_mcp_server(FakeSkillServer([PDF]), "hf", approve_execution=refusing)
        for registry in (first, second):
            await registry.load_skill(NAME)
        gate = ServedExecutionGate()

        allowed = await gate.refusal(
            first, {NAME}, "r1", "sandbox_bash", sandbox_tools={"sandbox_bash"}
        )
        refused = await gate.refusal(
            second, {NAME}, "r1", "sandbox_bash", sandbox_tools={"sandbox_bash"}
        )

        assert allowed is None
        assert refused is not None and NAME in refused
        assert len(refusing.asked) == 1
