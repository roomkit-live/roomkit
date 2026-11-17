"""Public sandbox carriers, catalogue and per-command timeout boundaries."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic import ValidationError

from roomkit import Agent, AIChannel, SandboxExecutor, SandboxResult
from roomkit.providers.ai import AIResponse, AITool, AIToolCall, MockAIProvider
from roomkit.sandbox import SANDBOX_PREAMBLE, SANDBOX_TOOL_PREFIX, SANDBOX_TOOL_SCHEMAS
from tests.conftest import make_event
from tests.test_tool_policy_exemptions import _binding
from tests.test_tool_search_text import _ctx
from tests.tool_loop_modes import LoopRun, respond


class RecordingSandbox(SandboxExecutor):
    def __init__(self, definitions: list[dict[str, Any]] | None = None) -> None:
        self.definitions = definitions if definitions is not None else [{"name": "sandbox_bash"}]
        self.calls: list[tuple[str, Any]] = []

    def tool_definitions(self) -> list[dict[str, Any]]:
        return self.definitions

    async def execute(
        self, command: str, arguments: dict[str, Any] | None = None
    ) -> SandboxResult:
        self.calls.append((command, arguments))
        return SandboxResult(exit_code=7, output="résultat", error="Unknown tool: frobnicate")


def channel_with_call(
    sandbox: SandboxExecutor, streaming: bool, name: str = "sandbox_bash", **options: Any
) -> tuple[AIChannel, MockAIProvider]:
    provider = MockAIProvider(
        streaming=streaming,
        ai_responses=[
            AIResponse(content="", tool_calls=[AIToolCall(id="s1", name=name, arguments={})]),
            AIResponse(content="done"),
        ],
    )
    return AIChannel("ai1", provider=provider, sandbox=sandbox, **options), provider


async def answer(channel: AIChannel) -> LoopRun:
    return await respond(channel, make_event(), _binding(), _ctx())


@pytest.mark.parametrize("exit_code", [0, 1, -1])
def test_success_is_derived_and_serialized_with_independent_defaults(exit_code: int) -> None:
    result = SandboxResult(exit_code=exit_code)
    other = SandboxResult(exit_code=0)
    result.metadata["attempt"] = 2
    assert other.metadata == {}
    assert result.success is (exit_code == 0)
    expected = {
        "exit_code": exit_code,
        "output": "",
        "error": "",
        "metadata": {"attempt": 2},
        "success": exit_code == 0,
    }
    assert result.model_dump() == expected
    assert json.loads(result.model_dump_json()) == expected
    result.exit_code = 0 if exit_code else 1
    assert result.success is (result.exit_code == 0)
    assert result.model_dump()["success"] is result.success


def test_exit_code_is_required_and_an_input_success_does_not_override_it() -> None:
    with pytest.raises(ValidationError):
        SandboxResult.model_validate({})
    assert SandboxResult.model_validate({"exit_code": 3, "success": True}).success is False
    assert SandboxResult.model_validate({"exit_code": 0, "success": False}).success is True


def test_reference_catalogue_is_ordered_and_does_not_supply_runtime_defaults() -> None:
    assert SANDBOX_TOOL_PREFIX == "sandbox_"
    names = [item["name"] for item in SANDBOX_TOOL_SCHEMAS]
    assert names == [
        "sandbox_read",
        "sandbox_write",
        "sandbox_edit",
        "sandbox_ls",
        "sandbox_grep",
        "sandbox_find",
        "sandbox_git",
        "sandbox_diff",
        "sandbox_delete",
        "sandbox_bash",
    ]
    assert all(item["parameters"]["type"] == "object" for item in SANDBOX_TOOL_SCHEMAS)
    bash = SANDBOX_TOOL_SCHEMAS[-1]["parameters"]
    assert bash["required"] == ["command"]
    assert bash["properties"]["timeout"] == {
        "type": "integer",
        "description": "Timeout in seconds. Default: 30.",
    }
    assert "Do not assume it is isolated from the host machine" in SANDBOX_PREAMBLE


async def test_execute_can_omit_arguments_and_inherits_a_no_op_close() -> None:
    sandbox = RecordingSandbox()
    await sandbox.execute("bash")
    await sandbox.close()
    assert sandbox.calls == [("bash", None)]


@pytest.mark.parametrize("channel_class", [AIChannel, Agent])
def test_the_constructor_refuses_a_host_tool_under_a_declared_sandbox_name(
    channel_class: type[AIChannel],
) -> None:
    sandbox = RecordingSandbox()
    with pytest.raises(ValueError, match="is a tool channel 'ai1' serves itself: rename it"):
        channel_class(
            "ai1",
            provider=MockAIProvider(),
            sandbox=sandbox,
            tools=[AITool(name="sandbox_bash", description="Host shadow")],
        )
    assert sandbox.calls == []


class FailingDefinitions(RecordingSandbox):
    def tool_definitions(self) -> list[dict[str, Any]]:
        raise ConnectionError("catalogue unavailable")


def test_a_catalogue_failure_reaches_the_constructor_without_running_a_command() -> None:
    sandbox = FailingDefinitions()
    with pytest.raises(ConnectionError, match="catalogue unavailable"):
        AIChannel("ai1", provider=MockAIProvider(), sandbox=sandbox)
    assert sandbox.calls == []


@pytest.mark.parametrize(
    ("definition", "error"), [({}, KeyError), ({"name": None}, AttributeError)]
)
def test_a_catalogue_with_no_readable_name_fails_before_execution(
    definition: dict[str, Any], error: type[Exception]
) -> None:
    sandbox = RecordingSandbox([definition])
    with pytest.raises(error):
        AIChannel("ai1", provider=MockAIProvider(), sandbox=sandbox)
    assert sandbox.calls == []


async def test_a_returned_nonzero_result_is_served_with_its_computed_success(
    streaming: bool,
) -> None:
    sandbox = RecordingSandbox()
    channel, provider = channel_with_call(sandbox, streaming)
    run = await answer(channel)
    await channel.close()
    assert sandbox.calls == [("bash", {})]
    assert len(run.calls) == 1 and not run.calls[0].failed
    assert json.loads(run.calls[0].result) == {
        "exit_code": 7,
        "output": "résultat",
        "error": "Unknown tool: frobnicate",
        "metadata": {},
        "success": False,
    }
    parts = [
        part
        for message in provider.calls[1].messages
        for part in message.content
        if getattr(part, "tool_call_id", None) == "s1"
    ]
    assert len(parts) == 1 and not parts[0].is_error
    assert parts[0].outcome == "served"


async def test_agent_forwards_the_executor_and_only_one_initial_prefix_is_removed(
    streaming: bool,
) -> None:
    sandbox = RecordingSandbox([{"name": "sandbox_sandbox_echo"}])
    _, provider = channel_with_call(sandbox, streaming, "sandbox_sandbox_echo")
    channel = Agent("ai1", provider=provider, sandbox=sandbox)
    await answer(channel)
    await channel.close()
    assert sandbox.calls == [("sandbox_echo", {})]


async def test_definitions_use_only_the_three_schema_fields_and_are_read_each_turn(
    streaming: bool,
) -> None:
    sandbox = RecordingSandbox(
        [
            {
                "name": "sandbox_first",
                "tags": ["ignored"],
                "changes_state": True,
                "defer_loading": True,
            }
        ]
    )
    provider = MockAIProvider(responses=["one", "two"], streaming=streaming)
    channel = AIChannel("ai1", provider=provider, sandbox=sandbox, tool_search=False)
    await answer(channel)
    first = next(tool for tool in provider.calls[0].tools or [] if tool.name == "sandbox_first")
    assert (first.description, first.parameters, first.tags) == ("", {}, [])
    assert not first.changes_state and not first.defer_loading
    sandbox.definitions = [{"name": "sandbox_second", "description": "Second"}]
    await answer(channel)
    await channel.close()
    names = {tool.name for tool in provider.calls[1].tools or []}
    assert "sandbox_second" in names and "sandbox_first" not in names
    assert sandbox.calls == []


@pytest.mark.parametrize("definitions", [[], [{"name": "host_only"}]])
async def test_empty_or_nonprefixed_catalogues_offer_no_sandbox_prompt(
    definitions: list[dict[str, Any]], streaming: bool, caplog: pytest.LogCaptureFixture
) -> None:
    sandbox = RecordingSandbox(definitions)
    provider = MockAIProvider(responses=["done"], streaming=streaming)
    channel = AIChannel("ai1", provider=provider, sandbox=sandbox)
    await answer(channel)
    await channel.close()
    assert not {tool.name for tool in provider.calls[0].tools or []} & {
        "host_only",
        "sandbox_bash",
    }
    assert SANDBOX_PREAMBLE not in (provider.calls[0].system_prompt or "")
    assert sandbox.calls == []
    assert ("does not start with" in caplog.text) is bool(definitions)


class HeldSandbox(RecordingSandbox):
    def __init__(self, name: str) -> None:
        super().__init__([{"name": name}])
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False

    async def execute(
        self, command: str, arguments: dict[str, Any] | None = None
    ) -> SandboxResult:
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return SandboxResult(exit_code=0, output="finished")


@pytest.mark.parametrize(
    ("name", "overrides", "expires"),
    [
        ("sandbox_bash", {}, False),
        ("sandbox_read", {}, True),
        ("sandbox_custom", {}, True),
        ("sandbox_bash", {"sandbox_bash": 0.02}, True),
        ("sandbox_read", {"sandbox_read": None}, False),
    ],
)
async def test_only_bash_escapes_the_default_and_a_named_bound_still_wins(
    name: str, overrides: dict[str, float | None], expires: bool, streaming: bool
) -> None:
    sandbox = HeldSandbox(name)
    channel, _ = channel_with_call(
        sandbox, streaming, name, tool_timeout_seconds=0.02, tool_timeouts=overrides
    )
    task = asyncio.create_task(answer(channel))
    try:
        await asyncio.wait_for(sandbox.entered.wait(), 3)
        if expires:
            run = await asyncio.wait_for(asyncio.shield(task), 3)
            assert sandbox.cancelled
            assert run.calls[0].failed
            assert "ToolTimeoutError" in json.loads(run.calls[0].result)["error"]
        else:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(task), 0.06)
            assert not sandbox.cancelled
            sandbox.release.set()
            run = await asyncio.wait_for(task, 3)
            assert not run.calls[0].failed
            assert json.loads(run.calls[0].result)["output"] == "finished"
    finally:
        sandbox.release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await channel.close()
