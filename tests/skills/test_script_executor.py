"""Public local-script execution and shutdown contracts for native ports."""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any

import pytest
from tests.conftest import make_event
from tests.test_skills_integration import _binding, _ctx, _make_skill_dir_full
from tests.tool_loop_modes import respond

from roomkit import AIChannel, RoomKit
from roomkit.providers.ai import AIResponse, AIToolCall, MockAIProvider
from roomkit.skills import ScriptExecutor, ScriptResult, Skill, SkillRegistry


class RecordingExecutor(ScriptExecutor):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any]] = []

    async def execute(
        self, skill: Skill, script_name: str, arguments: dict[str, str] | None = None
    ) -> ScriptResult:
        self.calls.append((skill.name, script_name, arguments))
        return ScriptResult(exit_code=7, stdout="résultat", metadata={"attempt": 2})


def script_channel(
    root: Path, executor: ScriptExecutor, streaming: bool, arguments: dict[str, Any]
) -> tuple[AIChannel, MockAIProvider]:
    _make_skill_dir_full(root, "local", scripts=["build.sh"])
    skills = SkillRegistry()
    skills.discover(root)
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="s1", name="run_skill_script", arguments=arguments)],
            ),
            AIResponse(content="done"),
        ],
        streaming=streaming,
    )
    return AIChannel("ai1", provider=provider, skills=skills, script_executor=executor), provider


@pytest.mark.parametrize("arguments", [None, {}, {"target": "release"}])
async def test_script_result_and_optional_arguments_cross_the_tool_boundary(
    tmp_path: Path, streaming: bool, arguments: dict[str, str] | None
) -> None:
    executor = RecordingExecutor()
    call: dict[str, Any] = {"skill_name": "local", "script_name": "build.sh"}
    if arguments is not None:
        call["arguments"] = arguments
    channel, _ = script_channel(tmp_path, executor, streaming, call)
    result = await respond(channel, make_event(), _binding(), _ctx())
    await channel.close()  # The executor's inherited close needs no override.

    assert executor.calls == [("local", "build.sh", arguments)]
    assert len(result.calls) == 1 and not result.calls[0].failed
    assert json.loads(result.calls[0].result) == {
        "exit_code": 7,
        "stdout": "résultat",
        "stderr": "",
        "success": True,
        "metadata": {"attempt": 2},
    }


class FailingCloseExecutor(RecordingExecutor):
    async def close(self) -> None:
        raise ConnectionError("script executor cleanup failed")


async def test_a_direct_channel_close_propagates_its_executor_failure() -> None:
    channel = AIChannel("ai1", provider=MockAIProvider(), script_executor=FailingCloseExecutor())
    # An executor still belongs to the channel when no registry is configured.
    with pytest.raises(ConnectionError, match="script executor cleanup failed"):
        await channel.close()


async def test_a_kit_close_reports_its_script_executor_failure() -> None:
    kit = RoomKit()
    kit.register_channel(
        AIChannel("ai1", provider=MockAIProvider(), script_executor=FailingCloseExecutor())
    )
    with pytest.raises(ExceptionGroup) as caught:
        await kit.close()
    assert any(
        isinstance(error, ConnectionError) and str(error) == "script executor cleanup failed"
        for error in caught.value.exceptions
    )


class HeldExecutor(RecordingExecutor):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.order: list[str] = []

    async def execute(
        self, skill: Skill, script_name: str, arguments: dict[str, str] | None = None
    ) -> ScriptResult:
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.order.append("call ended")
        return ScriptResult(exit_code=0)

    async def close(self) -> None:
        self.order.append("executor closed")


async def test_a_channel_close_ends_its_script_before_closing_the_executor(
    tmp_path: Path, streaming: bool
) -> None:
    executor = HeldExecutor()
    channel, provider = script_channel(
        tmp_path, executor, streaming, {"skill_name": "local", "script_name": "build.sh"}
    )
    turn = asyncio.create_task(respond(channel, make_event(), _binding(), _ctx()))
    try:
        await asyncio.wait_for(executor.started.wait(), 3)
        await asyncio.wait_for(channel.close(), 3)
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(turn, 3)
        assert executor.order == ["call ended", "executor closed"]
        assert len(provider.calls) == 1
    finally:
        turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)
