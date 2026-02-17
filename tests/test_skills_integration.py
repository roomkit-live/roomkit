"""Integration tests for AIChannel + skills."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import (
    AIContext,
    AIResponse,
    AIToolCall,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.skills.executor import ScriptExecutor
from roomkit.skills.models import ScriptResult, Skill
from roomkit.skills.registry import SkillRegistry
from tests.conftest import make_event

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_skill_dir(tmp_path: Path, name: str, body: str = "Do the thing.") -> Path:
    skill_dir = tmp_path / name
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: A test skill\n---\n{body}",
        encoding="utf-8",
    )
    return skill_dir


def _make_skill_dir_full(
    tmp_path: Path,
    name: str,
    *,
    scripts: list[str] | None = None,
    references: list[tuple[str, str]] | None = None,
    body: str = "Instructions here.",
) -> Path:
    skill_dir = tmp_path / name
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Full skill\n---\n{body}",
        encoding="utf-8",
    )
    if scripts:
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir()
        for s in scripts:
            (scripts_dir / s).write_text(f"# {s}", encoding="utf-8")
    if references:
        refs_dir = skill_dir / "references"
        refs_dir.mkdir()
        for fname, content in references:
            (refs_dir / fname).write_text(content, encoding="utf-8")
    return skill_dir


def _binding() -> ChannelBinding:
    return ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )


def _ctx() -> RoomContext:
    return RoomContext(room=Room(id="r1"))


class ToolCallMockProvider(MockAIProvider):
    """Mock provider that returns tool calls on first generate, then text."""

    def __init__(
        self,
        tool_calls: list[AIToolCall],
        final_response: str = "Done.",
    ) -> None:
        super().__init__(responses=[final_response])
        self._tool_calls = tool_calls
        self._first_call = True

    async def generate(self, context: AIContext) -> AIResponse:
        self.calls.append(context)
        if self._first_call:
            self._first_call = False
            return AIResponse(
                content="",
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                tool_calls=self._tool_calls,
            )
        return AIResponse(
            content=self.responses[0],
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )


class MockScriptExecutor(ScriptExecutor):
    """Test executor that records calls and returns canned results."""

    def __init__(self, result: ScriptResult | None = None) -> None:
        self.calls: list[tuple[str, str, dict[str, str] | None]] = []
        self._result = result or ScriptResult(exit_code=0, stdout="OK")

    async def execute(
        self,
        skill: Skill,
        script_name: str,
        arguments: dict[str, str] | None = None,
    ) -> ScriptResult:
        self.calls.append((skill.name, script_name, arguments))
        return self._result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSkillsSystemPrompt:
    """Skills preamble and XML are injected into the system prompt."""

    async def test_skills_injected_into_system_prompt(self, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "code-review")
        registry = SkillRegistry()
        registry.discover(tmp_path)

        provider = MockAIProvider(responses=["ok"])
        ch = AIChannel(
            "ai1",
            provider=provider,
            system_prompt="Be helpful.",
            skills=registry,
        )
        await ch.on_event(make_event(body="hello", channel_id="sms1"), _binding(), _ctx())

        assert len(provider.calls) == 1
        prompt = provider.calls[0].system_prompt
        assert prompt is not None
        assert "Be helpful." in prompt
        assert "<available_skills>" in prompt
        assert "code-review" in prompt
        assert "activate_skill" in prompt

    async def test_no_scripts_note_when_no_executor(self, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "no-exec")
        registry = SkillRegistry()
        registry.discover(tmp_path)

        provider = MockAIProvider(responses=["ok"])
        ch = AIChannel("ai1", provider=provider, skills=registry)
        await ch.on_event(make_event(body="hi", channel_id="sms1"), _binding(), _ctx())

        prompt = provider.calls[0].system_prompt
        assert prompt is not None
        assert "not available" in prompt

    async def test_no_scripts_note_absent_when_executor_set(self, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "has-exec")
        registry = SkillRegistry()
        registry.discover(tmp_path)

        provider = MockAIProvider(responses=["ok"])
        ch = AIChannel(
            "ai1",
            provider=provider,
            skills=registry,
            script_executor=MockScriptExecutor(),
        )
        await ch.on_event(make_event(body="hi", channel_id="sms1"), _binding(), _ctx())

        prompt = provider.calls[0].system_prompt
        assert prompt is not None
        assert "not available" not in prompt


class TestSkillToolInjection:
    """Skill tools are added to the AI context."""

    async def test_activate_and_read_tools_present(self, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "my-skill")
        registry = SkillRegistry()
        registry.discover(tmp_path)

        provider = MockAIProvider(responses=["ok"])
        ch = AIChannel("ai1", provider=provider, skills=registry)
        await ch.on_event(make_event(body="go", channel_id="sms1"), _binding(), _ctx())

        tools = provider.calls[0].tools
        tool_names = [t.name for t in tools]
        assert "activate_skill" in tool_names
        assert "read_skill_reference" in tool_names
        assert "run_skill_script" not in tool_names

    async def test_run_script_tool_present_with_executor(self, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "scripted")
        registry = SkillRegistry()
        registry.discover(tmp_path)

        provider = MockAIProvider(responses=["ok"])
        ch = AIChannel(
            "ai1",
            provider=provider,
            skills=registry,
            script_executor=MockScriptExecutor(),
        )
        await ch.on_event(make_event(body="go", channel_id="sms1"), _binding(), _ctx())

        tool_names = [t.name for t in provider.calls[0].tools]
        assert "run_skill_script" in tool_names

    async def test_user_tools_preserved(self, tmp_path: Path) -> None:
        """User-defined tools from binding metadata are kept alongside skill tools."""
        _make_skill_dir(tmp_path, "alongside")
        registry = SkillRegistry()
        registry.discover(tmp_path)

        provider = MockAIProvider(responses=["ok"])
        ch = AIChannel("ai1", provider=provider, skills=registry)
        binding = ChannelBinding(
            channel_id="ai1",
            room_id="r1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
            metadata={"tools": [{"name": "search", "description": "Search web"}]},
        )
        await ch.on_event(make_event(body="go", channel_id="sms1"), binding, _ctx())

        tool_names = [t.name for t in provider.calls[0].tools]
        assert "search" in tool_names
        assert "activate_skill" in tool_names


class TestActivateSkillHandler:
    """Test the activate_skill tool handler end-to-end."""

    async def test_activate_skill_returns_instructions(self, tmp_path: Path) -> None:
        _make_skill_dir_full(
            tmp_path,
            "code-gen",
            scripts=["gen.sh"],
            references=[("api.md", "# API")],
            body="Generate code from templates.",
        )
        registry = SkillRegistry()
        registry.discover(tmp_path)

        provider = ToolCallMockProvider(
            tool_calls=[
                AIToolCall(
                    id="tc1",
                    name="activate_skill",
                    arguments={"name": "code-gen"},
                )
            ],
            final_response="I activated code-gen.",
        )
        ch = AIChannel("ai1", provider=provider, skills=registry)
        output = await ch.on_event(
            make_event(body="activate code-gen", channel_id="sms1"),
            _binding(),
            _ctx(),
        )

        assert output.responded is True
        # The provider should have been called twice (tool call + final)
        assert len(provider.calls) == 2

        # Inspect tool result in second call's messages
        messages = provider.calls[1].messages
        tool_msg = [m for m in messages if m.role == "tool"]
        assert len(tool_msg) == 1
        result_parts = tool_msg[0].content
        assert isinstance(result_parts, list)
        result_json = json.loads(result_parts[0].result)
        assert result_json["name"] == "code-gen"
        assert "Generate code" in result_json["instructions"]
        assert "gen.sh" in result_json["scripts"]
        assert "api.md" in result_json["references"]

    async def test_activate_unknown_skill(self, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "known")
        registry = SkillRegistry()
        registry.discover(tmp_path)

        provider = ToolCallMockProvider(
            tool_calls=[
                AIToolCall(
                    id="tc1",
                    name="activate_skill",
                    arguments={"name": "unknown"},
                )
            ],
        )
        ch = AIChannel("ai1", provider=provider, skills=registry)
        await ch.on_event(make_event(body="go", channel_id="sms1"), _binding(), _ctx())

        messages = provider.calls[1].messages
        tool_msg = [m for m in messages if m.role == "tool"]
        result_json = json.loads(tool_msg[0].content[0].result)
        assert "not found" in result_json["error"]
        assert "known" in result_json["available_skills"]


class TestReadReferenceHandler:
    """Test the read_skill_reference tool handler."""

    async def test_read_reference(self, tmp_path: Path) -> None:
        _make_skill_dir_full(
            tmp_path,
            "ref-skill",
            references=[("guide.md", "# Guide\nStep 1...")],
        )
        registry = SkillRegistry()
        registry.discover(tmp_path)

        provider = ToolCallMockProvider(
            tool_calls=[
                AIToolCall(
                    id="tc1",
                    name="read_skill_reference",
                    arguments={"skill_name": "ref-skill", "filename": "guide.md"},
                )
            ],
        )
        ch = AIChannel("ai1", provider=provider, skills=registry)
        await ch.on_event(make_event(body="read guide", channel_id="sms1"), _binding(), _ctx())

        messages = provider.calls[1].messages
        tool_msg = [m for m in messages if m.role == "tool"]
        result_json = json.loads(tool_msg[0].content[0].result)
        assert result_json["filename"] == "guide.md"
        assert "Step 1" in result_json["content"]

    async def test_read_reference_traversal_blocked(self, tmp_path: Path) -> None:
        _make_skill_dir_full(
            tmp_path,
            "sec-skill",
            references=[("safe.txt", "OK")],
        )
        registry = SkillRegistry()
        registry.discover(tmp_path)

        provider = ToolCallMockProvider(
            tool_calls=[
                AIToolCall(
                    id="tc1",
                    name="read_skill_reference",
                    arguments={"skill_name": "sec-skill", "filename": "../secret"},
                )
            ],
        )
        ch = AIChannel("ai1", provider=provider, skills=registry)
        await ch.on_event(make_event(body="hack", channel_id="sms1"), _binding(), _ctx())

        messages = provider.calls[1].messages
        tool_msg = [m for m in messages if m.role == "tool"]
        result_json = json.loads(tool_msg[0].content[0].result)
        assert "error" in result_json


class TestRunScriptHandler:
    """Test the run_skill_script tool handler."""

    async def test_run_script(self, tmp_path: Path) -> None:
        _make_skill_dir_full(tmp_path, "scripted", scripts=["build.sh"])
        registry = SkillRegistry()
        registry.discover(tmp_path)

        executor = MockScriptExecutor(
            ScriptResult(exit_code=0, stdout="Build complete", success=True)
        )
        provider = ToolCallMockProvider(
            tool_calls=[
                AIToolCall(
                    id="tc1",
                    name="run_skill_script",
                    arguments={
                        "skill_name": "scripted",
                        "script_name": "build.sh",
                        "arguments": {"target": "release"},
                    },
                )
            ],
        )
        ch = AIChannel(
            "ai1",
            provider=provider,
            skills=registry,
            script_executor=executor,
        )
        await ch.on_event(make_event(body="build it", channel_id="sms1"), _binding(), _ctx())

        # Executor was called
        assert len(executor.calls) == 1
        assert executor.calls[0] == ("scripted", "build.sh", {"target": "release"})

        # Result was returned to provider
        messages = provider.calls[1].messages
        tool_msg = [m for m in messages if m.role == "tool"]
        result_json = json.loads(tool_msg[0].content[0].result)
        assert result_json["exit_code"] == 0
        assert result_json["stdout"] == "Build complete"

    async def test_run_script_no_executor(self, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "no-exec")
        registry = SkillRegistry()
        registry.discover(tmp_path)

        provider = ToolCallMockProvider(
            tool_calls=[
                AIToolCall(
                    id="tc1",
                    name="run_skill_script",
                    arguments={"skill_name": "no-exec", "script_name": "x.sh"},
                )
            ],
        )
        # No script_executor — run_skill_script tool shouldn't be injected,
        # but if AI calls it anyway, we handle gracefully
        ch = AIChannel("ai1", provider=provider, skills=registry)
        await ch.on_event(make_event(body="go", channel_id="sms1"), _binding(), _ctx())

        messages = provider.calls[1].messages
        tool_msg = [m for m in messages if m.role == "tool"]
        result_json = json.loads(tool_msg[0].content[0].result)
        assert "not available" in result_json["error"]


class TestUserToolHandlerDelegation:
    """Skill handler delegates non-skill tools to the user's handler."""

    async def test_user_tool_delegated(self, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "delegator")
        registry = SkillRegistry()
        registry.discover(tmp_path)

        user_calls: list[tuple[str, dict[str, Any]]] = []

        async def user_handler(name: str, args: dict[str, Any]) -> str:
            user_calls.append((name, args))
            return "user result"

        provider = ToolCallMockProvider(
            tool_calls=[
                AIToolCall(
                    id="tc1",
                    name="custom_search",
                    arguments={"query": "test"},
                )
            ],
        )
        ch = AIChannel(
            "ai1",
            provider=provider,
            skills=registry,
            tool_handler=user_handler,
        )
        await ch.on_event(make_event(body="search", channel_id="sms1"), _binding(), _ctx())

        assert len(user_calls) == 1
        assert user_calls[0] == ("custom_search", {"query": "test"})

    async def test_unknown_tool_no_user_handler(self, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "no-handler")
        registry = SkillRegistry()
        registry.discover(tmp_path)

        provider = ToolCallMockProvider(
            tool_calls=[
                AIToolCall(
                    id="tc1",
                    name="mystery_tool",
                    arguments={},
                )
            ],
        )
        ch = AIChannel("ai1", provider=provider, skills=registry)
        await ch.on_event(make_event(body="go", channel_id="sms1"), _binding(), _ctx())

        messages = provider.calls[1].messages
        tool_msg = [m for m in messages if m.role == "tool"]
        result_json = json.loads(tool_msg[0].content[0].result)
        assert "Unknown tool" in result_json["error"]


class TestStreamingGuard:
    """Skills with streaming provider use the streaming tool loop."""

    async def test_streaming_with_skills_uses_streaming_tool_loop(self, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "stream-test")
        registry = SkillRegistry()
        registry.discover(tmp_path)

        provider = MockAIProvider(responses=["ok"], streaming=True)
        ch = AIChannel("ai1", provider=provider, skills=registry)
        output = await ch.on_event(make_event(body="go", channel_id="sms1"), _binding(), _ctx())
        # Streaming provider with tools → streaming tool loop → response_stream
        assert output.responded is True
        assert output.response_stream is not None

        # Consume the stream to get the text
        chunks = [chunk async for chunk in output.response_stream]
        assert "".join(chunks) == "ok"

    async def test_non_streaming_provider_with_skills_uses_generate(self, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "no-stream")
        registry = SkillRegistry()
        registry.discover(tmp_path)

        provider = MockAIProvider(responses=["ok"])
        # MockAIProvider.supports_streaming is False by default
        ch = AIChannel("ai1", provider=provider, skills=registry)
        output = await ch.on_event(make_event(body="go", channel_id="sms1"), _binding(), _ctx())
        # Non-streaming provider → _generate_response path
        assert output.responded is True
        assert output.response_stream is None


class TestNoSkillsNoop:
    """Channel without skills behaves exactly as before."""

    async def test_no_skills_no_change(self) -> None:
        provider = MockAIProvider(responses=["hello"])
        ch = AIChannel("ai1", provider=provider, system_prompt="Be nice.")
        output = await ch.on_event(make_event(body="hi", channel_id="sms1"), _binding(), _ctx())
        assert output.responded is True
        assert provider.calls[0].system_prompt == "Be nice."
        assert len(provider.calls[0].tools) == 0

    async def test_empty_registry_no_change(self) -> None:
        registry = SkillRegistry()  # empty
        provider = MockAIProvider(responses=["hello"])
        ch = AIChannel("ai1", provider=provider, system_prompt="Be nice.", skills=registry)
        output = await ch.on_event(make_event(body="hi", channel_id="sms1"), _binding(), _ctx())
        assert output.responded is True
        assert provider.calls[0].system_prompt == "Be nice."
        assert len(provider.calls[0].tools) == 0
