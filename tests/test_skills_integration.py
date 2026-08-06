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
    AITool,
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

    async def test_activate_unavailable_skill_reports_reason(self, tmp_path: Path) -> None:
        """A skill marked unavailable answers with its reason, not "not found"."""
        _make_skill_dir(tmp_path, "known")
        registry = SkillRegistry()
        registry.discover(tmp_path)
        registry.mark_unavailable(
            "gated-skill", "requires tool 'artifacts' which is not granted in this context"
        )

        provider = ToolCallMockProvider(
            tool_calls=[
                AIToolCall(
                    id="tc1",
                    name="activate_skill",
                    arguments={"name": "gated-skill"},
                )
            ],
        )
        ch = AIChannel("ai1", provider=provider, skills=registry)
        await ch.on_event(make_event(body="go", channel_id="sms1"), _binding(), _ctx())

        messages = provider.calls[1].messages
        tool_msg = [m for m in messages if m.role == "tool"]
        result_json = json.loads(tool_msg[0].content[0].result)
        assert "unavailable in this context" in result_json["error"]
        assert "requires tool 'artifacts'" in result_json["error"]
        assert "not found" not in result_json["error"]
        assert result_json["available_skills"] == ["known"]


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
            tools=[AITool(name="custom_search", description="Search", parameters={})],
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
        assert "".join(c for c in chunks if isinstance(c, str)) == "ok"

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


class PerTurnToolCallProvider(MockAIProvider):
    """Emits the same tool call at the start of every turn, then answers.

    The channel calls ``generate`` twice per turn when the model uses one tool
    round — the call, then the answer — so alternating on the call count
    reproduces a model that re-issues ``activate_skill`` on each new turn,
    which is the behaviour this suite is about.
    """

    def __init__(self, name: str, arguments: dict[str, Any], *, streaming: bool = False) -> None:
        super().__init__(responses=["Done."], streaming=streaming)
        self._name = name
        self._arguments = arguments
        self._n = 0

    async def generate(self, context: AIContext) -> AIResponse:
        self.calls.append(context)
        self._n += 1
        if self._n % 2 == 1:
            return AIResponse(
                content="",
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                tool_calls=[
                    AIToolCall(id=f"tc{self._n}", name=self._name, arguments=self._arguments)
                ],
            )
        return AIResponse(
            content="Done.",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )


def _tool_results(context: AIContext) -> list[dict[str, Any]]:
    """Decode the tool results carried by a generate call's messages."""
    results: list[dict[str, Any]] = []
    for message in context.messages:
        if message.role != "tool" or not isinstance(message.content, list):
            continue
        results.extend(json.loads(part.result) for part in message.content)
    return results


class TestActivationSurvivesTheTurn:
    """A skill activated once stays active — body in the prompt, ACK on the tool."""

    async def test_body_once_then_ack_with_rules_in_the_prompt(self, tmp_path: Path) -> None:
        _make_skill_dir_full(
            tmp_path,
            "onboarding",
            references=[("catalog.md", "# Catalog")],
            body="Greet the member, then walk them through setup.",
        )
        registry = SkillRegistry()
        registry.discover(tmp_path)

        provider = PerTurnToolCallProvider("activate_skill", {"name": "onboarding"})
        ch = AIChannel("ai1", provider=provider, skills=registry)

        await ch.on_event(make_event(body="hi", channel_id="sms1"), _binding(), _ctx())
        await ch.on_event(make_event(body="and then?", channel_id="sms1"), _binding(), _ctx())

        # Turn 1: the prompt cannot carry a body nobody has activated yet, so
        # the tool answers with the full instructions.
        assert "Active skill instructions" not in (provider.calls[0].system_prompt or "")
        first = _tool_results(provider.calls[1])[0]
        assert "Greet the member" in first["instructions"]
        assert "catalog.md" in first["references"]

        # Turn 2: the rules ride the system prompt, so the tool only acks.
        second_prompt = provider.calls[2].system_prompt or ""
        assert "Active skill instructions" in second_prompt
        assert "Greet the member, then walk them through setup." in second_prompt
        ack = _tool_results(provider.calls[3])[-1]
        assert ack["already_active"] is True
        assert "instructions" not in ack
        assert "catalog.md" in ack["references"]

    async def test_bodies_reach_a_host_that_renders_its_own_manifest(self, tmp_path: Path) -> None:
        """``skills_in_prompt=False`` drops the catalogue, never the active rules.

        The flag says "I render the skills manifest myself"; a host cannot know
        what the model activated mid-conversation, so the runtime state stays
        RoomKit's to inject.
        """
        _make_skill_dir(tmp_path, "onboarding", body="Follow the onboarding script.")
        registry = SkillRegistry()
        registry.discover(tmp_path)

        provider = PerTurnToolCallProvider("activate_skill", {"name": "onboarding"})
        ch = AIChannel("ai1", provider=provider, skills=registry, skills_in_prompt=False)

        await ch.on_event(make_event(body="hi", channel_id="sms1"), _binding(), _ctx())
        await ch.on_event(make_event(body="next", channel_id="sms1"), _binding(), _ctx())

        prompt = provider.calls[2].system_prompt or ""
        assert "<available_skills>" not in prompt
        assert "Follow the onboarding script." in prompt

    async def test_the_streaming_loop_gets_the_same_lifecycle(self, tmp_path: Path) -> None:
        """Both tool loops run the same round, so streaming acks the same way."""
        _make_skill_dir(tmp_path, "onboarding", body="Follow the onboarding script.")
        registry = SkillRegistry()
        registry.discover(tmp_path)

        provider = PerTurnToolCallProvider(
            "activate_skill", {"name": "onboarding"}, streaming=True
        )
        ch = AIChannel("ai1", provider=provider, skills=registry)

        for body in ("hi", "next"):
            output = await ch.on_event(
                make_event(body=body, channel_id="sms1"), _binding(), _ctx()
            )
            assert output.response_stream is not None
            [chunk async for chunk in output.response_stream]

        assert "Follow the onboarding script." in (provider.calls[2].system_prompt or "")
        assert _tool_results(provider.calls[3])[-1]["already_active"] is True

    async def test_activation_is_scoped_to_its_room(self, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "onboarding", body="Follow the onboarding script.")
        registry = SkillRegistry()
        registry.discover(tmp_path)

        provider = PerTurnToolCallProvider("activate_skill", {"name": "onboarding"})
        ch = AIChannel("ai1", provider=provider, skills=registry)

        await ch.on_event(make_event(body="hi", channel_id="sms1"), _binding(), _ctx())

        other_binding = ChannelBinding(
            channel_id="ai1",
            room_id="r2",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        )
        other_ctx = RoomContext(room=Room(id="r2"))
        await ch.on_event(
            make_event(body="hi", channel_id="sms1", room_id="r2"), other_binding, other_ctx
        )

        # Another conversation gets the body, not one room's activation.
        assert "Active skill instructions" not in (provider.calls[2].system_prompt or "")
        other_result = _tool_results(provider.calls[3])[0]
        assert "Follow the onboarding script." in other_result["instructions"]

    async def test_gated_tools_stay_visible_on_later_turns(self, tmp_path: Path) -> None:
        """Activation counts for the conversation, so its tools stop re-hiding."""
        skill_dir = tmp_path / "publisher"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: publisher\ndescription: Publishes\n"
            "allowed_tools: publish_site\n---\nPublish carefully.",
            encoding="utf-8",
        )
        registry = SkillRegistry()
        registry.discover(tmp_path)

        provider = PerTurnToolCallProvider("activate_skill", {"name": "publisher"})
        ch = AIChannel("ai1", provider=provider, skills=registry)
        binding = ChannelBinding(
            channel_id="ai1",
            room_id="r1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
            metadata={"tools": [{"name": "publish_site", "description": "Publish"}]},
        )

        await ch.on_event(make_event(body="hi", channel_id="sms1"), binding, _ctx())
        await ch.on_event(make_event(body="publish it", channel_id="sms1"), binding, _ctx())

        # Turn 1 round 0: gated. Turn 2 round 0: still visible, no re-activation.
        assert "publish_site" not in [t.name for t in provider.calls[0].tools]
        assert "publish_site" in [t.name for t in provider.calls[2].tools]


class TestSkillBodyEscapesEviction:
    """Instructions are binding rules — never a preview behind a pointer."""

    async def test_large_skill_body_is_returned_whole(self, tmp_path: Path) -> None:
        body = "Follow this rule.\n" * 400  # ~7 KB, well past the threshold below
        _make_skill_dir(tmp_path, "verbose", body=body)
        registry = SkillRegistry()
        registry.discover(tmp_path)

        provider = PerTurnToolCallProvider("activate_skill", {"name": "verbose"})
        ch = AIChannel("ai1", provider=provider, skills=registry, evict_threshold_tokens=100)

        await ch.on_event(make_event(body="hi", channel_id="sms1"), _binding(), _ctx())

        result = _tool_results(provider.calls[1])[0]
        assert result["instructions"].count("Follow this rule.") == 400
        assert "read_stored_result" not in json.dumps(result)

    async def test_large_reference_is_still_evicted(self, tmp_path: Path) -> None:
        """A reference is data, and paginating data is what eviction is for."""
        _make_skill_dir_full(
            tmp_path,
            "documented",
            references=[("big.md", "line of reference\n" * 400)],
        )
        registry = SkillRegistry()
        registry.discover(tmp_path)

        provider = PerTurnToolCallProvider(
            "read_skill_reference", {"skill_name": "documented", "filename": "big.md"}
        )
        ch = AIChannel("ai1", provider=provider, skills=registry, evict_threshold_tokens=100)

        await ch.on_event(make_event(body="hi", channel_id="sms1"), _binding(), _ctx())

        messages = provider.calls[1].messages
        tool_msg = [m for m in messages if m.role == "tool"][0]
        assert "read_stored_result" in tool_msg.content[0].result
