"""Tests for RealtimeVoiceChannel skill support."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from roomkit import RoomKit
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.skills.registry import SkillRegistry
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_skill(
    tmp_path: Path,
    name: str,
    *,
    body: str = "Skill instructions here.",
    references: list[tuple[str, str]] | None = None,
    allowed_tools: str | None = None,
) -> Path:
    """Create a minimal skill directory."""
    skill_dir = tmp_path / name
    skill_dir.mkdir()
    extra = f"allowed_tools: {allowed_tools}\n" if allowed_tools else ""
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test skill\n{extra}---\n{body}",
        encoding="utf-8",
    )
    if references:
        refs_dir = skill_dir / "references"
        refs_dir.mkdir()
        for fname, content in references:
            (refs_dir / fname).write_text(content, encoding="utf-8")
    return skill_dir


def _registry_with_skill(tmp_path: Path, **kwargs: Any) -> SkillRegistry:
    """Create a registry with a single test skill."""
    _make_skill(tmp_path, "test-skill", **kwargs)
    registry = SkillRegistry()
    registry.discover(tmp_path)
    return registry


@pytest.fixture
def provider() -> MockRealtimeProvider:
    return MockRealtimeProvider()


@pytest.fixture
def transport() -> MockRealtimeTransport:
    return MockRealtimeTransport()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSkillsPromptInjection:
    async def test_system_prompt_includes_skills_preamble(
        self,
        tmp_path: Path,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
    ) -> None:
        registry = _registry_with_skill(tmp_path)

        # Spy on provider.connect to capture the system_prompt arg
        original_connect = provider.connect
        connect_calls: list[dict[str, Any]] = []

        async def spy_connect(session: Any, **kwargs: Any) -> None:
            connect_calls.append(kwargs)
            await original_connect(session, **kwargs)

        provider.connect = spy_connect  # type: ignore[method-assign]

        channel = RealtimeVoiceChannel(
            "rt-skill",
            provider=provider,
            transport=transport,
            system_prompt="Base prompt.",
            skills=registry,
        )
        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-skill")

        await channel.start_session(room.id, "user-1", "fake-ws")

        assert len(connect_calls) == 1
        connect_prompt = connect_calls[0]["system_prompt"]
        assert "Base prompt." in connect_prompt
        assert "Agent Skills" in connect_prompt
        assert "<available_skills>" in connect_prompt
        assert "test-skill" in connect_prompt

    async def test_no_skills_no_preamble(
        self,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
    ) -> None:
        """Without skills, the prompt is unchanged."""
        original_connect = provider.connect
        connect_calls: list[dict[str, Any]] = []

        async def spy_connect(session: Any, **kwargs: Any) -> None:
            connect_calls.append(kwargs)
            await original_connect(session, **kwargs)

        provider.connect = spy_connect  # type: ignore[method-assign]

        channel = RealtimeVoiceChannel(
            "rt-no-skill",
            provider=provider,
            transport=transport,
            system_prompt="Plain prompt.",
        )
        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-no-skill")

        await channel.start_session(room.id, "user-1", "fake-ws")

        assert connect_calls[0]["system_prompt"] == "Plain prompt."


class TestSkillToolHandling:
    async def test_activate_skill_returns_instructions(
        self,
        tmp_path: Path,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
    ) -> None:
        registry = _registry_with_skill(tmp_path, body="Do the special thing.")
        channel = RealtimeVoiceChannel(
            "rt-skill",
            provider=provider,
            transport=transport,
            skills=registry,
        )
        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-skill")

        session = await channel.start_session(room.id, "user-1", "fake-ws")
        await provider.simulate_tool_call(
            session, "call-1", "activate_skill", {"name": "test-skill"}
        )
        await asyncio.sleep(0.1)

        assert len(provider.tool_results) == 1
        _sid, call_id, result_json = provider.tool_results[0]
        assert call_id == "call-1"
        result = json.loads(result_json)
        assert result["name"] == "test-skill"
        assert result["instructions"] == "Do the special thing."

    async def test_activate_unknown_skill_returns_error(
        self,
        tmp_path: Path,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
    ) -> None:
        registry = _registry_with_skill(tmp_path)
        channel = RealtimeVoiceChannel(
            "rt-skill",
            provider=provider,
            transport=transport,
            skills=registry,
        )
        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-skill")

        session = await channel.start_session(room.id, "user-1", "fake-ws")
        await provider.simulate_tool_call(
            session, "call-2", "activate_skill", {"name": "nonexistent"}
        )
        await asyncio.sleep(0.1)

        result = json.loads(provider.tool_results[0][2])
        assert "error" in result
        assert "nonexistent" in result["error"]

    async def test_read_skill_reference(
        self,
        tmp_path: Path,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
    ) -> None:
        registry = _registry_with_skill(
            tmp_path, references=[("info.md", "Reference content here.")]
        )
        channel = RealtimeVoiceChannel(
            "rt-skill",
            provider=provider,
            transport=transport,
            skills=registry,
        )
        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-skill")

        session = await channel.start_session(room.id, "user-1", "fake-ws")
        await provider.simulate_tool_call(
            session,
            "call-3",
            "read_skill_reference",
            {"skill_name": "test-skill", "filename": "info.md"},
        )
        await asyncio.sleep(0.1)

        result = json.loads(provider.tool_results[0][2])
        assert result["content"] == "Reference content here."

    async def test_skill_tool_not_passed_to_user_handler(
        self,
        tmp_path: Path,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
    ) -> None:
        """Skill infrastructure tools bypass the user tool_handler."""
        registry = _registry_with_skill(tmp_path)
        user_handler = AsyncMock(return_value='{"result": "user"}')

        channel = RealtimeVoiceChannel(
            "rt-skill",
            provider=provider,
            transport=transport,
            skills=registry,
            tool_handler=user_handler,
        )
        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-skill")

        session = await channel.start_session(room.id, "user-1", "fake-ws")
        await provider.simulate_tool_call(
            session, "call-4", "activate_skill", {"name": "test-skill"}
        )
        await asyncio.sleep(0.1)

        # User handler should NOT have been called for skill tools
        user_handler.assert_not_called()
        # But the result should still be submitted
        assert len(provider.tool_results) == 1


class TestSkillToolGating:
    async def test_gated_tools_hidden_at_session_start(
        self,
        tmp_path: Path,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
    ) -> None:
        """Tools listed in allowed_tools are excluded until skill is activated."""
        registry = _registry_with_skill(tmp_path, allowed_tools="secret_tool")

        original_connect = provider.connect
        connect_calls: list[dict[str, Any]] = []

        async def spy_connect(session: Any, **kwargs: Any) -> None:
            connect_calls.append(kwargs)
            await original_connect(session, **kwargs)

        provider.connect = spy_connect  # type: ignore[method-assign]

        user_tool = {
            "name": "secret_tool",
            "description": "A gated tool",
            "parameters": {"type": "object", "properties": {}},
        }
        channel = RealtimeVoiceChannel(
            "rt-skill",
            provider=provider,
            transport=transport,
            skills=registry,
            tools=[user_tool],
        )
        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-skill")

        await channel.start_session(room.id, "user-1", "fake-ws")

        # The tools sent to provider should NOT include the gated tool
        connect_tools = connect_calls[0]["tools"]
        tool_names = [t["name"] for t in connect_tools]
        assert "secret_tool" not in tool_names
        assert "activate_skill" in tool_names

    async def test_activation_reveals_gated_tools(
        self,
        tmp_path: Path,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
    ) -> None:
        """After activate_skill, gated tools become visible via reconfigure."""
        registry = _registry_with_skill(tmp_path, allowed_tools="secret_tool")

        user_tool = {
            "name": "secret_tool",
            "description": "A gated tool",
            "parameters": {"type": "object", "properties": {}},
        }
        channel = RealtimeVoiceChannel(
            "rt-skill",
            provider=provider,
            transport=transport,
            skills=registry,
            tools=[user_tool],
        )
        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-skill")

        session = await channel.start_session(room.id, "user-1", "fake-ws")

        # Mock reconfigure to capture the updated tools
        provider.reconfigure = AsyncMock()  # type: ignore[method-assign]

        await provider.simulate_tool_call(
            session, "call-5", "activate_skill", {"name": "test-skill"}
        )
        await asyncio.sleep(0.1)

        # reconfigure should have been called with the now-visible tools
        provider.reconfigure.assert_called_once()
        reconfig_kwargs = provider.reconfigure.call_args
        reconfig_tools = reconfig_kwargs.kwargs.get("tools", [])
        tool_names = [t["name"] for t in reconfig_tools]
        assert "secret_tool" in tool_names
        assert "activate_skill" in tool_names


class TestSessionCleanup:
    async def test_end_session_removes_activation_state(
        self,
        tmp_path: Path,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
    ) -> None:
        registry = _registry_with_skill(tmp_path)
        channel = RealtimeVoiceChannel(
            "rt-skill",
            provider=provider,
            transport=transport,
            skills=registry,
        )
        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-skill")

        session = await channel.start_session(room.id, "user-1", "fake-ws")
        session_id = session.id

        # Verify state exists
        assert session_id in channel._skill_support._activated_skills  # type: ignore[union-attr]

        await channel.end_session(session)

        # Verify state cleaned up
        assert session_id not in channel._skill_support._activated_skills  # type: ignore[union-attr]


class TestSkillToolsInjected:
    async def test_skill_tools_sent_at_session_start(
        self,
        tmp_path: Path,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
    ) -> None:
        """Skill infrastructure tools are included in provider.connect tools."""
        registry = _registry_with_skill(tmp_path)

        original_connect = provider.connect
        connect_calls: list[dict[str, Any]] = []

        async def spy_connect(session: Any, **kwargs: Any) -> None:
            connect_calls.append(kwargs)
            await original_connect(session, **kwargs)

        provider.connect = spy_connect  # type: ignore[method-assign]

        channel = RealtimeVoiceChannel(
            "rt-skill",
            provider=provider,
            transport=transport,
            skills=registry,
        )
        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-skill")

        await channel.start_session(room.id, "user-1", "fake-ws")

        tool_names = [t["name"] for t in connect_calls[0]["tools"]]
        assert "activate_skill" in tool_names
        assert "read_skill_reference" in tool_names

    async def test_no_run_script_without_executor(
        self,
        tmp_path: Path,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
    ) -> None:
        """run_skill_script tool is NOT sent without a ScriptExecutor."""
        registry = _registry_with_skill(tmp_path)

        original_connect = provider.connect
        connect_calls: list[dict[str, Any]] = []

        async def spy_connect(session: Any, **kwargs: Any) -> None:
            connect_calls.append(kwargs)
            await original_connect(session, **kwargs)

        provider.connect = spy_connect  # type: ignore[method-assign]

        channel = RealtimeVoiceChannel(
            "rt-skill",
            provider=provider,
            transport=transport,
            skills=registry,
        )
        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-skill")

        await channel.start_session(room.id, "user-1", "fake-ws")

        tool_names = [t["name"] for t in connect_calls[0]["tools"]]
        assert "run_skill_script" not in tool_names


class TestNoSkillToolDoubling:
    async def test_reconfigure_does_not_double_skill_tools(
        self,
        tmp_path: Path,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
    ) -> None:
        """Regression: reconfigure_session must not double skill tool defs."""
        registry = _registry_with_skill(tmp_path)
        user_tool = {
            "name": "my_tool",
            "description": "User tool",
            "parameters": {"type": "object", "properties": {}},
        }
        channel = RealtimeVoiceChannel(
            "rt-skill",
            provider=provider,
            transport=transport,
            skills=registry,
            tools=[user_tool],
        )
        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-skill")

        session = await channel.start_session(room.id, "user-1", "fake-ws")

        # Spy on reconfigure
        reconfig_calls: list[dict[str, Any]] = []
        original_reconfig = provider.reconfigure

        async def spy_reconfig(sess: Any, **kwargs: Any) -> None:
            reconfig_calls.append(kwargs)
            await original_reconfig(sess, **kwargs)

        provider.reconfigure = spy_reconfig  # type: ignore[method-assign]

        # Reconfigure with new tools
        new_tool = {
            "name": "new_tool",
            "description": "Replacement",
            "parameters": {"type": "object", "properties": {}},
        }
        await channel.reconfigure_session(session, tools=[new_tool])

        # Verify skill tools present exactly once
        reconfig_tools = reconfig_calls[0]["tools"]
        activate_count = sum(1 for t in reconfig_tools if t["name"] == "activate_skill")
        assert activate_count == 1

        # Now start a second session — skill tools should still appear once
        connect_calls: list[dict[str, Any]] = []
        original_connect = provider.connect

        async def spy_connect(sess: Any, **kwargs: Any) -> None:
            connect_calls.append(kwargs)
            await original_connect(sess, **kwargs)

        provider.connect = spy_connect  # type: ignore[method-assign]

        await channel.start_session(room.id, "user-2", "fake-ws-2")
        connect_tools = connect_calls[0]["tools"]
        activate_count = sum(1 for t in connect_tools if t["name"] == "activate_skill")
        assert activate_count == 1

    async def test_configure_preserves_skills_on_next_session(
        self,
        tmp_path: Path,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
    ) -> None:
        """Regression: configure(tools=...) must not wipe out skill tools."""
        registry = _registry_with_skill(tmp_path)
        channel = RealtimeVoiceChannel(
            "rt-skill",
            provider=provider,
            transport=transport,
            skills=registry,
        )
        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-skill")

        # Override tools via configure
        channel.configure(
            tools=[
                {
                    "name": "replaced_tool",
                    "description": "Replacement",
                    "parameters": {"type": "object", "properties": {}},
                }
            ]
        )

        # Spy on provider.connect
        connect_calls: list[dict[str, Any]] = []
        original_connect = provider.connect

        async def spy_connect(session: Any, **kwargs: Any) -> None:
            connect_calls.append(kwargs)
            await original_connect(session, **kwargs)

        provider.connect = spy_connect  # type: ignore[method-assign]

        await channel.start_session(room.id, "user-1", "fake-ws")

        tool_names = [t["name"] for t in connect_calls[0]["tools"]]
        assert "activate_skill" in tool_names
        assert "replaced_tool" in tool_names
