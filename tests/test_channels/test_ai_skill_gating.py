"""Integration tests for AIChannel tool policy and skill gating."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

from roomkit.channels.ai import (
    _TOOL_ACTIVATE_SKILL,
    _TOOL_READ_REFERENCE,
    AIChannel,
    _current_loop_ctx,
    _ToolLoopContext,
)
from roomkit.models.channel import ChannelBinding, ChannelCapabilities
from roomkit.models.context import RoomContext
from roomkit.models.enums import (
    ChannelCategory,
    ChannelDirection,
    ChannelMediaType,
    ChannelType,
    ParticipantRole,
)
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.participant import Participant
from roomkit.models.room import Room
from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIResponse,
    AIToolCall,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.skills.models import SkillMetadata
from roomkit.skills.registry import SkillRegistry
from roomkit.tools.policy import RoleOverride, ToolPolicy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_binding(tools: list[dict] | None = None) -> ChannelBinding:
    return ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        direction=ChannelDirection.BIDIRECTIONAL,
        capabilities=ChannelCapabilities(media_types=[ChannelMediaType.TEXT]),
        metadata={"tools": tools or []},
    )


def _make_context() -> RoomContext:
    room = Room(id="r1")
    return RoomContext(
        room=room,
        bindings=[_make_binding()],
    )


def _make_event() -> RoomEvent:
    return RoomEvent(
        room_id="r1",
        source=EventSource(channel_id="user", channel_type=ChannelType.SMS, provider="mock"),
        content=TextContent(body="hello"),
    )


def _skill_registry(
    *skill_defs: tuple[str, str, str | None],
) -> SkillRegistry:
    """Create a SkillRegistry from (name, description, allowed_tools) tuples.

    Directly populates registry internals to avoid needing real SKILL.md files.
    """
    from roomkit.skills.models import Skill

    registry = SkillRegistry()
    for name, desc, allowed in skill_defs:
        meta = SkillMetadata(name=name, description=desc, allowed_tools=allowed)
        skill = Skill(metadata=meta, instructions=f"Instructions for {name}", path=Path("/tmp"))
        registry._metadata[name] = meta
        registry._skills[name] = skill
        registry._paths[name] = Path("/tmp")
    return registry


def _tool_response(tool_name: str = "search") -> AIResponse:
    return AIResponse(
        content="",
        tool_calls=[AIToolCall(id="tc1", name=tool_name, arguments={"q": "test"})],
    )


def _final_response(content: str = "Done") -> AIResponse:
    return AIResponse(content=content, tool_calls=[])


def _set_loop_ctx(**kwargs: object) -> _ToolLoopContext:
    """Set a _ToolLoopContext on the contextvar and return it."""
    ctx = _ToolLoopContext(**kwargs)  # type: ignore[arg-type]
    _current_loop_ctx.set(ctx)
    return ctx


def _clear_loop_ctx() -> None:
    """Clear the contextvar."""
    _current_loop_ctx.set(None)


# ---------------------------------------------------------------------------
# SkillMetadata.gated_tool_names
# ---------------------------------------------------------------------------


class TestGatedToolNames:
    def test_none_returns_empty(self) -> None:
        meta = SkillMetadata(name="s", description="d", allowed_tools=None)
        assert meta.gated_tool_names == []

    def test_empty_string_returns_empty(self) -> None:
        meta = SkillMetadata(name="s", description="d", allowed_tools="")
        assert meta.gated_tool_names == []

    def test_single_tool(self) -> None:
        meta = SkillMetadata(name="s", description="d", allowed_tools="my_tool")
        assert meta.gated_tool_names == ["my_tool"]

    def test_comma_separated(self) -> None:
        meta = SkillMetadata(name="s", description="d", allowed_tools="a, b, c")
        assert meta.gated_tool_names == ["a", "b", "c"]

    def test_whitespace_stripped(self) -> None:
        meta = SkillMetadata(name="s", description="d", allowed_tools="  x , y ,  ")
        assert meta.gated_tool_names == ["x", "y"]


# ---------------------------------------------------------------------------
# ToolPolicy visibility filter
# ---------------------------------------------------------------------------


class TestPolicyVisibilityFilter:
    async def test_denied_tools_hidden_from_context(self) -> None:
        """Tools matching deny patterns should not appear in context.tools."""
        provider = MockAIProvider(ai_responses=[_final_response()])
        policy = ToolPolicy(deny=["search"])
        ch = AIChannel("ai1", provider=provider, tool_handler=AsyncMock(), tool_policy=policy)
        binding = _make_binding(
            tools=[
                {"name": "search", "description": "Search"},
                {"name": "read", "description": "Read"},
            ]
        )
        _set_loop_ctx()
        try:
            ctx = await ch._build_context(_make_event(), binding, _make_context())
            tool_names = [t.name for t in ctx.tools]
            assert "search" not in tool_names
            assert "read" in tool_names
        finally:
            _clear_loop_ctx()

    async def test_allowed_tools_pass(self) -> None:
        provider = MockAIProvider(ai_responses=[_final_response()])
        policy = ToolPolicy(allow=["read_*"])
        ch = AIChannel("ai1", provider=provider, tool_handler=AsyncMock(), tool_policy=policy)
        binding = _make_binding(
            tools=[
                {"name": "read_file", "description": "Read"},
                {"name": "delete_file", "description": "Delete"},
            ]
        )
        _set_loop_ctx()
        try:
            ctx = await ch._build_context(_make_event(), binding, _make_context())
            tool_names = [t.name for t in ctx.tools]
            assert "read_file" in tool_names
            assert "delete_file" not in tool_names
        finally:
            _clear_loop_ctx()

    async def test_empty_policy_no_filtering(self) -> None:
        provider = MockAIProvider(ai_responses=[_final_response()])
        ch = AIChannel("ai1", provider=provider, tool_handler=AsyncMock(), tool_policy=None)
        binding = _make_binding(
            tools=[
                {"name": "search", "description": "Search"},
                {"name": "delete", "description": "Delete"},
            ]
        )
        _set_loop_ctx()
        try:
            ctx = await ch._build_context(_make_event(), binding, _make_context())
            tool_names = [t.name for t in ctx.tools]
            assert "search" in tool_names
            assert "delete" in tool_names
        finally:
            _clear_loop_ctx()

    async def test_skill_infra_tools_never_filtered(self) -> None:
        """activate_skill and read_skill_reference must always be visible."""
        provider = MockAIProvider(ai_responses=[_final_response()])
        policy = ToolPolicy(deny=["*"])  # Deny everything
        registry = _skill_registry(("s1", "Skill 1", None))
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=AsyncMock(),
            tool_policy=policy,
            skills=registry,
        )
        binding = _make_binding(tools=[{"name": "search", "description": "Search"}])
        _set_loop_ctx()
        try:
            ctx = await ch._build_context(_make_event(), binding, _make_context())
            tool_names = [t.name for t in ctx.tools]
            assert _TOOL_ACTIVATE_SKILL in tool_names
            assert _TOOL_READ_REFERENCE in tool_names
            assert "search" not in tool_names
        finally:
            _clear_loop_ctx()


# ---------------------------------------------------------------------------
# Policy execution guard
# ---------------------------------------------------------------------------


class TestPolicyExecutionGuard:
    async def test_denied_tool_blocked_at_execution(self) -> None:
        """Defense-in-depth: even if a tool call sneaks past visibility, execution blocks it."""
        provider = MockAIProvider(
            ai_responses=[
                _tool_response("forbidden_tool"),
                _final_response(),
            ]
        )
        handler = AsyncMock(return_value="ok")
        policy = ToolPolicy(deny=["forbidden_*"])
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=handler,
            tool_policy=policy,
            max_tool_rounds=5,
        )
        context = AIContext(messages=[AIMessage(role="user", content="go")])
        await ch._run_tool_loop(context)

        # The handler should NOT have been called for the forbidden tool
        handler.assert_not_called()

    async def test_allowed_tool_executes(self) -> None:
        provider = MockAIProvider(
            ai_responses=[
                _tool_response("safe_tool"),
                _final_response(),
            ]
        )
        handler = AsyncMock(return_value="ok")
        policy = ToolPolicy(allow=["safe_*"])
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=handler,
            tool_policy=policy,
            max_tool_rounds=5,
        )
        context = AIContext(messages=[AIMessage(role="user", content="go")])
        await ch._run_tool_loop(context)

        handler.assert_called_once_with("safe_tool", {"q": "test"})


# ---------------------------------------------------------------------------
# Skill gating visibility
# ---------------------------------------------------------------------------


class TestSkillGatingVisibility:
    async def test_gated_tools_hidden_before_activation(self) -> None:
        provider = MockAIProvider(ai_responses=[_final_response()])
        registry = _skill_registry(("analytics", "Analytics skill", "run_query,export_csv"))
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=AsyncMock(),
            skills=registry,
        )
        binding = _make_binding(
            tools=[
                {"name": "run_query", "description": "Run a query"},
                {"name": "export_csv", "description": "Export CSV"},
                {"name": "search", "description": "Search"},
            ]
        )
        _set_loop_ctx()
        try:
            ctx = await ch._build_context(_make_event(), binding, _make_context())
            tool_names = [t.name for t in ctx.tools]
            assert "run_query" not in tool_names
            assert "export_csv" not in tool_names
            assert "search" in tool_names
        finally:
            _clear_loop_ctx()

    async def test_gated_tools_visible_after_activation(self) -> None:
        provider = MockAIProvider(ai_responses=[_final_response()])
        registry = _skill_registry(("analytics", "Analytics skill", "run_query,export_csv"))
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=AsyncMock(),
            skills=registry,
        )

        # Simulate activation via contextvar
        _set_loop_ctx(activated_skills={"analytics"})
        try:
            binding = _make_binding(
                tools=[
                    {"name": "run_query", "description": "Run a query"},
                    {"name": "export_csv", "description": "Export CSV"},
                ]
            )
            ctx = await ch._build_context(_make_event(), binding, _make_context())
            tool_names = [t.name for t in ctx.tools]
            assert "run_query" in tool_names
            assert "export_csv" in tool_names
        finally:
            _clear_loop_ctx()

    async def test_non_gated_tools_always_visible(self) -> None:
        provider = MockAIProvider(ai_responses=[_final_response()])
        registry = _skill_registry(("analytics", "Analytics skill", "special_tool"))
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=AsyncMock(),
            skills=registry,
        )
        binding = _make_binding(
            tools=[
                {"name": "search", "description": "Search"},
                {"name": "special_tool", "description": "Gated"},
            ]
        )
        _set_loop_ctx()
        try:
            ctx = await ch._build_context(_make_event(), binding, _make_context())
            tool_names = [t.name for t in ctx.tools]
            assert "search" in tool_names
            assert "special_tool" not in tool_names  # gated, not activated
        finally:
            _clear_loop_ctx()


class TestSkillGatingExecution:
    async def test_gated_tool_blocked_at_execution(self) -> None:
        """Gated tool execution returns helpful error before activation."""
        provider = MockAIProvider(
            ai_responses=[
                _tool_response("run_query"),
                _final_response(),
            ]
        )
        handler = AsyncMock(return_value="ok")
        registry = _skill_registry(("analytics", "Analytics skill", "run_query"))
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=handler,
            skills=registry,
            max_tool_rounds=5,
        )
        context = AIContext(messages=[AIMessage(role="user", content="go")])
        await ch._run_tool_loop(context)

        # Handler not called — gating blocked it
        handler.assert_not_called()

    async def test_gated_tool_allowed_after_activation(self) -> None:
        """After activate_skill within the same tool loop, gated tools execute."""
        # Sequence: activate_skill("analytics") -> run_query
        provider = MockAIProvider(
            ai_responses=[
                # Round 1: model calls activate_skill
                AIResponse(
                    content="",
                    tool_calls=[
                        AIToolCall(
                            id="tc_activate",
                            name="activate_skill",
                            arguments={"name": "analytics"},
                        )
                    ],
                ),
                # Round 2: model calls the gated tool (now ungated)
                _tool_response("run_query"),
                _final_response(),
            ]
        )
        handler = AsyncMock(return_value="ok")
        registry = _skill_registry(("analytics", "Analytics skill", "run_query"))
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=handler,
            skills=registry,
            max_tool_rounds=10,
        )
        context = AIContext(messages=[AIMessage(role="user", content="go")])
        await ch._run_tool_loop(context)

        # The skill-aware handler dispatched activate_skill internally,
        # then run_query was forwarded to the user handler.
        handler.assert_called_once_with("run_query", {"q": "test"})


# ---------------------------------------------------------------------------
# Activation tracking
# ---------------------------------------------------------------------------


class TestActivationTracking:
    async def test_activation_resets_per_tool_loop(self) -> None:
        """activated_skills starts fresh for each tool loop invocation."""
        provider = MockAIProvider(ai_responses=[_final_response()])
        registry = _skill_registry(("s1", "Skill 1", "t1"))
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=AsyncMock(return_value="ok"),
            skills=registry,
            max_tool_rounds=5,
        )
        # Set an activation on a parent context — the tool loop creates its own
        _set_loop_ctx(activated_skills={"s1"})
        try:
            context = AIContext(messages=[AIMessage(role="user", content="go")])
            # _run_tool_loop creates a fresh _ToolLoopContext, so "s1" is NOT inherited
            await ch._run_tool_loop(context)
            # After tool loop, active_loops should be empty
            assert len(ch._active_loops) == 0
        finally:
            _clear_loop_ctx()

    async def test_handle_activate_skill_tracks_name(self) -> None:
        """_handle_activate_skill adds the skill name to activated_skills."""
        provider = MockAIProvider(ai_responses=[_final_response()])
        registry = _skill_registry(("analytics", "Analytics skill", "run_query"))
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=AsyncMock(),
            skills=registry,
        )
        loop_ctx = _set_loop_ctx()
        try:
            result = await ch._handle_activate_skill({"name": "analytics"})
            parsed = json.loads(result)
            assert parsed["name"] == "analytics"
            assert "analytics" in loop_ctx.activated_skills
        finally:
            _clear_loop_ctx()


class TestMultipleSkillGating:
    async def test_activating_one_skill_does_not_ungate_another(self) -> None:
        provider = MockAIProvider(ai_responses=[_final_response()])
        registry = _skill_registry(
            ("analytics", "Analytics", "run_query"),
            ("export", "Export", "export_csv"),
        )
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=AsyncMock(),
            skills=registry,
        )
        # Activate only analytics
        _set_loop_ctx(activated_skills={"analytics"})
        try:
            binding = _make_binding(
                tools=[
                    {"name": "run_query", "description": "Query"},
                    {"name": "export_csv", "description": "Export"},
                ]
            )
            ctx = await ch._build_context(_make_event(), binding, _make_context())
            tool_names = [t.name for t in ctx.tools]
            assert "run_query" in tool_names
            assert "export_csv" not in tool_names  # export skill not activated
        finally:
            _clear_loop_ctx()

    async def test_both_skills_activated(self) -> None:
        provider = MockAIProvider(ai_responses=[_final_response()])
        registry = _skill_registry(
            ("analytics", "Analytics", "run_query"),
            ("export", "Export", "export_csv"),
        )
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=AsyncMock(),
            skills=registry,
        )
        _set_loop_ctx(activated_skills={"analytics", "export"})
        try:
            binding = _make_binding(
                tools=[
                    {"name": "run_query", "description": "Query"},
                    {"name": "export_csv", "description": "Export"},
                ]
            )
            ctx = await ch._build_context(_make_event(), binding, _make_context())
            tool_names = [t.name for t in ctx.tools]
            assert "run_query" in tool_names
            assert "export_csv" in tool_names
        finally:
            _clear_loop_ctx()


# ---------------------------------------------------------------------------
# Combined policy + gating
# ---------------------------------------------------------------------------


class TestPolicyPlusGating:
    async def test_policy_deny_overrides_skill_activation(self) -> None:
        """Even if a skill is activated, policy deny still blocks the tool."""
        provider = MockAIProvider(ai_responses=[_final_response()])
        registry = _skill_registry(("analytics", "Analytics", "run_query"))
        policy = ToolPolicy(deny=["run_query"])
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=AsyncMock(),
            skills=registry,
            tool_policy=policy,
        )
        _set_loop_ctx(activated_skills={"analytics"})
        try:
            binding = _make_binding(tools=[{"name": "run_query", "description": "Query"}])
            ctx = await ch._build_context(_make_event(), binding, _make_context())
            tool_names = [t.name for t in ctx.tools]
            assert "run_query" not in tool_names
        finally:
            _clear_loop_ctx()


# ---------------------------------------------------------------------------
# Role-based tool policy
# ---------------------------------------------------------------------------


def _make_event_with_participant(participant_id: str = "p1") -> RoomEvent:
    return RoomEvent(
        room_id="r1",
        source=EventSource(
            channel_id="user",
            channel_type=ChannelType.SMS,
            provider="mock",
            participant_id=participant_id,
        ),
        content=TextContent(body="hello"),
    )


def _make_context_with_participants(
    participants: list[Participant] | None = None,
) -> RoomContext:
    room = Room(id="r1")
    return RoomContext(
        room=room,
        bindings=[_make_binding()],
        participants=participants or [],
    )


class TestRoleBasedPolicy:
    async def test_observer_restricted_by_role_override(self) -> None:
        """Observer role override restricts available tools.

        We set up the loop context with the resolved participant role and
        verify that _build_context correctly filters tools.
        """
        provider = MockAIProvider(ai_responses=[_final_response()])
        policy = ToolPolicy(
            allow=["search", "write_file", "read_file"],
            role_overrides={
                "observer": RoleOverride(allow=["search", "read_file"]),
            },
        )
        ch = AIChannel("ai1", provider=provider, tool_handler=AsyncMock(), tool_policy=policy)
        binding = _make_binding(
            tools=[
                {"name": "search", "description": "Search"},
                {"name": "write_file", "description": "Write"},
                {"name": "read_file", "description": "Read"},
            ]
        )
        event = _make_event_with_participant("p_obs")
        context = _make_context_with_participants(
            [
                Participant(
                    id="p_obs", room_id="r1", channel_id="user", role=ParticipantRole.OBSERVER
                ),
            ]
        )

        # Simulate what on_event does: resolve role and set contextvar
        role = ch._resolve_participant_role(event, context)
        assert role == ParticipantRole.OBSERVER
        _set_loop_ctx(current_participant_role=role)
        try:
            ctx = await ch._build_context(event, binding, context)
            tool_names = [t.name for t in ctx.tools]
            assert "search" in tool_names
            assert "read_file" in tool_names
            assert "write_file" not in tool_names
        finally:
            _clear_loop_ctx()

    async def test_no_participant_match_uses_base_policy(self) -> None:
        """When participant_id doesn't match any participant, base policy is used."""
        provider = MockAIProvider(ai_responses=[_final_response()])
        policy = ToolPolicy(
            allow=["search", "write_file"],
            role_overrides={
                "observer": RoleOverride(allow=["search"]),
            },
        )
        ch = AIChannel("ai1", provider=provider, tool_handler=AsyncMock(), tool_policy=policy)
        binding = _make_binding(
            tools=[
                {"name": "search", "description": "Search"},
                {"name": "write_file", "description": "Write"},
            ]
        )
        event = _make_event_with_participant("unknown_pid")
        context = _make_context_with_participants(
            [
                Participant(
                    id="p_obs", room_id="r1", channel_id="user", role=ParticipantRole.OBSERVER
                ),
            ]
        )

        # Resolve role — unknown participant gives None
        role = ch._resolve_participant_role(event, context)
        assert role is None
        _set_loop_ctx(current_participant_role=role)
        try:
            ctx = await ch._build_context(event, binding, context)
            tool_names = [t.name for t in ctx.tools]
            assert "search" in tool_names
            assert "write_file" in tool_names  # base policy allows it
        finally:
            _clear_loop_ctx()

    async def test_replace_mode_owner_gets_full_access(self) -> None:
        """Owner with replace mode gets unrestricted access."""
        provider = MockAIProvider(ai_responses=[_final_response()])
        policy = ToolPolicy(
            allow=["search"],
            deny=["admin_*"],
            role_overrides={
                "owner": RoleOverride(mode="replace"),  # empty allow+deny = permit all
            },
        )
        ch = AIChannel("ai1", provider=provider, tool_handler=AsyncMock(), tool_policy=policy)
        binding = _make_binding(
            tools=[
                {"name": "search", "description": "Search"},
                {"name": "admin_panel", "description": "Admin"},
                {"name": "delete_all", "description": "Delete"},
            ]
        )
        event = _make_event_with_participant("p_own")
        context = _make_context_with_participants(
            [
                Participant(
                    id="p_own", room_id="r1", channel_id="user", role=ParticipantRole.OWNER
                ),
            ]
        )

        role = ch._resolve_participant_role(event, context)
        _set_loop_ctx(current_participant_role=role)
        try:
            ctx = await ch._build_context(event, binding, context)
            tool_names = [t.name for t in ctx.tools]
            assert "search" in tool_names
            assert "admin_panel" in tool_names
            assert "delete_all" in tool_names
        finally:
            _clear_loop_ctx()
