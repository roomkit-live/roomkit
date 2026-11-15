"""Registry changes keep the escaped manifest bounded and atomic (RFC §24.5)."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.test_skills import _make_skill_dir_full

from roomkit.skills import SkillManifestTooLargeError, SkillRegistry
from roomkit.skills.models import Skill, SkillMetadata

LIMIT = 65_536


def _reason_at_limit(name: str, prefix: str = "") -> str:
    registry = SkillRegistry()
    registry.mark_unavailable(name, prefix)
    return prefix + "é" * (LIMIT - len(registry.to_prompt_xml()))


@pytest.mark.parametrize("prefix", ["", "<&\"'😀"])
def test_an_unavailable_reason_counts_escaped_characters_at_the_boundary(prefix: str) -> None:
    registry = SkillRegistry()
    name = 'missing<&"'
    reason = _reason_at_limit(name, prefix)
    registry.mark_unavailable(name, reason)
    before = registry.to_prompt_xml()
    assert len(before) == LIMIT
    assert len(before.encode()) > LIMIT  # The contract counts characters, not UTF-8 bytes.

    with pytest.raises(SkillManifestTooLargeError) as raised:
        registry.mark_unavailable(name, reason + "x")

    assert (raised.value.characters, raised.value.limit) == (LIMIT + 1, LIMIT)
    assert registry.to_prompt_xml() == before
    assert registry.get_unavailable_reason(name) == reason
    registry.mark_unavailable(name, "short")
    assert registry.get_unavailable_reason(name) == "short"
    assert len(registry.to_prompt_xml()) < LIMIT


async def test_a_rejected_mark_keeps_activation_and_tool_execution_working(
    tmp_path: Path, streaming: bool
) -> None:
    from tests.conftest import make_event
    from tests.tool_loop_modes import respond

    from roomkit.channels.ai import AIChannel
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.context import RoomContext
    from roomkit.models.enums import ChannelCategory, ChannelType
    from roomkit.models.room import Room
    from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
    from roomkit.providers.ai.mock import MockAIProvider

    registry = SkillRegistry()
    registry.register(_make_skill_dir_full(tmp_path, "cal", extra_fm="allowed_tools: calendar\n"))
    with pytest.raises(SkillManifestTooLargeError):
        registry.mark_unavailable("cal", "x" * LIMIT)
    executed: list[str] = []

    async def handle(name: str, arguments: dict[str, object]) -> str:
        executed.append(name)
        return "ran"

    calls = [
        AIToolCall(id="before", name="calendar", arguments={}),
        AIToolCall(id="activate", name="activate_skill", arguments={"name": "cal"}),
        AIToolCall(id="after", name="calendar", arguments={}),
    ]
    provider = MockAIProvider(
        streaming=streaming,
        ai_responses=[
            *(
                AIResponse(content="", tool_calls=[call], finish_reason="tool_calls")
                for call in calls
            ),
            AIResponse(content="done"),
        ],
    )
    channel = AIChannel(
        "ai",
        provider=provider,
        skills=registry,
        tool_search=False,
        tools=[AITool(name="calendar", description="Run", parameters={"type": "object"})],
        tool_handler=handle,
    )
    binding = ChannelBinding(
        channel_id="ai",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )
    result = await respond(
        channel, make_event(room_id="r1", body="Go"), binding, RoomContext(room=Room(id="r1"))
    )
    assert result.text == "done"
    assert [call.failed for call in result.calls] == [True, False, False]
    assert "Activate the skill first" in str(result.calls[0].result)
    assert result.calls[2].result == "ran"
    assert executed == ["calendar"]


@pytest.mark.parametrize("unlisted", [False, True])
def test_a_refused_mark_preserves_the_available_skill_and_its_gates(
    tmp_path: Path, unlisted: bool
) -> None:
    path = _make_skill_dir_full(tmp_path, "cal", extra_fm="allowed_tools: calendar_*\n")
    registry = SkillRegistry()
    registry.register(path)
    if unlisted:
        registry.mark_unlisted("cal")
    loaded = registry.get_skill("cal")
    metadata = registry.get_metadata("cal")
    before = registry.to_prompt_xml()

    with pytest.raises(SkillManifestTooLargeError):
        registry.mark_unavailable("cal", "x" * LIMIT)

    assert registry.to_prompt_xml() == before
    assert registry.get_skill("cal") is loaded
    assert registry.get_metadata("cal") is metadata
    assert registry.listed_names == ([] if unlisted else ["cal"])
    assert registry.skill_names == ["cal"]
    assert registry.unavailable_skills == {}
    assert registry.gated_tool_names() == {"calendar_*"}
    assert registry.closed_tool_names() == set()


def test_a_refused_reason_update_preserves_order_and_closed_gates(tmp_path: Path) -> None:
    registry = SkillRegistry()
    path = _make_skill_dir_full(tmp_path, "cal", extra_fm="allowed_tools: calendar_*\n")
    registry.register(path)
    registry.mark_unavailable("cal", "first")
    registry.mark_unavailable("other", "second")
    before = registry.to_prompt_xml()

    for name in ("cal", "new"):
        with pytest.raises(SkillManifestTooLargeError):
            registry.mark_unavailable(name, "&" * 14_000)
        assert registry.to_prompt_xml() == before
        assert registry.unavailable_skills == {"cal": "first", "other": "second"}
        assert registry.closed_tool_names() == {"calendar_*"}
        assert registry.get_skill("cal") is None


def test_copy_cannot_relist_an_oversized_hidden_catalogue(tmp_path: Path) -> None:
    registry = SkillRegistry()
    for i in range(70):
        name = f"skill-{i:02}"
        registry.add(Skill(SkillMetadata(name=name, description="d" * 1024), "body", tmp_path))
        registry.mark_unlisted(name)
    assert registry.to_prompt_xml() == ""

    with pytest.raises(SkillManifestTooLargeError) as raised:
        registry.copy(marks=False)

    assert raised.value.characters > LIMIT
    assert registry.to_prompt_xml() == ""
    assert registry.listed_names == []
    assert registry.copy().to_prompt_xml() == ""
    selected = registry.copy(["skill-00"], marks=False)
    assert selected.listed_names == ["skill-00"]
    assert len(selected.to_prompt_xml()) < LIMIT
    assert selected.get_skill("skill-00") is registry.get_skill("skill-00")


def test_a_replacement_drops_the_old_reason_before_checking_its_size(tmp_path: Path) -> None:
    registry = SkillRegistry()
    registry.mark_unavailable("cal", _reason_at_limit("cal"))
    path = _make_skill_dir_full(tmp_path, "cal", extra_fm="allowed_tools: calendar_*\n")
    registry.register(path)
    assert registry.get_unavailable_reason("cal") is None
    assert registry.skill_names == ["cal"]
    assert registry.closed_tool_names() == set()
    assert len(registry.to_prompt_xml()) < LIMIT


@pytest.mark.parametrize("repair", ["hide", "replace", "copy"])
def test_aliased_metadata_cannot_render_an_oversized_manifest_but_can_be_repaired(
    tmp_path: Path, repair: str
) -> None:
    registry = SkillRegistry()
    skill = Skill(SkillMetadata(name="cal", description="small"), "body", tmp_path)
    registry.add(skill)
    registry.add(Skill(SkillMetadata(name="other", description="small"), "other", tmp_path))
    copied = registry.copy()
    assert registry.get_metadata("cal") is skill.metadata
    skill.metadata.description = "x" * LIMIT

    with pytest.raises(SkillManifestTooLargeError):
        registry.to_prompt_xml()
    with pytest.raises(SkillManifestTooLargeError):
        copied.to_prompt_xml()

    assert registry.get_metadata("cal") is skill.metadata
    if repair == "hide":
        registry.mark_unlisted("cal")
    elif repair == "replace":
        registry.add(Skill(SkillMetadata(name="cal", description="small"), "new", tmp_path))
    else:
        registry = registry.copy(["other"])
    assert len(registry.to_prompt_xml()) < LIMIT


@pytest.mark.parametrize("operation", ["mark", "mcp"])
async def test_preflights_use_the_renderers_visibility_after_an_aliased_rename(
    tmp_path: Path, operation: str
) -> None:
    from tests.mcp_skill_fakes import FakeSkillServer

    registry = SkillRegistry()
    skill = Skill(SkillMetadata(name="hidden", description="small"), "body", tmp_path)
    registry.add(skill)
    registry.mark_unlisted("hidden")
    skill.metadata.name = "renamed"
    skill.metadata.description = "x" * LIMIT
    with pytest.raises(SkillManifestTooLargeError):
        if operation == "mark":
            registry.mark_unavailable("other", "short")
        else:
            await registry.add_mcp_server(FakeSkillServer(), "hf")
    assert registry.unavailable_skills == {}
    assert not registry.has_served
    assert registry.get_metadata("hidden") is skill.metadata
