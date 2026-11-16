"""In-memory registration preserves the supplied skill and commits atomically."""

from __future__ import annotations

from pathlib import Path

import pytest

from roomkit.skills import Skill, SkillManifestTooLargeError, SkillMetadata, SkillRegistry


def _skill(name: str, path: Path, description: str = "Stored") -> Skill:
    return Skill(SkillMetadata(name, description, allowed_tools="calendar_*"), "Body", path)


def test_add_preserves_metadata_instructions_and_a_path_without_skill_md(tmp_path: Path) -> None:
    registry = SkillRegistry()
    skill = _skill("Store Handbook", tmp_path / "not-created", "")
    registry.add(skill)
    assert registry.get_skill(skill.name) is skill
    assert registry.get_metadata(skill.name) is skill.metadata
    assert registry.skill_names == [skill.name]
    assert not skill.path.exists()


def test_add_replaces_in_place_and_clears_both_marks_and_closed_gates(tmp_path: Path) -> None:
    registry = SkillRegistry()
    for name in ("first", "second"):
        registry.add(_skill(name, tmp_path / name))
    replacement = _skill("first", tmp_path / "replacement", "New")
    replacement.instructions = "New body"
    registry.add(replacement)
    assert registry.skill_names == ["first", "second"]
    assert registry.get_skill("first") is replacement

    registry.mark_unlisted("first")
    registry.mark_unavailable("first", "not here")
    assert registry.closed_tool_names() == {"calendar_*"}
    registry.add(replacement)
    assert registry.get_skill("first") is replacement
    assert set(registry.listed_names) == {"first", "second"}
    assert registry.unavailable_skills == {}
    assert registry.closed_tool_names() == set()


@pytest.mark.parametrize("marked", [False, True])
def test_a_refused_add_preserves_the_previous_skill_and_marks(
    tmp_path: Path, marked: bool
) -> None:
    registry = SkillRegistry()
    original = _skill("stored", tmp_path / "original")
    registry.add(original)
    if marked:
        registry.mark_unavailable("stored", "offline")
    before = registry.to_prompt_xml()
    skill_before = registry.get_skill("stored")
    gates_before = registry.closed_tool_names()

    with pytest.raises(SkillManifestTooLargeError):
        registry.add(_skill("stored", tmp_path / "rejected", "&" * 65_536))

    assert registry.to_prompt_xml() == before
    assert registry.get_skill("stored") is skill_before
    assert registry.closed_tool_names() == gates_before


def test_add_drops_the_replaced_unavailable_reason_before_checking_the_bound(
    tmp_path: Path,
) -> None:
    registry = SkillRegistry()
    registry.mark_unavailable("stored", "")
    registry.mark_unavailable("stored", "é" * (65_536 - len(registry.to_prompt_xml())))
    assert len(registry.to_prompt_xml()) == 65_536
    skill = _skill("stored", tmp_path)

    registry.add(skill)

    assert registry.get_skill("stored") is skill
    assert registry.unavailable_skills == {}
