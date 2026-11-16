"""Direct skill file helpers keep their public filesystem contract (RFC §24.6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from roomkit.skills import Skill, SkillMetadata, SkillPathError, safe_join_filename


def _skill(tmp_path: Path) -> Skill:
    directory = tmp_path / "local"
    directory.mkdir()
    return Skill(SkillMetadata(name="local", description="Local files"), "Body", directory)


@pytest.mark.parametrize(
    ("requested", "stored"),
    [
        ("café.md", "café.md"),
        ("my guide.md", "my guide.md"),
        ("ｎｏｔｅｓ.md", "notes.md"),
        ("e\u0301.md", "é.md"),
    ],
)
def test_direct_file_helpers_resolve_safe_names_after_nfkc(
    tmp_path: Path, requested: str, stored: str
) -> None:
    skill = _skill(tmp_path)
    for name in ("references", "scripts"):
        directory = skill.path / name
        directory.mkdir()
        (directory / stored).write_text("Contents", encoding="utf-8")
        assert safe_join_filename(directory, requested) == (directory / stored).resolve()

    assert skill.read_reference(requested) == "Contents"
    assert skill.resolve_script(requested) == (skill.path / "scripts" / stored).resolve()


def test_missing_bundled_directories_are_empty_but_reads_raise(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    assert not skill.has_scripts
    assert not skill.has_references
    assert skill.list_scripts() == []
    assert skill.list_references() == []
    with pytest.raises(FileNotFoundError, match="Reference not found"):
        skill.read_reference("missing.md")
    with pytest.raises(FileNotFoundError, match="Script not found"):
        skill.resolve_script("missing.py")


@pytest.mark.parametrize("directory_name", ["scripts", "references"])
def test_a_bundled_directory_entry_counts_without_being_a_listed_file(
    tmp_path: Path, directory_name: str
) -> None:
    skill = _skill(tmp_path)
    directory = skill.path / directory_name
    directory.mkdir()
    listed = skill.list_scripts if directory_name == "scripts" else skill.list_references
    assert not getattr(skill, f"has_{directory_name}")
    assert listed() == []

    (directory / "nested").mkdir()

    assert getattr(skill, f"has_{directory_name}")
    assert listed() == []


def test_direct_reference_read_is_not_the_model_tool_bounded_read(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    references = skill.path / "references"
    references.mkdir()
    text = "é" * (256 * 1024)
    (references / "large.md").write_text(text, encoding="utf-8")

    assert skill.read_reference("large.md") == text


def test_direct_reference_read_preserves_an_encoding_failure(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    references = skill.path / "references"
    references.mkdir()
    (references / "invalid.md").write_bytes(b"\xff")

    with pytest.raises(UnicodeDecodeError):
        skill.read_reference("invalid.md")


def test_safe_join_can_resolve_a_file_that_does_not_exist(tmp_path: Path) -> None:
    directory = tmp_path / "missing" / "references"

    assert safe_join_filename(directory, "guide.md") == directory / "guide.md"


@pytest.mark.parametrize(
    "name",
    [
        "",
        " ",
        ".",
        "..",
        "... ",
        "a/b",
        "a\\b",
        "/etc/passwd",
        "C:guide.md",
        "．．／secret",
        "null\x00.md",
        "line\n.md",
        "control\x85.md",
    ],
)
def test_direct_file_helpers_reject_unsafe_names(tmp_path: Path, name: str) -> None:
    skill = _skill(tmp_path)
    with pytest.raises(SkillPathError):
        skill.read_reference(name)
    with pytest.raises(SkillPathError):
        skill.resolve_script(name)


def test_direct_file_helpers_reject_a_dangling_symlink_escape(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    for name in ("scripts", "references"):
        directory = skill.path / name
        directory.mkdir()
        (directory / "dangling").symlink_to(tmp_path / "absent")
    with pytest.raises(SkillPathError, match="escapes"):
        skill.read_reference("dangling")
    with pytest.raises(SkillPathError, match="escapes"):
        skill.resolve_script("dangling")


def test_lists_preserve_child_names_but_reading_still_checks_containment(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    outside = tmp_path / "outside"
    outside.write_text("Outside", encoding="utf-8")
    for name in ("scripts", "references"):
        directory = skill.path / name
        directory.mkdir()
        (directory / "escape").symlink_to(outside)
        (directory / "safe name").write_text("Safe", encoding="utf-8")
    assert skill.list_scripts() == ["escape", "safe name"]
    assert skill.list_references() == ["escape", "safe name"]
    with pytest.raises(SkillPathError):
        skill.read_reference("escape")
    with pytest.raises(SkillPathError):
        skill.resolve_script("escape")
