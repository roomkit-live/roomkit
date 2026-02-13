"""Unit tests for skills parser, registry, and models."""

from __future__ import annotations

from pathlib import Path

import pytest

from roomkit.skills.models import ScriptResult
from roomkit.skills.parser import (
    SkillParseError,
    SkillValidationError,
    find_skill_md,
    parse_frontmatter,
    parse_skill,
    parse_skill_metadata,
    validate_metadata,
)
from roomkit.skills.registry import SkillRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_skill_dir(tmp_path: Path, name: str, body: str = "Do the thing.") -> Path:
    """Create a minimal valid skill directory."""
    skill_dir = tmp_path / name
    skill_dir.mkdir()
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
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
    extra_fm: str = "",
) -> Path:
    """Create a skill directory with optional scripts and references."""
    skill_dir = tmp_path / name
    skill_dir.mkdir()
    fm = f"---\nname: {name}\ndescription: Full skill\n{extra_fm}---\n{body}"
    (skill_dir / "SKILL.md").write_text(fm, encoding="utf-8")

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


# ---------------------------------------------------------------------------
# parse_frontmatter
# ---------------------------------------------------------------------------


class TestParseFrontmatter:
    def test_basic_frontmatter(self) -> None:
        content = "---\nname: my-skill\ndescription: Hello\n---\nBody text."
        data, body = parse_frontmatter(content)
        assert data["name"] == "my-skill"
        assert data["description"] == "Hello"
        assert body == "Body text."

    def test_bom_stripped(self) -> None:
        content = "\ufeff---\nname: bom-skill\ndescription: Has BOM\n---\nBody."
        data, body = parse_frontmatter(content)
        assert data["name"] == "bom-skill"
        assert body == "Body."

    def test_missing_opening_delimiter(self) -> None:
        with pytest.raises(SkillParseError, match="must start with"):
            parse_frontmatter("name: no-delimiter\n---\nBody.")

    def test_missing_closing_delimiter(self) -> None:
        with pytest.raises(SkillParseError, match="missing closing"):
            parse_frontmatter("---\nname: unclosed\ndescription: Oops\n")

    def test_multiline_body(self) -> None:
        content = "---\nname: test\ndescription: D\n---\nLine 1\n\nLine 2"
        _, body = parse_frontmatter(content)
        assert "Line 1" in body
        assert "Line 2" in body

    def test_extra_metadata_keys(self) -> None:
        content = "---\nname: x\ndescription: D\nlicense: MIT\ncustom: value\n---\nBody"
        data, _ = parse_frontmatter(content)
        assert data["license"] == "MIT"
        assert data["custom"] == "value"


# ---------------------------------------------------------------------------
# validate_metadata
# ---------------------------------------------------------------------------


class TestValidateMetadata:
    def test_valid(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        errors = validate_metadata({"name": "my-skill", "description": "Good skill"}, skill_dir)
        assert errors == []

    def test_missing_name(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "x"
        skill_dir.mkdir()
        errors = validate_metadata({"description": "No name"}, skill_dir)
        assert any("name" in e.lower() for e in errors)

    def test_invalid_name_format(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "BadName"
        skill_dir.mkdir()
        errors = validate_metadata({"name": "BadName", "description": "Bad"}, skill_dir)
        assert any("kebab-case" in e for e in errors)

    def test_name_too_long(self, tmp_path: Path) -> None:
        long_name = "a" * 65
        skill_dir = tmp_path / long_name
        skill_dir.mkdir()
        errors = validate_metadata({"name": long_name, "description": "Long"}, skill_dir)
        assert any("too long" in e.lower() for e in errors)

    def test_name_dir_mismatch(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "different-name"
        skill_dir.mkdir()
        errors = validate_metadata({"name": "wrong-name", "description": "Mismatch"}, skill_dir)
        assert any("does not match" in e for e in errors)

    def test_missing_description(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "no-desc"
        skill_dir.mkdir()
        errors = validate_metadata({"name": "no-desc"}, skill_dir)
        assert any("description" in e.lower() for e in errors)

    def test_description_too_long(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "long-desc"
        skill_dir.mkdir()
        errors = validate_metadata({"name": "long-desc", "description": "x" * 1025}, skill_dir)
        assert any("too long" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# find_skill_md
# ---------------------------------------------------------------------------


class TestFindSkillMd:
    def test_uppercase(self, tmp_path: Path) -> None:
        (tmp_path / "SKILL.md").write_text("---\n---\n", encoding="utf-8")
        assert find_skill_md(tmp_path) is not None

    def test_lowercase(self, tmp_path: Path) -> None:
        (tmp_path / "skill.md").write_text("---\n---\n", encoding="utf-8")
        assert find_skill_md(tmp_path) is not None

    def test_not_found(self, tmp_path: Path) -> None:
        assert find_skill_md(tmp_path) is None


# ---------------------------------------------------------------------------
# parse_skill_metadata / parse_skill
# ---------------------------------------------------------------------------


class TestParseSkillMetadata:
    def test_basic(self, tmp_path: Path) -> None:
        skill_dir = _make_skill_dir(tmp_path, "hello-world")
        meta = parse_skill_metadata(skill_dir)
        assert meta.name == "hello-world"
        assert meta.description == "A test skill"

    def test_no_skill_md(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(SkillParseError, match="No SKILL.md"):
            parse_skill_metadata(empty_dir)

    def test_invalid_metadata(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "bad"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: INVALID\ndescription: Bad\n---\nBody",
            encoding="utf-8",
        )
        with pytest.raises(SkillValidationError):
            parse_skill_metadata(skill_dir)

    def test_extra_metadata_preserved(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "extras"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: extras\ndescription: Has extras\n"
            "license: MIT\ncustom_key: custom_val\n---\nBody",
            encoding="utf-8",
        )
        meta = parse_skill_metadata(skill_dir)
        assert meta.license == "MIT"
        assert meta.extra_metadata["custom_key"] == "custom_val"


class TestParseSkill:
    def test_full_parse(self, tmp_path: Path) -> None:
        skill_dir = _make_skill_dir(tmp_path, "full-skill", body="# Instructions\nDo stuff.")
        skill = parse_skill(skill_dir)
        assert skill.name == "full-skill"
        assert "Instructions" in skill.instructions
        assert skill.path == skill_dir.resolve()


# ---------------------------------------------------------------------------
# Skill model
# ---------------------------------------------------------------------------


class TestSkillModel:
    def test_list_scripts(self, tmp_path: Path) -> None:
        skill_dir = _make_skill_dir_full(tmp_path, "scripted", scripts=["run.sh", "test.py"])
        skill = parse_skill(skill_dir)
        scripts = skill.list_scripts()
        assert "run.sh" in scripts
        assert "test.py" in scripts

    def test_list_references(self, tmp_path: Path) -> None:
        skill_dir = _make_skill_dir_full(
            tmp_path,
            "with-refs",
            references=[("api.md", "# API"), ("schema.json", "{}")],
        )
        skill = parse_skill(skill_dir)
        refs = skill.list_references()
        assert "api.md" in refs
        assert "schema.json" in refs

    def test_read_reference(self, tmp_path: Path) -> None:
        skill_dir = _make_skill_dir_full(
            tmp_path,
            "ref-reader",
            references=[("data.txt", "Some data")],
        )
        skill = parse_skill(skill_dir)
        content = skill.read_reference("data.txt")
        assert content == "Some data"

    def test_read_reference_traversal_blocked(self, tmp_path: Path) -> None:
        skill_dir = _make_skill_dir_full(
            tmp_path,
            "traversal",
            references=[("safe.txt", "OK")],
        )
        skill = parse_skill(skill_dir)
        with pytest.raises(ValueError, match="Invalid reference"):
            skill.read_reference("../etc/passwd")
        with pytest.raises(ValueError, match="Invalid reference"):
            skill.read_reference("sub/file.txt")

    def test_read_reference_not_found(self, tmp_path: Path) -> None:
        skill_dir = _make_skill_dir_full(
            tmp_path,
            "missing-ref",
            references=[("exists.txt", "OK")],
        )
        skill = parse_skill(skill_dir)
        with pytest.raises(FileNotFoundError, match="not found"):
            skill.read_reference("nope.txt")

    def test_has_scripts_empty(self, tmp_path: Path) -> None:
        skill_dir = _make_skill_dir(tmp_path, "no-scripts")
        skill = parse_skill(skill_dir)
        assert skill.has_scripts is False

    def test_has_references_empty(self, tmp_path: Path) -> None:
        skill_dir = _make_skill_dir(tmp_path, "no-refs")
        skill = parse_skill(skill_dir)
        assert skill.has_references is False


# ---------------------------------------------------------------------------
# ScriptResult
# ---------------------------------------------------------------------------


class TestScriptResult:
    def test_json_serialization(self) -> None:
        result = ScriptResult(exit_code=0, stdout="OK", stderr="", success=True)
        data = result.model_dump()
        assert data["exit_code"] == 0
        assert data["success"] is True

    def test_json_roundtrip(self) -> None:
        result = ScriptResult(exit_code=1, stderr="fail", success=False)
        json_str = result.model_dump_json()
        restored = ScriptResult.model_validate_json(json_str)
        assert restored.exit_code == 1
        assert restored.success is False


# ---------------------------------------------------------------------------
# SkillRegistry
# ---------------------------------------------------------------------------


class TestSkillRegistry:
    def test_discover(self, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "skill-a")
        _make_skill_dir(tmp_path, "skill-b")
        # Non-skill directory (no SKILL.md)
        (tmp_path / "not-a-skill").mkdir()

        registry = SkillRegistry()
        count = registry.discover(tmp_path)
        assert count == 2
        assert registry.skill_count == 2
        assert set(registry.skill_names) == {"skill-a", "skill-b"}

    def test_register_single(self, tmp_path: Path) -> None:
        skill_dir = _make_skill_dir(tmp_path, "single")
        registry = SkillRegistry()
        meta = registry.register(skill_dir)
        assert meta.name == "single"
        assert registry.get_metadata("single") is not None

    def test_get_skill_lazy_load(self, tmp_path: Path) -> None:
        skill_dir = _make_skill_dir(tmp_path, "lazy", body="Lazy instructions.")
        registry = SkillRegistry()
        registry.register(skill_dir)

        # First call loads
        skill = registry.get_skill("lazy")
        assert skill is not None
        assert "Lazy instructions" in skill.instructions

        # Second call returns cached
        skill2 = registry.get_skill("lazy")
        assert skill2 is skill

    def test_get_skill_not_found(self) -> None:
        registry = SkillRegistry()
        assert registry.get_skill("nonexistent") is None

    def test_get_metadata_not_found(self) -> None:
        registry = SkillRegistry()
        assert registry.get_metadata("nonexistent") is None

    def test_all_metadata(self, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "alpha")
        _make_skill_dir(tmp_path, "beta")
        registry = SkillRegistry()
        registry.discover(tmp_path)
        metas = registry.all_metadata()
        assert len(metas) == 2
        names = {m.name for m in metas}
        assert names == {"alpha", "beta"}

    def test_to_prompt_xml_empty(self) -> None:
        registry = SkillRegistry()
        assert registry.to_prompt_xml() == ""

    def test_to_prompt_xml(self, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "xml-test")
        registry = SkillRegistry()
        registry.discover(tmp_path)
        xml = registry.to_prompt_xml()
        assert "<available_skills>" in xml
        assert "</available_skills>" in xml
        assert 'name="xml-test"' in xml
        assert "<description>" in xml

    def test_to_prompt_xml_escapes_html(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "esc-test"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            '---\nname: esc-test\ndescription: Has <b>bold</b> & "quotes"\n---\nBody',
            encoding="utf-8",
        )
        registry = SkillRegistry()
        registry.register(skill_dir)
        xml = registry.to_prompt_xml()
        assert "&lt;b&gt;" in xml
        assert "&amp;" in xml

    def test_discover_skips_invalid(self, tmp_path: Path) -> None:
        """Invalid skills are warned and skipped."""
        # Valid skill
        _make_skill_dir(tmp_path, "valid-one")
        # Invalid skill (bad name)
        bad_dir = tmp_path / "BadName"
        bad_dir.mkdir()
        (bad_dir / "SKILL.md").write_text(
            "---\nname: BadName\ndescription: Bad\n---\nBody",
            encoding="utf-8",
        )
        registry = SkillRegistry()
        count = registry.discover(tmp_path)
        assert count == 1
        assert registry.skill_count == 1

    def test_discover_nonexistent_dir(self) -> None:
        """Non-existent directories are warned and skipped."""
        registry = SkillRegistry()
        count = registry.discover("/nonexistent/path")
        assert count == 0

    def test_re_register_invalidates_cache(self, tmp_path: Path) -> None:
        skill_dir = _make_skill_dir(tmp_path, "cached", body="v1")
        registry = SkillRegistry()
        registry.register(skill_dir)
        skill = registry.get_skill("cached")
        assert skill is not None
        assert "v1" in skill.instructions

        # Update and re-register
        (skill_dir / "SKILL.md").write_text(
            "---\nname: cached\ndescription: Updated\n---\nv2",
            encoding="utf-8",
        )
        registry.register(skill_dir)
        skill2 = registry.get_skill("cached")
        assert skill2 is not None
        assert "v2" in skill2.instructions
