"""Unit tests for skills parser, registry, and models."""

from __future__ import annotations

from pathlib import Path

import pytest

from roomkit.skills.errors import (
    SkillDiscoveryError,
    SkillError,
    SkillParseError,
    SkillPathError,
    SkillValidationError,
)
from roomkit.skills.models import ScriptResult
from roomkit.skills.parser import (
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


def _make_invalid_skill_dir(tmp_path: Path, dirname: str) -> Path:
    """Create a skill directory whose frontmatter fails validation.

    The declared name is neither kebab-case nor equal to the directory name, so
    it fails regardless of what the directory is called — letting callers pick a
    directory name purely to control discovery order.
    """
    skill_dir = tmp_path / dirname
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: NotKebabCase\ndescription: Bad\n---\nBody",
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

    def test_read_reference_symlink_escape_blocked(self, tmp_path: Path) -> None:
        """A symlink planted in references/ cannot serve a file from outside it.

        The name alone is clean — no "..", no separator — so only resolving the
        link and re-checking containment catches this.
        """
        secret = tmp_path / "secret.txt"
        secret.write_text("classified", encoding="utf-8")
        skill_dir = _make_skill_dir_full(
            tmp_path,
            "symlinked",
            references=[("safe.txt", "OK")],
        )
        (skill_dir / "references" / "notes.md").symlink_to(secret)

        skill = parse_skill(skill_dir)
        with pytest.raises(SkillPathError, match="escapes"):
            skill.read_reference("notes.md")

    def test_read_reference_rejects_fullwidth_traversal(self, tmp_path: Path) -> None:
        """Full-width look-alikes are normalised before the separator check."""
        skill_dir = _make_skill_dir_full(
            tmp_path,
            "fullwidth",
            references=[("safe.txt", "OK")],
        )
        skill = parse_skill(skill_dir)
        with pytest.raises(SkillPathError):
            skill.read_reference("．．／secret.txt")

    def test_read_reference_rejects_absolute(self, tmp_path: Path) -> None:
        skill_dir = _make_skill_dir_full(
            tmp_path,
            "absolute",
            references=[("safe.txt", "OK")],
        )
        skill = parse_skill(skill_dir)
        with pytest.raises(SkillPathError, match="relative"):
            skill.read_reference("/etc/passwd")

    def test_read_reference_path_error_is_value_error(self, tmp_path: Path) -> None:
        """Integrators catching ValueError keep working."""
        skill_dir = _make_skill_dir_full(tmp_path, "compat", references=[("safe.txt", "OK")])
        skill = parse_skill(skill_dir)
        with pytest.raises(ValueError):
            skill.read_reference("../etc/passwd")

    def test_resolve_script_returns_path(self, tmp_path: Path) -> None:
        skill_dir = _make_skill_dir_full(tmp_path, "runner", scripts=["run.sh"])
        skill = parse_skill(skill_dir)
        assert skill.resolve_script("run.sh") == (skill_dir / "scripts" / "run.sh").resolve()

    def test_resolve_script_symlink_escape_blocked(self, tmp_path: Path) -> None:
        """The same containment applies to what an executor is asked to run."""
        outside = tmp_path / "payload.sh"
        outside.write_text("#!/bin/sh\necho pwned", encoding="utf-8")
        skill_dir = _make_skill_dir_full(tmp_path, "sneaky", scripts=["ok.sh"])
        (skill_dir / "scripts" / "innocent.sh").symlink_to(outside)

        skill = parse_skill(skill_dir)
        with pytest.raises(SkillPathError, match="escapes"):
            skill.resolve_script("innocent.sh")

    def test_resolve_script_traversal_blocked(self, tmp_path: Path) -> None:
        skill_dir = _make_skill_dir_full(tmp_path, "guarded", scripts=["ok.sh"])
        skill = parse_skill(skill_dir)
        with pytest.raises(SkillPathError):
            skill.resolve_script("../../bin/sh")

    def test_resolve_script_missing(self, tmp_path: Path) -> None:
        skill_dir = _make_skill_dir_full(tmp_path, "empty-scripts", scripts=["ok.sh"])
        skill = parse_skill(skill_dir)
        with pytest.raises(FileNotFoundError):
            skill.resolve_script("nope.sh")

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

    def test_mark_unavailable_removes_from_available(self, tmp_path: Path) -> None:
        skill_dir = _make_skill_dir(tmp_path, "gated")
        registry = SkillRegistry()
        registry.register(skill_dir)

        registry.mark_unavailable("gated", "requires tool 'artifacts' not granted")

        assert registry.get_skill("gated") is None
        assert registry.get_metadata("gated") is None
        assert "gated" not in registry.skill_names
        assert registry.get_unavailable_reason("gated") == "requires tool 'artifacts' not granted"
        assert registry.unavailable_skills == {"gated": "requires tool 'artifacts' not granted"}

    def test_register_clears_unavailable_mark(self, tmp_path: Path) -> None:
        skill_dir = _make_skill_dir(tmp_path, "revived")
        registry = SkillRegistry()
        registry.mark_unavailable("revived", "some reason")
        registry.register(skill_dir)

        assert registry.get_unavailable_reason("revived") is None
        assert registry.get_skill("revived") is not None

    def test_to_prompt_xml_unavailable_block(self, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "usable")
        registry = SkillRegistry()
        registry.discover(tmp_path)
        registry.mark_unavailable("gated", "requires tool <artifacts> & friends")

        xml = registry.to_prompt_xml()
        assert "<available_skills>" in xml
        assert 'name="usable"' in xml
        assert "<unavailable_skills>" in xml
        assert 'name="gated"' in xml
        # Reason is escaped and present
        assert "<reason>requires tool &lt;artifacts&gt; &amp; friends</reason>" in xml

    def test_to_prompt_xml_unavailable_only(self) -> None:
        """A registry with only unavailable skills still renders — the gap
        must stay visible instead of collapsing to an empty manifest."""
        registry = SkillRegistry()
        registry.mark_unavailable("gated", "missing tool")
        xml = registry.to_prompt_xml()
        assert "<available_skills>" not in xml
        assert "<unavailable_skills>" in xml
        assert 'name="gated"' in xml

    def test_discover_raises_on_invalid(self, tmp_path: Path) -> None:
        """An invalid skill stops discovery instead of vanishing from the catalogue."""
        _make_skill_dir(tmp_path, "valid-one")
        _make_invalid_skill_dir(tmp_path, "BadName")
        registry = SkillRegistry()
        with pytest.raises(SkillValidationError):
            registry.discover(tmp_path)

    def test_discover_strict_leaves_registry_untouched(self, tmp_path: Path) -> None:
        """A strict failure commits nothing — not even the skills that parsed."""
        _make_skill_dir(tmp_path, "valid-one")
        _make_invalid_skill_dir(tmp_path, "zz-bad")
        registry = SkillRegistry()
        with pytest.raises(SkillValidationError):
            registry.discover(tmp_path)
        assert registry.skill_count == 0
        assert registry.skill_names == []

    def test_discover_strict_preserves_prior_skills(self, tmp_path: Path) -> None:
        """A failed discovery does not disturb skills registered earlier."""
        first = tmp_path / "first"
        first.mkdir()
        registry = SkillRegistry()
        registry.register(_make_skill_dir(first, "already-here"))

        broken = tmp_path / "second"
        broken.mkdir()
        _make_invalid_skill_dir(broken, "BadName")
        with pytest.raises(SkillValidationError):
            registry.discover(broken)
        assert registry.skill_names == ["already-here"]

    def test_discover_lenient_skips_invalid(self, tmp_path: Path) -> None:
        """strict=False keeps the old warn-and-continue behaviour."""
        _make_skill_dir(tmp_path, "valid-one")
        _make_invalid_skill_dir(tmp_path, "BadName")
        registry = SkillRegistry()
        count = registry.discover(tmp_path, strict=False)
        assert count == 1
        assert registry.skill_count == 1

    def test_discover_raises_on_nonexistent_dir(self) -> None:
        """A missing skills directory is a deployment error, not an empty result."""
        registry = SkillRegistry()
        with pytest.raises(SkillDiscoveryError, match="not found"):
            registry.discover("/nonexistent/path")

    def test_discover_lenient_skips_nonexistent_dir(self) -> None:
        registry = SkillRegistry()
        assert registry.discover("/nonexistent/path", strict=False) == 0

    def test_discover_errors_share_a_base(self, tmp_path: Path) -> None:
        """SkillError covers every discovery failure a caller wants to catch."""
        _make_invalid_skill_dir(tmp_path, "BadName")
        registry = SkillRegistry()
        with pytest.raises(SkillError):
            registry.discover(tmp_path)
        with pytest.raises(SkillError):
            registry.discover("/nonexistent/path")

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
