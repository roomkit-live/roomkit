"""SKILL.md frontmatter parsing and validation."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from roomkit.skills.models import Skill, SkillMetadata

logger = logging.getLogger("roomkit.skills")

_NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_NAME_MAX_LEN = 64
_DESC_MAX_LEN = 1024
_SKILL_FILENAMES = ("SKILL.md", "skill.md")

# Known frontmatter keys that map to SkillMetadata fields
_KNOWN_KEYS = {"name", "description", "license", "compatibility", "allowed_tools"}


class SkillParseError(Exception):
    """Failed to parse SKILL.md content."""


class SkillValidationError(Exception):
    """SKILL.md metadata failed validation."""


def find_skill_md(skill_dir: Path) -> Path | None:
    """Find SKILL.md in a directory (case-insensitive)."""
    for name in _SKILL_FILENAMES:
        candidate = skill_dir / name
        if candidate.is_file():
            return candidate
    # Fallback: scan for any case variation
    for child in skill_dir.iterdir():
        if child.is_file() and child.name.lower() == "skill.md":
            return child
    return None


def parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    """Split SKILL.md into frontmatter dict and body string.

    Tries yaml.safe_load first, falls back to simple key: value parsing.

    Raises:
        SkillParseError: If content has no valid frontmatter delimiters.
    """
    stripped = content.lstrip("\ufeff")  # strip BOM
    if not stripped.startswith("---"):
        raise SkillParseError("SKILL.md must start with '---' frontmatter delimiter")

    # Find closing delimiter
    end_idx = stripped.find("\n---", 3)
    if end_idx == -1:
        raise SkillParseError("SKILL.md missing closing '---' frontmatter delimiter")

    fm_text = stripped[3:end_idx].strip()
    body = stripped[end_idx + 4 :].strip()  # skip past \n---

    data = _parse_yaml_or_fallback(fm_text)
    return data, body


def _parse_yaml_or_fallback(fm_text: str) -> dict[str, str]:
    """Try yaml.safe_load, fall back to regex key-value parsing."""
    try:
        import yaml  # type: ignore[import-untyped]

        result = yaml.safe_load(fm_text)
        if isinstance(result, dict):
            return {str(k): str(v) for k, v in result.items() if v is not None}
    except Exception:
        logger.debug("YAML parse failed, falling back to regex parser")

    # Simple key: value parser
    data: dict[str, str] = {}
    for line in fm_text.splitlines():
        match = re.match(r"^(\w[\w_-]*)\s*:\s*(.+)$", line.strip())
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip().strip("\"'")
            data[key] = value
    return data


def validate_metadata(data: dict[str, str], skill_dir: Path) -> list[str]:
    """Validate parsed frontmatter data. Returns list of error strings."""
    errors: list[str] = []

    name = data.get("name", "")
    if not name:
        errors.append("Missing required field: name")
    elif not _NAME_PATTERN.match(name):
        errors.append(
            f"Invalid name {name!r}: must be kebab-case (lowercase alphanumeric + hyphens)"
        )
    elif len(name) > _NAME_MAX_LEN:
        errors.append(f"Name too long: {len(name)} chars (max {_NAME_MAX_LEN})")

    # Per spec: name must match directory name
    if name and name != skill_dir.name:
        errors.append(f"Name {name!r} does not match directory name {skill_dir.name!r}")

    description = data.get("description", "")
    if not description:
        errors.append("Missing required field: description")
    elif len(description) > _DESC_MAX_LEN:
        errors.append(f"Description too long: {len(description)} chars (max {_DESC_MAX_LEN})")

    return errors


def parse_skill_metadata(skill_dir: Path) -> SkillMetadata:
    """Parse SKILL.md frontmatter only (level 1 — lightweight).

    Raises:
        SkillParseError: If SKILL.md cannot be found or parsed.
        SkillValidationError: If metadata fails validation.
    """
    skill_dir = skill_dir.resolve()
    md_path = find_skill_md(skill_dir)
    if md_path is None:
        raise SkillParseError(f"No SKILL.md found in {skill_dir}")

    content = md_path.read_text(encoding="utf-8")
    data, _ = parse_frontmatter(content)

    errors = validate_metadata(data, skill_dir)
    if errors:
        raise SkillValidationError(f"Validation failed for {skill_dir.name}: {'; '.join(errors)}")

    extra = {k: v for k, v in data.items() if k not in _KNOWN_KEYS}
    return SkillMetadata(
        name=data["name"],
        description=data["description"],
        license=data.get("license"),
        compatibility=data.get("compatibility"),
        allowed_tools=data.get("allowed_tools"),
        extra_metadata=extra,
    )


def parse_skill(skill_dir: Path) -> Skill:
    """Parse full SKILL.md including instructions body (level 2).

    Raises:
        SkillParseError: If SKILL.md cannot be found or parsed.
        SkillValidationError: If metadata fails validation.
    """
    skill_dir = skill_dir.resolve()
    md_path = find_skill_md(skill_dir)
    if md_path is None:
        raise SkillParseError(f"No SKILL.md found in {skill_dir}")

    content = md_path.read_text(encoding="utf-8")
    data, body = parse_frontmatter(content)

    errors = validate_metadata(data, skill_dir)
    if errors:
        raise SkillValidationError(f"Validation failed for {skill_dir.name}: {'; '.join(errors)}")

    extra = {k: v for k, v in data.items() if k not in _KNOWN_KEYS}
    metadata = SkillMetadata(
        name=data["name"],
        description=data["description"],
        license=data.get("license"),
        compatibility=data.get("compatibility"),
        allowed_tools=data.get("allowed_tools"),
        extra_metadata=extra,
    )

    return Skill(metadata=metadata, instructions=body, path=skill_dir)
