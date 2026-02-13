"""Skill registry for discovering and managing Agent Skills."""

from __future__ import annotations

import logging
from html import escape
from pathlib import Path

from roomkit.skills.models import Skill, SkillMetadata
from roomkit.skills.parser import (
    SkillParseError,
    SkillValidationError,
    find_skill_md,
    parse_skill,
    parse_skill_metadata,
)

logger = logging.getLogger("roomkit.skills")


class SkillRegistry:
    """Discover, load, and manage Agent Skills.

    Lightweight metadata is parsed on discover/register. Full skill
    instructions are loaded lazily on first ``get_skill()`` call and
    cached for subsequent access.
    """

    def __init__(self) -> None:
        self._metadata: dict[str, SkillMetadata] = {}
        self._skills: dict[str, Skill] = {}
        self._paths: dict[str, Path] = {}

    def discover(self, *directories: str | Path) -> int:
        """Scan directories for subdirectories containing SKILL.md.

        Returns the number of newly discovered skills. Invalid skills
        are warned and skipped.
        """
        count = 0
        for directory in directories:
            dir_path = Path(directory).resolve()
            if not dir_path.is_dir():
                logger.warning("Skill directory not found: %s", dir_path)
                continue
            for child in sorted(dir_path.iterdir()):
                if not child.is_dir():
                    continue
                if find_skill_md(child) is None:
                    continue
                try:
                    self.register(child)
                    count += 1
                except (SkillParseError, SkillValidationError) as exc:
                    logger.warning("Skipping invalid skill %s: %s", child.name, exc)
        return count

    def register(self, skill_dir: str | Path) -> SkillMetadata:
        """Register a single skill directory.

        Parses frontmatter only (lightweight). Replaces any existing
        skill with the same name.

        Raises:
            SkillParseError: If SKILL.md cannot be found or parsed.
            SkillValidationError: If metadata fails validation.
        """
        skill_path = Path(skill_dir).resolve()
        metadata = parse_skill_metadata(skill_path)
        self._metadata[metadata.name] = metadata
        self._paths[metadata.name] = skill_path
        # Invalidate cached full skill if re-registering
        self._skills.pop(metadata.name, None)
        logger.info("Registered skill: %s", metadata.name)
        return metadata

    def get_metadata(self, name: str) -> SkillMetadata | None:
        """Get metadata for a skill by name."""
        return self._metadata.get(name)

    def get_skill(self, name: str) -> Skill | None:
        """Get full skill (with instructions), loading lazily if needed."""
        if name in self._skills:
            return self._skills[name]

        if name not in self._paths:
            return None

        try:
            skill = parse_skill(self._paths[name])
            self._skills[name] = skill
            return skill
        except (SkillParseError, SkillValidationError) as exc:
            logger.error("Failed to load skill %s: %s", name, exc)
            return None

    def all_metadata(self) -> list[SkillMetadata]:
        """Return metadata for all registered skills."""
        return list(self._metadata.values())

    @property
    def skill_names(self) -> list[str]:
        """Names of all registered skills."""
        return list(self._metadata.keys())

    @property
    def skill_count(self) -> int:
        """Number of registered skills."""
        return len(self._metadata)

    def to_prompt_xml(self) -> str:
        """Generate spec-compliant <available_skills> XML block.

        Content is HTML-escaped to prevent injection.
        """
        if not self._metadata:
            return ""

        lines = ["<available_skills>"]
        for meta in self._metadata.values():
            lines.append(f'  <skill name="{escape(meta.name)}">')
            lines.append(f"    <description>{escape(meta.description)}</description>")
            if meta.license:
                lines.append(f"    <license>{escape(meta.license)}</license>")
            if meta.compatibility:
                lines.append(f"    <compatibility>{escape(meta.compatibility)}</compatibility>")
            lines.append("  </skill>")
        lines.append("</available_skills>")
        return "\n".join(lines)
