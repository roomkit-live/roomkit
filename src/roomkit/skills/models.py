"""Data models for Agent Skills."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("roomkit.skills")

_SCRIPTS_DIR = "scripts"
_REFERENCES_DIR = "references"


@dataclass
class SkillMetadata:
    """Lightweight metadata parsed from SKILL.md frontmatter."""

    name: str
    description: str
    license: str | None = None
    compatibility: str | None = None
    allowed_tools: str | None = None
    extra_metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class Skill:
    """Full skill definition including instructions body."""

    metadata: SkillMetadata
    instructions: str
    path: Path

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def description(self) -> str:
        return self.metadata.description

    @property
    def has_scripts(self) -> bool:
        scripts_dir = self.path / _SCRIPTS_DIR
        return scripts_dir.is_dir() and any(scripts_dir.iterdir())

    @property
    def has_references(self) -> bool:
        refs_dir = self.path / _REFERENCES_DIR
        return refs_dir.is_dir() and any(refs_dir.iterdir())

    def list_scripts(self) -> list[str]:
        """List script filenames in the skill's scripts/ directory."""
        scripts_dir = self.path / _SCRIPTS_DIR
        if not scripts_dir.is_dir():
            return []
        return sorted(f.name for f in scripts_dir.iterdir() if f.is_file())

    def list_references(self) -> list[str]:
        """List reference filenames in the skill's references/ directory."""
        refs_dir = self.path / _REFERENCES_DIR
        if not refs_dir.is_dir():
            return []
        return sorted(f.name for f in refs_dir.iterdir() if f.is_file())

    def read_reference(self, filename: str) -> str:
        """Read a reference file by name.

        Raises:
            ValueError: If filename contains path traversal characters.
            FileNotFoundError: If the reference file does not exist.
        """
        if ".." in filename or "/" in filename or "\\" in filename:
            raise ValueError(f"Invalid reference filename: {filename!r}")
        refs_dir = self.path / _REFERENCES_DIR
        ref_path = refs_dir / filename
        if not ref_path.is_file():
            raise FileNotFoundError(f"Reference not found: {filename}")
        return ref_path.read_text(encoding="utf-8")


class ScriptResult(BaseModel):
    """Result of executing a skill script."""

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    success: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
