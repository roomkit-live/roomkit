"""Data models for Agent Skills."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from roomkit.skills.paths import _resolve_contained_directory, safe_join_filename

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
    allowed_tools: str | list[str] | None = None
    """Tool access patterns this skill gates (RFC §24.2).

    Both frontmatter forms are accepted: a YAML list, which is what the RFC's
    own example uses, and a comma-separated scalar. Entries are ``ToolPolicy``
    globs, so ``search_*`` covers every tool whose name starts with ``search_``.
    """

    extra_metadata: dict[str, str] = field(default_factory=dict)

    @property
    def gated_tool_names(self) -> list[str]:
        """The patterns this skill gates, one per entry.

        Accepts either frontmatter form. Returns an empty list when the field
        is ``None`` or blank. The entries are patterns, not names — match them
        with :meth:`gates`, never with ``in``.
        """
        if not self.allowed_tools:
            return []
        if isinstance(self.allowed_tools, list):
            return [str(t).strip() for t in self.allowed_tools if str(t).strip()]
        return [t.strip() for t in self.allowed_tools.split(",") if t.strip()]

    def gates(self, tool_name: str) -> bool:
        """Whether this skill gates *tool_name* (RFC §24.2 glob semantics).

        Uses the same ``fnmatch`` matching as :class:`~roomkit.tools.policy.ToolPolicy`,
        so a skill and a policy agree on what a pattern covers.
        """
        return any(fnmatch(tool_name, pattern) for pattern in self.gated_tool_names)


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
        scripts_dir = self._bundled_directory(_SCRIPTS_DIR, kind="script")
        return scripts_dir is not None and any(scripts_dir.iterdir())

    @property
    def has_references(self) -> bool:
        refs_dir = self._bundled_directory(_REFERENCES_DIR, kind="reference")
        return refs_dir is not None and any(refs_dir.iterdir())

    def _bundled_directory(self, name: str, *, kind: str) -> Path | None:
        """Return one real directory contained by the skill, if it exists."""
        directory = self.path / name
        if not directory.is_dir():
            return None
        return _resolve_contained_directory(directory, kind=kind)

    def list_scripts(self) -> list[str]:
        """List script filenames in the skill's scripts/ directory."""
        scripts_dir = self._bundled_directory(_SCRIPTS_DIR, kind="script")
        if scripts_dir is None:
            return []
        return sorted(f.name for f in scripts_dir.iterdir() if f.is_file())

    def list_references(self) -> list[str]:
        """List reference filenames in the skill's references/ directory."""
        refs_dir = self._bundled_directory(_REFERENCES_DIR, kind="reference")
        if refs_dir is None:
            return []
        return sorted(f.name for f in refs_dir.iterdir() if f.is_file())

    def read_reference(self, filename: str) -> str:
        """Read a reference file by name.

        Raises:
            SkillPathError: If filename is not a plain name inside the skill's
                ``references/`` directory. Subclasses ValueError.
            FileNotFoundError: If the reference file does not exist.
        """
        ref_path = safe_join_filename(self.path / _REFERENCES_DIR, filename, kind="reference")
        if not ref_path.is_file():
            raise FileNotFoundError(f"Reference not found: {filename}")
        return ref_path.read_text(encoding="utf-8")

    def resolve_script(self, script_name: str) -> Path:
        """Resolve a script inside the skill's ``scripts/`` directory.

        ``ScriptExecutor`` implementations should build their command from this
        rather than joining the name themselves. Execution policy — sandboxing,
        timeouts, allowed interpreters — stays the integrator's call, but which
        file gets run should not depend on each integrator repeating the same
        containment check.

        Raises:
            SkillPathError: If script_name is not a plain name inside the
                skill's ``scripts/`` directory. Subclasses ValueError.
            FileNotFoundError: If the script does not exist.
        """
        script_path = safe_join_filename(self.path / _SCRIPTS_DIR, script_name, kind="script")
        if not script_path.is_file():
            raise FileNotFoundError(f"Script not found: {script_name}")
        return script_path


class ScriptResult(BaseModel):
    """Result of executing a skill script."""

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    success: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
