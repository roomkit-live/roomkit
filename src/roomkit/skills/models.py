"""Data models for Agent Skills."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from roomkit.skills.paths import _resolve_contained_directory, safe_join_filename
from roomkit.tools.policy import matches_any_pattern

logger = logging.getLogger("roomkit.skills")

_SCRIPTS_DIR = "scripts"
_REFERENCES_DIR = "references"


def parse_tool_names(value: object) -> list[str]:
    """The tool names or patterns of an ``allowed_tools`` or ``requires``
    value (RFC §24.2): a list, or a text separated by commas, read alike
    whether bracketed (``[a, b]``) or quoted entries."""
    if isinstance(value, list | tuple):
        entries = [str(entry) for entry in value]
    elif isinstance(value, str):
        text = value.strip()
        if _bracketed(text):
            text = text[1:-1]
        entries = text.split(",")
    else:
        return []
    return _names(entries)


def _bracketed(text: str) -> bool:
    """Whether *text* is one pair of brackets around entries holding none."""
    inner = text[1:-1]
    return text.startswith("[") and text.endswith("]") and "[" not in inner and "]" not in inner


def _names(entries: Iterable[str]) -> list[str]:
    """*entries* trimmed of spaces and quotes, the empty ones left out."""
    names = (entry.strip().strip("'\"").strip() for entry in entries)
    return [name for name in names if name]


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
    def required_tool_names(self) -> list[str]:
        """Exact prerequisite names from the optional ``requires`` metadata.

        These describe dependencies, not grants. Applications resolve skill
        availability against their authorized catalogue before using it.
        """
        return parse_tool_names(self.extra_metadata.get("requires"))

    @property
    def gated_tool_names(self) -> list[str]:
        """The patterns this skill gates, one per entry.

        Accepts either frontmatter form. Returns an empty list when the field
        is ``None`` or blank. The entries are patterns, not names — match them
        with :meth:`gates`, never with ``in``.
        """
        return parse_tool_names(self.allowed_tools)

    def gates(self, tool_name: str) -> bool:
        """Whether this skill gates *tool_name* (RFC §24.2 glob semantics).

        Uses the same glob matching as :class:`~roomkit.tools.policy.ToolPolicy`,
        so a skill and a policy agree on what a pattern covers.
        """
        return matches_any_pattern(tool_name, self.gated_tool_names)


RequiresMatch = Callable[[str, Collection[str]], bool]
"""Whether a skill's ``requires`` name is served by the tool names offered: a
host's own reading of its names (a hub whose actions share its name as a
prefix, say). The default reads an exact tool name."""


def serves_exactly(required: str, offered: Collection[str]) -> bool:
    """The default :data:`RequiresMatch`: *required* is an offered tool's name."""
    return required in offered


def missing_required_tools(
    metadata: SkillMetadata,
    admitted: Iterable[str],
    closed: Iterable[str] = (),
    *,
    match: RequiresMatch = serves_exactly,
) -> list[str]:
    """The ``requires`` names of *metadata* the conversation does not serve
    (RFC §24.3), as *match* reads a name against the tools offered: those of
    *admitted* (the tools the conversation declares once its tool policy is
    applied, skill gating aside, since activating a skill opens what it
    gates) that match no *closed* pattern, which nothing can open
    (:meth:`SkillRegistry.closed_tool_names`). Every door checks an activation
    with this one rule."""
    shut = list(closed)
    offered = {name for name in admitted if not matches_any_pattern(name, shut)}
    return [name for name in metadata.required_tool_names if not match(name, offered)]


def missing_tools_error(missing: list[str]) -> str:
    """What the model reads of an activation whose required tools are missing."""
    return json.dumps({"error": f"Required tools not available: {', '.join(missing)}"})


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
