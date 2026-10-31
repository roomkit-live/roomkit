"""SKILL.md frontmatter parsing and validation."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from roomkit.skills._frontmatter_lines import GATING_KEYS, parse_lines
from roomkit.skills.errors import SkillParseError, SkillValidationError
from roomkit.skills.models import Skill, SkillMetadata, bracketed_pattern_errors

__all__ = [
    "SkillParseError",
    "SkillValidationError",
    "find_skill_md",
    "frontmatter_text",
    "parse_frontmatter",
    "parse_skill",
    "parse_skill_metadata",
    "read_frontmatter_yaml",
    "split_frontmatter",
    "valid_skill_name",
    "validate_metadata",
]

logger = logging.getLogger("roomkit.skills")

FrontmatterValue = str | list[str]
"""A SKILL.md frontmatter value: a scalar, or the YAML list form ``allowed_tools``
uses in the RFC's own example."""

_NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_NAME_MAX_LEN = 64
_DESC_MAX_LEN = 1024
_SKILL_FILENAMES = ("SKILL.md", "skill.md")

# Known frontmatter keys that map to SkillMetadata fields
_KNOWN_KEYS = {"name", "description", "license", "compatibility", "allowed_tools"}

FRONTMATTER_MAX = 64 * 1024
"""Characters a skill's frontmatter may take, in its entry's JSON as in its
``SKILL.md``'s YAML: far above any real one (a name, a description of 1024
characters, some metadata), and a bound on what reading and comparing it costs."""

_FLOW_DEPTH_MAX = 64
"""Brackets and braces open at once, read at most: pure-Python YAML takes time
quadratic in that depth (5000 levels, 3.5 s)."""

_DEPTH_MAX = 64
"""Collections a frontmatter may nest, in flow or block style: the C loader
builds a node by recursion, and 31000 nested ``- `` overflow its stack and
end the process."""


_CONSTRUCTOR_ERRORS = (AttributeError, IndexError, KeyError, OverflowError, TypeError, ValueError)
"""What PyYAML's safe constructors raise, outside ``YAMLError``, on a scalar
its tag cannot hold (``!!bool maybe``, ``!!int ''``, ``!!timestamp abc``) or
an integer past Python's digit limit."""


class _FrontmatterNotReadError(SkillParseError):
    """A frontmatter that was not read as a YAML mapping: the text is not
    YAML or not a mapping, or it holds a value JSON cannot hold. Unlike a
    bound, it says nothing against the text, which a caller may read another
    way."""


def _read_skill_md(skill_dir: Path) -> tuple[Path, str]:
    """Resolve and decode one SKILL.md using the public error hierarchy."""
    try:
        resolved_dir = skill_dir.resolve()
        md_path = find_skill_md(resolved_dir)
        if md_path is None:
            raise SkillParseError(f"No SKILL.md found in {resolved_dir}")
        resolved_md = md_path.resolve()
        if resolved_dir not in resolved_md.parents:
            raise SkillParseError(f"SKILL.md escapes its skill directory: {md_path}")
        return resolved_dir, resolved_md.read_text(encoding="utf-8")
    except SkillParseError:
        raise
    except (OSError, RuntimeError, UnicodeError) as exc:
        raise SkillParseError(f"Unable to read SKILL.md in {skill_dir}: {exc}") from exc


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


def valid_skill_name(name: str) -> bool:
    """Whether *name* is a skill name as SKILL.md spells one (RFC §24.1):
    lowercase letters and digits in hyphen-separated runs, at most 64
    characters."""
    return bool(_NAME_PATTERN.fullmatch(name)) and len(name) <= _NAME_MAX_LEN


def split_frontmatter(content: str) -> tuple[str, str]:
    """Split SKILL.md into its frontmatter text and its body, both stripped.

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

    return stripped[3:end_idx].strip(), stripped[end_idx + 4 :].strip()  # skip past \n---


def read_frontmatter_yaml(text: str) -> dict[str, Any]:
    """The frontmatter *text* read as YAML within the bounds of
    :func:`_load_bounded_yaml`, as JSON renders it (a date as its ISO string).

    Raises:
        SkillParseError: The text is past a bound, was not read as a YAML
            mapping, or holds a key or a value JSON cannot hold.
    """
    data = _load_bounded_yaml(text)
    try:
        rendered: dict[str, Any] = json.loads(json.dumps(data, default=str))
    except (OverflowError, TypeError, ValueError) as exc:
        # A date or binary key, an integer past Python's digit limit.
        raise _FrontmatterNotReadError("the frontmatter holds a value JSON cannot hold") from exc
    return rendered


def _load_bounded_yaml(text: str) -> dict[Any, Any]:
    """The frontmatter *text* loaded as a YAML mapping, once it is within the
    bounds.

    An alias is refused before anything is built: YAML keeps an aliased node
    shared, so a few hundred bytes of aliases of aliases parse at once and
    then expand exponentially once read. Collections nested past
    :data:`_DEPTH_MAX` are refused from the same pass, before the loader
    recurses into them, and so is a frontmatter past :data:`FRONTMATTER_MAX`
    or holding more than :data:`_FLOW_DEPTH_MAX` brackets open at once.

    Raises:
        SkillParseError: The text is past one of those bounds, or was not
            read as a YAML mapping.
    """
    if len(text) > FRONTMATTER_MAX:
        raise SkillParseError(f"the frontmatter is past {FRONTMATTER_MAX} characters")
    if _flow_depth(text) > _FLOW_DEPTH_MAX:
        raise SkillParseError("the frontmatter nests too deep")
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    try:
        _refuse_unsafe_events(yaml.parse(text, loader))
        if loader is yaml.SafeLoader:
            data = yaml.safe_load(text)
        else:
            data = yaml.load(text, Loader=yaml.CSafeLoader)
    except yaml.YAMLError as exc:
        raise _FrontmatterNotReadError("the frontmatter is not YAML") from exc
    except RecursionError as exc:
        raise SkillParseError("the frontmatter nests too deep") from exc
    except _CONSTRUCTOR_ERRORS as exc:
        raise _FrontmatterNotReadError(
            "the frontmatter holds a value its YAML tag cannot construct"
        ) from exc
    if not isinstance(data, dict):
        raise _FrontmatterNotReadError("the frontmatter is not a mapping")
    return data


def _refuse_unsafe_events(events: Iterable[Any]) -> None:
    """Refuse an alias, or collections nested past :data:`_DEPTH_MAX`, from
    the parse events alone: nothing is built yet, and the pass stops at the
    first one found.

    Only an alias to an anchor defined before it is refused. One naming no
    anchor cannot expand, and the load that follows fails on it: the text is
    not YAML (``allowed_tools: *_admin``, a glob)."""
    depth = 0
    anchors: set[str] = set()
    for event in events:
        if isinstance(event, yaml.AliasEvent):
            if event.anchor in anchors:
                raise SkillParseError("the frontmatter holds a YAML alias")
            continue
        if isinstance(event, yaml.NodeEvent) and event.anchor is not None:
            anchors.add(event.anchor)
        if isinstance(event, yaml.CollectionStartEvent):
            depth += 1
            if depth > _DEPTH_MAX:
                raise SkillParseError("the frontmatter nests too deep")
        elif isinstance(event, yaml.CollectionEndEvent):
            depth -= 1


def _flow_depth(text: str) -> int:
    """The most brackets and braces *text* holds open at once, quotes not
    read: an upper bound on its YAML flow nesting."""
    depth = deepest = 0
    for char in text:
        if char in "[{":
            depth += 1
            deepest = max(deepest, depth)
        elif char in "]}" and depth:
            depth -= 1
    return deepest


def parse_frontmatter(content: str) -> tuple[dict[str, FrontmatterValue], str]:
    """Split SKILL.md into frontmatter dict and body string.

    The frontmatter is read as YAML within the bounds of
    :func:`read_frontmatter_yaml`, and line by line (``key: value``) when
    YAML does not read it as a mapping.

    Raises:
        SkillParseError: If content has no valid frontmatter delimiters, or
            its YAML holds an alias, nests past 64 levels or is past
            :data:`FRONTMATTER_MAX` characters.
    """
    fm_text, body = split_frontmatter(content)
    data = _parse_yaml_or_fallback(fm_text)
    return data, body


def _parse_yaml_or_fallback(fm_text: str) -> dict[str, FrontmatterValue]:
    """Read the frontmatter as bounded YAML, or line by line when it is not
    read as YAML at all."""
    try:
        mapping = _load_bounded_yaml(fm_text)
        _refuse_nested_lists(mapping)
        return _as_strings(mapping)
    except _FrontmatterNotReadError:
        logger.debug("YAML parse failed, falling back to regex parser")
        return parse_lines(fm_text)


def _refuse_nested_lists(mapping: dict[Any, Any]) -> None:
    """Refuse an ``allowed_tools`` or ``requires`` that is neither a text nor
    a list of texts (RFC §24.1): turned into one text, it would gate nothing."""
    key = unreadable_gating_key(mapping)
    if key is not None:
        raise SkillParseError(f"the frontmatter's {key} is neither a text nor a list of texts")


def unreadable_gating_key(mapping: Mapping[Any, Any]) -> str | None:
    """The first of ``allowed_tools`` and ``requires`` in *mapping* that is
    neither a text nor a list of texts, or ``None``: one rule for a local
    skill's frontmatter and a served skill's entry (RFC §24.1, whichever
    reader)."""
    for key in GATING_KEYS:
        value = mapping.get(key)
        items = value if isinstance(value, list) else [value]
        if any(isinstance(item, dict | list | set | tuple) for item in items):
            return key
    return None


def _as_strings(mapping: dict[Any, Any]) -> dict[str, FrontmatterValue]:
    """A loaded frontmatter with each key and value as a string, and each list
    as a list of strings, an empty item left out: ``allowed_tools`` and
    ``requires`` take the YAML list form of the RFC's own SKILL.md example."""
    try:
        return {
            str(k): [str(item) for item in v if item is not None]
            if isinstance(v, list)
            else str(v)
            for k, v in mapping.items()
            if v is not None
        }
    except ValueError as exc:
        # A hexadecimal integer past Python's digit limit for a decimal string.
        raise _FrontmatterNotReadError("the frontmatter holds a value text cannot hold") from exc


def frontmatter_text(value: object) -> str:
    """A frontmatter value as text, as ``SkillMetadata`` keeps a scalar and
    ``extra_metadata`` every other key, local or served alike: a list joined
    by commas, an empty item left out.

    ``allowed_tools`` is the one field whose list form carries meaning. A list
    written where a scalar belongs is joined rather than refused — a skill with
    an odd ``license:`` line should still load.
    """
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if item is not None)
    return str(value)


def validate_metadata(data: dict[str, FrontmatterValue], skill_dir: Path) -> list[str]:
    """Validate parsed frontmatter data. Returns list of error strings."""
    errors: list[str] = []

    name = frontmatter_text(data.get("name"))
    if not name:
        errors.append("Missing required field: name")
    elif not _NAME_PATTERN.fullmatch(name):
        errors.append(
            f"Invalid name {name!r}: must be kebab-case (lowercase alphanumeric + hyphens)"
        )
    elif len(name) > _NAME_MAX_LEN:
        errors.append(f"Name too long: {len(name)} chars (max {_NAME_MAX_LEN})")

    # Per spec: name must match directory name
    if name and name != skill_dir.name:
        errors.append(f"Name {name!r} does not match directory name {skill_dir.name!r}")

    description = frontmatter_text(data.get("description"))
    if not description:
        errors.append("Missing required field: description")
    elif len(description) > _DESC_MAX_LEN:
        errors.append(f"Description too long: {len(description)} chars (max {_DESC_MAX_LEN})")

    errors.extend(bracketed_pattern_errors(data))
    return errors


def parse_skill_metadata(skill_dir: Path) -> SkillMetadata:
    """Parse SKILL.md frontmatter only (level 1 — lightweight).

    Raises:
        SkillParseError: If SKILL.md cannot be found or parsed.
        SkillValidationError: If metadata fails validation.
    """
    _, metadata, _ = _parse_skill_md(skill_dir)
    return metadata


def parse_skill(skill_dir: Path) -> Skill:
    """Parse full SKILL.md including instructions body (level 2).

    Raises:
        SkillParseError: If SKILL.md cannot be found or parsed.
        SkillValidationError: If metadata fails validation.
    """
    resolved_dir, metadata, body = _parse_skill_md(skill_dir)
    return Skill(metadata=metadata, instructions=body, path=resolved_dir)


def _parse_skill_md(skill_dir: Path) -> tuple[Path, SkillMetadata, str]:
    """Read the SKILL.md of *skill_dir* into its resolved directory, its
    validated metadata and its body; an error names the skill (RFC §24.5)."""
    resolved_dir, content = _read_skill_md(skill_dir)
    try:
        data, body = parse_frontmatter(content)
    except SkillParseError as exc:
        raise SkillParseError(f"Parsing failed for {resolved_dir.name}: {exc}") from exc
    return resolved_dir, _validated_metadata(data, resolved_dir), body


def _validated_metadata(data: dict[str, FrontmatterValue], skill_dir: Path) -> SkillMetadata:
    """The metadata a parsed frontmatter declares, once it validates."""
    errors = validate_metadata(data, skill_dir)
    if errors:
        raise SkillValidationError(f"Validation failed for {skill_dir.name}: {'; '.join(errors)}")

    extra = {k: frontmatter_text(v) for k, v in data.items() if k not in _KNOWN_KEYS}
    return SkillMetadata(
        name=frontmatter_text(data["name"]),
        description=frontmatter_text(data["description"]),
        license=frontmatter_text(data.get("license")) or None,
        compatibility=frontmatter_text(data.get("compatibility")) or None,
        allowed_tools=data.get("allowed_tools"),
        extra_metadata=extra,
    )
