"""Containment-checked path resolution for files bundled inside a skill.

A skill directory is content — often authored elsewhere, sometimes vendored from
a third party — and both the reference reader and any ``ScriptExecutor`` turn a
name chosen by the model into a filesystem read or an execution. Filtering the
*name* is not enough to make that safe: ``notes.md`` contains no ``..``, no
separator and no drive letter, yet reads ``/etc/passwd`` when a symlink of that
name sits in ``references/``.

So the check here is positional rather than lexical. Both ends are resolved and
the result must still be inside the base directory. Symlink escapes fall out of
that comparison instead of having to be enumerated.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path, PureWindowsPath

from roomkit.skills.errors import SkillPathError


def _has_control_chars(text: str) -> bool:
    """True when text contains a null byte, newline, or other control char."""
    return any(unicodedata.category(char) == "Cc" for char in text)


def _is_absolute_anywhere(name: str) -> bool:
    """True when name is absolute on POSIX or on Windows (drive letter, UNC)."""
    if name.startswith("/"):
        return True
    pure = PureWindowsPath(name)
    return bool(pure.drive) or pure.is_absolute()


def _resolve_contained_directory(directory: Path, *, kind: str) -> Path:
    """Resolve a bundled directory without letting the directory itself escape.

    Checking only the final filename is insufficient when ``references/`` or
    ``scripts/`` is itself a symlink.  Anchor the resolved directory in its
    lexical parent before resolving any child selected by the model.
    """
    try:
        resolved_parent = directory.parent.resolve()
        resolved_directory = directory.resolve()
    except (OSError, RuntimeError) as exc:
        raise SkillPathError(f"Invalid {kind} directory: cannot be resolved") from exc

    if resolved_directory == resolved_parent or resolved_parent not in resolved_directory.parents:
        raise SkillPathError(
            f"Invalid {kind} directory: {directory.name!r} escapes the skill directory"
        )
    return resolved_directory


def safe_join_filename(directory: Path, filename: str, *, kind: str = "file") -> Path:
    """Resolve ``filename`` inside ``directory``, or raise.

    ``filename`` must be a single path component. Path components are rejected
    rather than silently discarded: a caller asking for ``sub/file.txt`` wants
    that file, and quietly serving ``file.txt`` instead answers a question
    nobody asked.

    The returned path is resolved, so callers get a path they can read or
    execute directly without re-deriving it.

    Args:
        directory: Bundled directory the result must stay within. The directory
            itself must also resolve inside its lexical parent.
        filename: Single-component name to resolve inside it.
        kind: Word used in the error message ("reference", "script").

    Returns:
        The resolved path, guaranteed to be inside ``directory``.

    Raises:
        SkillPathError: If ``filename`` is empty, absolute, contains a path
            separator or a control character, or resolves outside
            ``directory``.
    """
    if not filename or not filename.strip():
        raise SkillPathError(f"Invalid {kind} filename: {filename!r}")

    # NFKC folds full-width look-alikes (．． ／) onto the ASCII forms below, so
    # normalise before inspecting rather than after.
    normalized = unicodedata.normalize("NFKC", filename)

    if _has_control_chars(normalized):
        raise SkillPathError(f"Invalid {kind} filename: {filename!r}")
    if _is_absolute_anywhere(normalized):
        raise SkillPathError(f"Invalid {kind} filename: {filename!r} must be relative")
    if "/" in normalized or "\\" in normalized:
        raise SkillPathError(f"Invalid {kind} filename: {filename!r} must not contain a path")
    if normalized.strip(". ") == "":
        raise SkillPathError(f"Invalid {kind} filename: {filename!r}")

    try:
        resolved_base = _resolve_contained_directory(directory, kind=kind)
        resolved_target = (resolved_base / normalized).resolve()
    except (OSError, RuntimeError) as exc:
        raise SkillPathError(f"Invalid {kind} filename: {filename!r} cannot be resolved") from exc

    # resolve() follows symlinks, so this is where a planted link is caught.
    if resolved_target != resolved_base and resolved_base not in resolved_target.parents:
        raise SkillPathError(
            f"Invalid {kind} filename: {filename!r} escapes {resolved_base.name}/"
        )

    return resolved_target
