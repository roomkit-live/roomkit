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

A checked path is then read through the file it opened, never by its name
again (RFC §24.6, §17.8): someone who can write into the skill could otherwise
swap the file for a link out between the check and the read.
:func:`_read_text_inside` opens the resolved path, then checks the descriptor.
On Linux the path the kernel gives the open file (``/proc/self/fd``) must be
inside, so a file exchanged for a link out, even atomically, is never read,
and a ``/proc`` that cannot be read refuses the file. On every platform the
path resolved anew must still be inside and name the device and inode the
descriptor holds.
"""

from __future__ import annotations

import errno
import os
import stat
import sys
import unicodedata
from collections.abc import Callable
from enum import Enum
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


class _OpenRefusal(Enum):
    """Why a path was not read as the regular file it resolves to."""

    MISSING = "missing"
    """It resolves to nothing."""
    ESCAPES = "escapes"
    """It resolves where it may not."""
    NOT_A_FILE = "not a file"
    """It resolves to no regular file."""
    CHANGED = "changed"
    """Once opened, it resolves elsewhere or names another file, or the
    descriptor holds a file elsewhere: it changed while it was opened."""
    UNVERIFIED = "unverified"
    """The descriptor's path could not be read (``/proc/self/fd``, on Linux):
    the file is refused, its place unknown."""


_MISSING_ERRNOS = frozenset({errno.ENOENT, errno.ENOTDIR, errno.ELOOP})
"""What a path that resolves to nothing fails with: no entry, a component that
is no directory, a symlink loop."""


class _RefusedOpenError(Exception):
    """An open refused for :attr:`refusal`, which the reader turns into its
    own error."""

    def __init__(self, refusal: _OpenRefusal) -> None:
        super().__init__(refusal.value)
        self.refusal = refusal


def _read_text_inside(
    path: Path,
    inside: Callable[[Path], bool],
    refused: Callable[[_OpenRefusal], Exception],
) -> str:
    """The UTF-8 text, with universal newlines, of the regular file *path*
    resolves to, when *inside* takes where it resolves; read from the file
    the check opened (RFC §24.6, §17.8).

    Raises:
        Exception: ``refused(why)`` for a path refused (:class:`_OpenRefusal`).
        UnicodeDecodeError: The file is not UTF-8 text.
        OSError: The path could not be resolved, opened or read otherwise.
    """
    try:
        fd = _open_inside(path, inside)
    except _RefusedOpenError as refusal:
        raise refused(refusal.refusal) from refusal.__cause__
    try:
        handle = open(fd, encoding="utf-8")  # noqa: SIM115 - closed below
    except BaseException:
        os.close(fd)
        raise
    with handle:
        return handle.read()


def _open_inside(path: Path, inside: Callable[[Path], bool]) -> int:
    """A descriptor open on the regular file *path* resolves to, when *inside*
    takes where it resolves, checked once it is open: the path resolved anew
    must still resolve there and name the file the descriptor holds, and on
    Linux the descriptor's own path must be inside. A FIFO is refused before
    the open, which it would block.

    Raises:
        _RefusedOpenError: The path is refused.
        OSError: It could not be resolved or opened otherwise.
    """
    resolved = _resolved(path)
    if not inside(resolved):
        raise _RefusedOpenError(_OpenRefusal.ESCAPES)
    if not stat.S_ISREG(_disk(os.stat, resolved).st_mode):
        raise _RefusedOpenError(_OpenRefusal.NOT_A_FILE)
    fd = _disk(os.open, resolved, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise _RefusedOpenError(_OpenRefusal.NOT_A_FILE)
        if not _still_names(path, resolved, opened, inside):
            raise _RefusedOpenError(_OpenRefusal.CHANGED)
        if not _descriptor_inside(fd, inside):
            raise _RefusedOpenError(_OpenRefusal.CHANGED)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _resolved(path: Path) -> Path:
    """*path* resolved with its links followed, each of them existing.

    Raises:
        _RefusedOpenError: It resolves to nothing (:attr:`_OpenRefusal.MISSING`).
        OSError: It could not be resolved otherwise.
    """
    try:
        return _disk(path.resolve, strict=True)
    except RuntimeError as exc:
        # Python 3.12 reports a symlink loop so; later versions raise ELOOP.
        raise _RefusedOpenError(_OpenRefusal.MISSING) from exc


def _disk[**P, R](call: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    """``call(*args, **kwargs)``, an error for a path that resolves to nothing
    raised as :attr:`_OpenRefusal.MISSING`."""
    try:
        return call(*args, **kwargs)
    except OSError as exc:
        if exc.errno in _MISSING_ERRNOS:
            raise _RefusedOpenError(_OpenRefusal.MISSING) from exc
        raise


def _still_names(
    path: Path, resolved: Path, opened: os.stat_result, inside: Callable[[Path], bool]
) -> bool:
    """Whether *path*, resolved anew, still resolves to *resolved*, inside, and
    names the file *opened* describes: the same device and inode."""
    try:
        again = path.resolve(strict=True)
        now = os.stat(again)
    except (OSError, RuntimeError):
        return False
    same_file = (now.st_dev, now.st_ino) == (opened.st_dev, opened.st_ino)
    return again == resolved and inside(again) and same_file


def _descriptor_inside(fd: int, inside: Callable[[Path], bool]) -> bool:
    """Whether the path the kernel gives the open *fd* is inside: what was
    opened, whatever the path names now. Elsewhere than Linux there is no
    such path to read, and the device and inode the path names are the check.

    Raises:
        _RefusedOpenError: ``/proc/self/fd`` could not be read
            (:attr:`_OpenRefusal.UNVERIFIED`).
    """
    if sys.platform != "linux":
        return True
    try:
        held = os.readlink(f"/proc/self/fd/{fd}")
    except OSError as exc:
        raise _RefusedOpenError(_OpenRefusal.UNVERIFIED) from exc
    return inside(Path(held))
