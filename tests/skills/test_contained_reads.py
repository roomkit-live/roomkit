"""A skill file is read through the descriptor its check opened (RFC §24.6,
§17.8): one swapped for a link out between the check and the read is
refused, never read (RMK-1008).

Each test makes the swap deterministic at a point of the read: right after
the reader resolves the path for its check, or at the open itself, which a
swap of the path while it is opened answers with another file. The three
readers are the host's `Skill.read_reference`, the model's tool path
(`SkillRegistry.read_file`) and the loading of SKILL.md.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from roomkit.skills import SkillError, SkillRegistry

_NAME = "probe-skill"
_INSIDE = "inside guide"
_OUTSIDE = "OUTSIDE-SECRET"

Read = Callable[[], Awaitable[str]]


def _skill_md(description: str) -> str:
    return f"---\nname: {_NAME}\ndescription: {description}\n---\nBody\n"


@dataclass
class _Layout:
    skill: Path
    outside: Path


@pytest.fixture
def layout(tmp_path: Path) -> _Layout:
    skill = (tmp_path / "skills" / _NAME).resolve()
    (skill / "references").mkdir(parents=True)
    (skill / "SKILL.md").write_text(_skill_md(_INSIDE), encoding="utf-8")
    (skill / "references" / "guide.md").write_text(_INSIDE, encoding="utf-8")
    (skill / "references" / "other.md").write_text("another inside", encoding="utf-8")
    outside = (tmp_path / "outside").resolve()
    outside.mkdir()
    (outside / "guide.md").write_text(_OUTSIDE, encoding="utf-8")
    (outside / "SKILL.md").write_text(_skill_md(_OUTSIDE), encoding="utf-8")
    return _Layout(skill, outside)


def _reference_reader(skill: Path) -> Read:
    loaded = _registered(skill).get_skill(_NAME)
    assert loaded is not None

    async def read() -> str:
        return loaded.read_reference("guide.md")

    return read


def _tool_reader(skill: Path) -> Read:
    registry = _registered(skill)
    assert registry.get_skill(_NAME) is not None

    async def read() -> str:
        return await registry.read_file(_NAME, "guide.md")

    return read


def _skill_md_reader(skill: Path) -> Read:
    async def read() -> str:
        metadata = _registered(skill).get_metadata(_NAME)
        assert metadata is not None
        return metadata.description

    return read


def _registered(skill: Path) -> SkillRegistry:
    registry = SkillRegistry()
    registry.register(skill)
    return registry


_READERS = [
    pytest.param("references/guide.md", _reference_reader, id="Skill.read_reference"),
    pytest.param("references/guide.md", _tool_reader, id="SkillRegistry.read_file"),
    pytest.param("SKILL.md", _skill_md_reader, id="SKILL.md"),
]


async def _assert_refused(read: Read) -> None:
    try:
        text = await read()
    except SkillError:
        return
    pytest.fail(f"read {text!r} where the file was refused")


@pytest.mark.parametrize(("file", "reader"), _READERS)
async def test_a_file_swapped_for_a_link_out_after_its_check_is_never_read(
    monkeypatch: pytest.MonkeyPatch,
    layout: _Layout,
    file: str,
    reader: Callable[[Path], Read],
) -> None:
    read = reader(layout.skill)
    target = layout.skill / file
    real = os.path.realpath
    swapped = False

    def swap_once_resolved(path: Any, *args: Any, **kwargs: Any) -> str:
        nonlocal swapped
        answer = real(path, *args, **kwargs)
        if not swapped and os.fspath(path) == str(target):
            swapped = True
            link = target.with_name(target.name + ".link")
            link.symlink_to(layout.outside / target.name)
            os.replace(link, target)
        return answer

    monkeypatch.setattr(os.path, "realpath", swap_once_resolved)

    await _assert_refused(read)
    assert swapped


_HELD = [
    pytest.param("another file inside", id="replaced-once-open"),
    pytest.param(
        "the outside file its path names",
        id="exchanged-while-opened",
        marks=pytest.mark.skipif(sys.platform != "linux", reason="/proc/self/fd is Linux's"),
    ),
]


@pytest.mark.parametrize("held", _HELD)
@pytest.mark.parametrize(("file", "reader"), _READERS)
async def test_a_descriptor_that_holds_another_file_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    layout: _Layout,
    file: str,
    reader: Callable[[Path], Read],
    held: str,
) -> None:
    """The open hands back a descriptor on another file than the one the
    path names: what a swap of the path while it is opened leaves. Inside,
    its device and inode differ from the path's. Outside, the path is made
    a hard link to it, so that only the descriptor's own path tells."""
    target = layout.skill / file
    if held == "another file inside":
        opened = layout.skill / "references" / "other.md"
    else:
        opened = layout.outside / target.name
        target.unlink()
        target.hardlink_to(opened)
    read = reader(layout.skill)
    real = os.open

    def open_another(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        chosen = opened if os.fspath(path) == str(target) else path
        return real(chosen, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", open_another)

    await _assert_refused(read)


@pytest.mark.skipif(sys.platform != "linux", reason="/proc/self/fd is Linux's")
@pytest.mark.parametrize(("file", "reader"), _READERS)
async def test_a_file_whose_descriptor_cannot_be_placed_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    layout: _Layout,
    file: str,
    reader: Callable[[Path], Read],
) -> None:
    """Where ``/proc/self/fd`` cannot be read, nothing tells where the open
    file is: it is refused, as roomkit-rs refuses it."""
    read = reader(layout.skill)
    real = os.readlink

    def unreadable_proc(path: Any, *args: Any, **kwargs: Any) -> str:
        if os.fspath(path).startswith("/proc/self/fd/"):
            raise PermissionError(13, "Permission denied", os.fspath(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "readlink", unreadable_proc)

    await _assert_refused(read)
