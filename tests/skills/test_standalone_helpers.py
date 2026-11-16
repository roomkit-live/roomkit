"""Standalone skill helpers expose the same rules as their callers."""

from __future__ import annotations

import sys
from collections.abc import Collection
from pathlib import Path

import pytest

from roomkit.skills import (
    SkillMetadata,
    SkillPathError,
    missing_required_tools,
    parse_tool_names,
    safe_join_filename,
    serves_exactly,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (" [ 'search_*', \"fetch\" , '', search_* ] ", ["search_*", "fetch", "search_*"]),
        ([" a,b ", "'search_*'", "", " search_* "], ["a,b", "search_*", "search_*"]),
        (("first", "second"), ["first", "second"]),
        ("[ab]", ["ab"]),
        ("[sw]earch_*, fetch]", ["[sw]earch_*", "fetch]"]),
        ("[[a,b]]", ["[[a", "b]]"]),
        (" , '' , \"\" ", []),
        (None, []),
        (42, []),
        (True, []),
        ({"name": "lookup"}, []),
        (["", None, True, False, 42, 1.5], ["None", "True", "False", "42", "1.5"]),
        ([["nested"], {"flag": False}], ["['nested']", "{'flag': False}"]),
    ],
)
def test_parse_tool_names_preserves_forms_order_and_duplicates(
    value: object, expected: list[str]
) -> None:
    assert parse_tool_names(value) == expected


@pytest.mark.parametrize("space", [" ", "\t", "\u00a0", "\u2003", "\x1c", "\x1d", "\x1e", "\x1f"])
def test_parse_tool_names_uses_python_whitespace(space: str) -> None:
    assert parse_tool_names(f"{space}[{space}lookup{space}]{space}") == ["lookup"]
    assert parse_tool_names([f"{space}'lookup'{space}", space]) == ["lookup"]


@pytest.mark.parametrize(
    ("required", "offered", "expected"),
    [
        ("lookup", [], False),
        ("lookup", ["lookup", "lookup"], True),
        ("lookup", {"lookup", "other"}, True),
        ("lookup", ("LOOKUP", "lookup_extra"), False),
        ("lookup*", ["lookup_extra"], False),
        ("", [""], True),
    ],
)
def test_serves_exactly_is_case_sensitive_membership(
    required: str, offered: Collection[str], expected: bool
) -> None:
    assert serves_exactly(required, offered) is expected


def test_missing_required_tools_keeps_order_duplicates_and_defaults() -> None:
    metadata = SkillMetadata(
        "local", "", extra_metadata={"requires": "second, first, second, served"}
    )
    assert missing_required_tools(metadata, ["served", "served"]) == ["second", "first", "second"]
    assert missing_required_tools(SkillMetadata("local", ""), []) == []


def test_missing_required_tools_consumes_iterables_and_filters_closed_patterns() -> None:
    metadata = SkillMetadata("local", "", extra_metadata={"requires": "lookup, fetch, Lookup"})
    admitted = iter(["lookup", "lookup", "fetch", "Lookup"])
    closed = iter(["look*", "f?tch"])
    assert missing_required_tools(metadata, admitted, closed) == ["lookup", "fetch"]
    assert list(admitted) == list(closed) == []


def test_missing_required_tools_passes_only_open_unique_names_to_the_matcher() -> None:
    metadata = SkillMetadata("local", "", extra_metadata={"requires": "boards, missing, boards"})
    seen: list[tuple[str, set[str]]] = []

    def match(required: str, offered: Collection[str]) -> bool:
        seen.append((required, set(offered)))
        return any(name.startswith(required + "_") for name in offered)

    assert missing_required_tools(
        metadata, ["boards_read", "boards_read", "boards_delete"], ["*_delete"], match=match
    ) == ["missing"]
    assert seen == [(name, {"boards_read"}) for name in ["boards", "missing", "boards"]]


def test_missing_required_tools_propagates_the_matcher_error_without_more_calls() -> None:
    metadata = SkillMetadata("local", "", extra_metadata={"requires": "first, second"})
    error = RuntimeError("matcher unavailable")
    seen: list[str] = []

    def match(required: str, offered: Collection[str]) -> bool:
        seen.append(required)
        raise error

    with pytest.raises(RuntimeError) as raised:
        missing_required_tools(metadata, [], match=match)
    assert raised.value is error
    assert seen == ["first"]


@pytest.mark.parametrize("kind", [None, "reference", "custom attachment"])
def test_safe_join_filename_uses_the_default_or_custom_error_kind(
    tmp_path: Path, kind: str | None
) -> None:
    options = {} if kind is None else {"kind": kind}
    with pytest.raises(SkillPathError, match=f"Invalid {kind or 'file'} filename"):
        safe_join_filename(tmp_path / "references", "../outside", **options)


@pytest.mark.parametrize("name", ["guide:notes.md", " a ", "...x"])
def test_safe_join_filename_accepts_punctuation_that_is_not_a_drive(
    tmp_path: Path, name: str
) -> None:
    directory = tmp_path / "references"
    assert safe_join_filename(directory, name) == directory / name


@pytest.mark.parametrize(
    "name", ["C:guide.md", "z:guide.md", "Z:/guide.md", "1:guide.md", "é:guide.md"]
)
def test_safe_join_filename_rejects_windows_drives_on_every_platform(
    tmp_path: Path, name: str
) -> None:
    with pytest.raises(SkillPathError, match="must be relative"):
        safe_join_filename(tmp_path / "references", name)


def test_safe_join_filename_refuses_a_directory_escape_and_a_dangling_child(
    tmp_path: Path,
) -> None:
    local = tmp_path / "local"
    local.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    directory = local / "references"
    directory.symlink_to(outside, target_is_directory=True)
    with pytest.raises(SkillPathError, match="escapes"):
        safe_join_filename(directory, "guide.md")
    directory.unlink()
    directory.mkdir()
    (directory / "guide.md").symlink_to(outside / "missing.md")
    with pytest.raises(SkillPathError, match="escapes"):
        safe_join_filename(directory, "guide.md")


def test_safe_join_filename_cycle_matches_the_python_runtime(tmp_path: Path) -> None:
    directory = tmp_path / "references"
    directory.mkdir()
    (directory / "first").symlink_to("second")
    (directory / "second").symlink_to("first")
    if sys.version_info < (3, 13):
        with pytest.raises(SkillPathError, match="cannot be resolved"):
            safe_join_filename(directory, "first")
    else:
        assert safe_join_filename(directory, "first") == directory / "first"
