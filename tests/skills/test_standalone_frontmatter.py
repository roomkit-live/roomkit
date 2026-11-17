"""Frontmatter read from application text, independently of a skill directory."""

from __future__ import annotations

import pytest

from roomkit.skills import SkillParseError, parse_frontmatter


def test_standalone_frontmatter_preserves_unknown_fields_order_lists_and_body() -> None:
    values, body = parse_frontmatter(
        "\ufeff---\ncustom: [z, a]\nallowed_tools: [read_*, write_*]\n"
        "other:\n  - a\n  - null\n  - b\nempty: []\n"
        "missing: null\ncontext: worker\n---\n\n# Body\ntext\n"
    )
    assert list(values.items()) == [
        ("custom", ["z", "a"]),
        ("allowed_tools", ["read_*", "write_*"]),
        ("other", ["a", "b"]),
        ("empty", []),
        ("context", "worker"),
    ]
    assert body == "# Body\ntext"


def test_standalone_frontmatter_reads_a_fallback_glob_without_a_skill_name() -> None:
    values, body = parse_frontmatter(
        "---\ndescription: Refunds: process them\nallowed_tools: [*_admin, read_*]\n"
        "requires:\n  - search\n---\n Body "
    )
    assert values == {
        "description": "Refunds: process them",
        "allowed_tools": ["*_admin", "read_*"],
        "requires": ["search"],
    }
    assert body == "Body"


@pytest.mark.parametrize(
    "content", ["body", "---\na: value\n", "---\nx: " + "a" * 65536 + "\n---"]
)
def test_standalone_frontmatter_rejects_missing_delimiters_and_excess_text(content: str) -> None:
    with pytest.raises(SkillParseError):
        parse_frontmatter(content)


def test_fallback_applies_to_the_whole_document_and_keeps_repeated_gates() -> None:
    tail = (
        "custom: [z, a]\nmissing: null\nnote: x # note\n"
        "allowed_tools: [first]\nallowed_tools: [second]\n---\n"
    )
    simple, _ = parse_frontmatter('---\ndescription: "Refunds: process them"\n' + tail)
    by_lines, _ = parse_frontmatter("---\ndescription: Refunds: process them\n" + tail)
    assert simple["allowed_tools"] == ["second"]
    assert simple["custom"] == ["z", "a"]
    assert simple["note"] == "x" and "missing" not in simple
    assert by_lines["allowed_tools"] == ["first", "second"]
    assert by_lines["custom"] == "[z, a]"
    assert by_lines["note"] == "x # note" and by_lines["missing"] == "null"


def test_fallback_keeps_empty_gates_and_appends_block_entries_verbatim() -> None:
    fields, _ = parse_frontmatter(
        "---\ndescription: Refunds: expand\nrequires: null\nallowed_tools: first\n"
        "allowed_tools:\n- first\n- ''\n- later\nother:\n- ''\n---"
    )
    assert fields["requires"] == []
    assert fields["allowed_tools"] == ["first", "first", "later"]
    assert fields["other"] == []
