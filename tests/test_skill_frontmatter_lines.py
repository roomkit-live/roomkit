"""A local SKILL.md frontmatter read line by line, one that is not YAML,
keeps what the skill gates and requires (RFC §24.1); of a text YAML reads too,
the line reader gates at least what YAML does."""

from __future__ import annotations

from pathlib import Path

import pytest

from roomkit.skills._frontmatter_lines import parse_lines
from roomkit.skills.errors import SkillParseError
from roomkit.skills.models import parse_tool_names
from roomkit.skills.parser import parse_frontmatter, split_frontmatter
from roomkit.skills.registry import SkillRegistry

RFC_EXAMPLE = """---
name: "web-search"
description: "Search the web for current information"
license: "MIT"
compatibility: "Needs network access"
allowed_tools:
  - "search_*"
  - "fetch_*"
requires:
  - search_web
---

## Instructions
"""


@pytest.fixture
def head() -> str:
    """The first lines of a frontmatter the line reader reads: the
    description's colon makes the text invalid YAML."""
    return "---\nname: refunds\ndescription: Refunds: process them\n"


def _by_lines(text: str) -> dict[str, str | list[str]]:
    """The frontmatter of *text* as the line reader reads it, YAML or not."""
    return parse_lines(split_frontmatter(text)[0])


def _write_skill(root: Path, text: str) -> None:
    skill_dir = root / "refunds"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(text + "Body\n", encoding="utf-8")


class TestALineByLineReadingKeepsItsLists:
    def test_a_key_followed_by_items(self, head: str) -> None:
        text = head + (
            'allowed_tools:\n  - "issue_refund"\n  - void_*  # partial refunds\n'
            "requires:\n- lookup_order\n---\nBody\n"
        )
        data, _ = parse_frontmatter(text)
        assert data["allowed_tools"] == ["issue_refund", "void_*"]
        assert data["requires"] == ["lookup_order"]

    def test_a_list_on_the_key_line(self, head: str) -> None:
        text = head + "allowed_tools: [issue_refund, 'void_*']\nrequires: [lookup_order]\n---\n"
        data, _ = parse_frontmatter(text)
        assert data["allowed_tools"] == ["issue_refund", "void_*"]
        assert data["requires"] == ["lookup_order"]

    def test_a_key_with_no_items_is_left_out(self, head: str) -> None:
        data, _ = parse_frontmatter(head + "allowed_tools:\nlicense: MIT\n---\n")
        assert "allowed_tools" not in data
        assert data["license"] == "MIT"

    def test_the_registry_keeps_the_tool_closed_until_activation(
        self, head: str, tmp_path: Path
    ) -> None:
        _write_skill(
            tmp_path, head + "allowed_tools:\n  - issue_refund\nrequires:\n  - lookup_order\n---\n"
        )
        registry = SkillRegistry()
        registry.discover(tmp_path)
        assert registry.gated_tool_names() == {"issue_refund"}
        assert registry.gated_tool_names({"refunds"}) == set()
        metadata = registry.get_metadata("refunds")
        assert metadata is not None
        assert metadata.required_tool_names == ["lookup_order"]


@pytest.mark.parametrize("key", ["allowed_tools", "requires"])
def test_a_line_it_cannot_place_under_a_list_key_refuses_the_skill(
    head: str, tmp_path: Path, key: str
) -> None:
    # Read without that line, the skill would leave open what it gates.
    _write_skill(tmp_path, head + f"{key}:\n  - issue_refund\n  scope: refunds\n---\n")
    with pytest.raises(
        SkillParseError, match=f"Parsing failed for refunds: .*{key} cannot be read line by line"
    ):
        SkillRegistry().discover(tmp_path)


@pytest.mark.parametrize(
    "lines",
    [
        "allowed_tools: [\n  issue_refund,\n  void_*\n]",
        "allowed_tools: >-\n  issue_refund,\n  void_*",
        "allowed_tools: |\n  issue_refund",
        "allowed_tools:\nissue_refund",
        "allowed_tools: issue_refund\n  - void_*",
        "allowed_tools: [issue_refund, [void_*]]",
        "allowed_tools: {issue_refund: true}",
        "allowed_tools:\n  - &money issue_refund",
        "requires:\n  - !tool lookup_order",
        "allowed_tools: 3",
        "allowed_tools: [on, 010, 1_000, yes, 0x1A]",
        "requires:\n  - lookup_order\n  - off",
        "allowed_tools: issue_refund: money",
        "allowed_tools: lookup_order, issue_refund: money",
        "allowed_tools:\n  - issue_refund: money",
        "allowed_tools: [lookup_order, issue_refund: money]",
        "requires:\n  - lookup_order:\tfirst",
        "allowed_tools: issue_refund:",
        "allowed_tools:\n  - issue_refund:",
        "requires: [lookup_order:, issue_refund]",
        "allowed_tools: [lookup, 0b_]",
        "allowed_tools: [lookup, ?admin]",
        "requires: [?lookup_order]",
    ],
    ids=[
        "flow over lines",
        "folded",
        "literal",
        "bare line",
        "item after a value",
        "nested flow",
        "mapping",
        "anchor",
        "tag",
        "number",
        "names YAML retypes",
        "item YAML retypes",
        "value holding a key",
        "text holding a key",
        "item holding a key",
        "flow item holding a key",
        "item holding a key after a tab",
        "value ending in a colon",
        "item ending in a colon",
        "flow item ending in a colon",
        "binary prefix YAML types",
        "flow item YAML reads as a key",
        "requires flow item YAML reads as a key",
    ],
)
def test_a_gating_key_a_line_cannot_hold_whole_refuses_the_skill(head: str, lines: str) -> None:
    with pytest.raises(SkillParseError, match="cannot be read line by line"):
        parse_frontmatter(head + lines + "\n---\nBody\n")


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        ('allowed_tools:\n  - "issue_refund"\n  - void_*', ["issue_refund", "void_*"]),
        ("allowed_tools:\n- issue_refund  # money", ["issue_refund"]),
        ("allowed_tools: [issue_refund, 'void_*']", ["issue_refund", "void_*"]),
        ("allowed_tools: [issue_refund, void_*]  # money", ["issue_refund", "void_*"]),
        ("allowed_tools: issue_refund, void_*", "issue_refund, void_*"),
        ('allowed_tools: "issue_refund, void_*"', "issue_refund, void_*"),
        ("allowed_tools: [\"on\", '010', 3d_render, v1.2]", ["on", "010", "3d_render", "v1.2"]),
        ('allowed_tools:\n  - "issue_refund: money"', ["issue_refund: money"]),
        ("allowed_tools: 'issue_refund: money'", "issue_refund: money"),
        ("allowed_tools: [issue_refund:money, 'void: *']", ["issue_refund:money", "void: *"]),
        ("allowed_tools:\n  - ?admin", ["?admin"]),
        (
            "allowed_tools: [yEs, 08, 1e5, 0:30, 10-20, 2026-1-5]",
            ["yEs", "08", "1e5", "0:30", "10-20", "2026-1-5"],
        ),
    ],
    ids=[
        "block",
        "block with comment",
        "flow",
        "flow with comment",
        "text",
        "quoted text",
        "quoted names YAML would retype",
        "quoted item holding a colon",
        "quoted text holding a colon",
        "colons YAML keeps in a name",
        "block item starting with a question mark",
        "names YAML keeps as texts",
    ],
)
def test_each_form_reads_as_yaml_reads_it(lines: str, expected: object) -> None:
    text = "---\nname: refunds\ndescription: d\n" + lines + "\n---\n"
    as_yaml, _ = parse_frontmatter(text)
    by_lines = _by_lines(text)
    assert by_lines == as_yaml
    assert by_lines["allowed_tools"] == expected


@pytest.mark.parametrize(
    "first_lines",
    [
        "---\nname: refunds\ndescription: Refunds: process them\n",
        "---\nname: refunds\ndescription: Refunds\n",
    ],
    ids=["text YAML refuses", "YAML but for the star"],
)
@pytest.mark.parametrize(
    ("lines", "key", "expected"),
    [
        ("allowed_tools: *", "allowed_tools", "*"),
        ("allowed_tools: *_admin", "allowed_tools", "*_admin"),
        ("allowed_tools:\n  - *_admin\n  - search_*", "allowed_tools", ["*_admin", "search_*"]),
        ("allowed_tools: [*_admin, search_*]", "allowed_tools", ["*_admin", "search_*"]),
        ("license: *Use with care* when refunding", "license", "*Use with care* when refunding"),
    ],
    ids=["every tool", "one glob", "block", "flow", "text"],
)
def test_a_star_naming_no_anchor_is_read_line_by_line(
    first_lines: str, lines: str, key: str, expected: object
) -> None:
    # YAML reads a leading * as an alias and fails; no line defines an
    # anchor, so the alias cannot expand and is not refused as one: the line
    # reader takes the glob or the text it is, even when the star is the only
    # thing YAML cannot read.
    data, _ = parse_frontmatter(first_lines + lines + "\n---\n")
    assert data[key] == expected


def test_an_alias_to_an_anchor_is_still_refused() -> None:
    text = "---\nname: refunds\ndescription: &d Refunds\nlicense: *d\n---\n"
    with pytest.raises(SkillParseError, match="YAML alias"):
        parse_frontmatter(text)


@pytest.mark.parametrize("item", ["~", "null", "NULL"])
def test_a_null_item_reads_as_nothing(head: str, item: str) -> None:
    # YAML reads these as no value, so a line must not read them as a name
    # a skill then requires and nothing serves.
    data, _ = parse_frontmatter(head + f"requires:\n  - {item}\n  - lookup_order\n---\n")
    assert data["requires"] == ["lookup_order"]


def test_a_bracketed_description_stays_whole() -> None:
    # Only allowed_tools and requires read [a, b] as a list; this one is not
    # YAML either, so it goes to the line reader.
    text = "---\nname: refunds\ndescription: [Beta] Refunds, returns: see [docs]\n---\n"
    data, _ = parse_frontmatter(text)
    assert data["description"] == "[Beta] Refunds, returns: see [docs]"


@pytest.mark.parametrize(
    "lines",
    [
        "allowed_tools: {issue_refund: true}",
        "allowed_tools:\n  - [a, b]",
        "requires: [a, [b]]",
        "allowed_tools: 3",
        "allowed_tools: true",
        "allowed_tools: 2026-11-27",
        "requires: [1, 2]",
        "allowed_tools: [on, 010, 1_000, yes, 0x1A]",
        "requires:\n  - lookup_order\n  - off",
        "allowed_tools:\n  - issue_refund: money",
        "allowed_tools: [lookup_order, issue_refund: money]",
        "allowed_tools: [lookup, ?admin]",
    ],
    ids=[
        "mapping",
        "nested list",
        "nested flow",
        "number",
        "boolean",
        "date",
        "list of numbers",
        "names YAML retypes",
        "item YAML retypes",
        "item holding a key",
        "flow item holding a key",
        "flow item YAML reads as a key",
    ],
)
def test_yaml_refuses_a_gating_key_that_is_not_a_list_of_texts(tmp_path: Path, lines: str) -> None:
    # As one text, it would gate nothing; as the text YAML retypes it into
    # (`on` read as True, `010` as 8), it would gate another name and leave
    # the one written open before any activation (RFC §24.1).
    _write_skill(tmp_path, "---\nname: refunds\ndescription: d\n" + lines + "\n---\n")
    with pytest.raises(
        SkillParseError, match="Parsing failed for refunds: .*neither a text nor a list of texts"
    ):
        SkillRegistry().discover(tmp_path)


@pytest.mark.parametrize(
    "lines",
    [
        "allowed_tools: [issue_refund]\nmetadata:\n  allowed_tools:\n",
        "allowed_tools: [issue_refund]\nmetadata:\n  allowed_tools: other\n",
        "requires:\n  - lookup_order\nmetadata:\n  requires: []\n",
        '"allowed_tools": [issue_refund]\n',
        "'allowed_tools':\n  - issue_refund\n",
        'allowed_tools: "issue_refund #1, void_*"\n',
        "allowed_tools:\n  - 'issue_refund # money'\n",
        'allowed_tools: ["issue,refund", void_*]\n',
        "allowed_tools:\n  - - issue_refund\n",
        "allowed_tools:\n  - ? issue_refund\n",
        "allowed_tools:\n  - issue_refund\n  - #void_*\n",
        "allowed_tools: issue_refund # money, void_*\n",
        " allowed_tools: [issue_refund]\n name: refunds\n",
        "allowed_tools: [issue_refund]\nmetadata: {a: 1,\nallowed_tools: x}\n",
        "allowed_tools: [issue_refund]\nnotes: [a,\nallowed_tools: x]\n",
        'allowed_tools: [issue_refund]\nnotes: "a\nallowed_tools:\nlicense: MIT"\n',
        "allowed_tools: [issue_refund]\nnotes: 'a\nallowed_tools: x'\n",
        "? allowed_tools\n: [issue_refund]\n",
        "!!str allowed_tools: [issue_refund]\n",
        "&key allowed_tools: [issue_refund]\n",
        "<<: {allowed_tools: [issue_refund]}\n",
        '"allowed\\x5ftools": [issue_refund]\n',
        'allowed_tools:\n  - "issue\\x5frefund"\n',
        'allowed_tools: ["issue\\u005frefund"]\n',
        "requires: '\"[ab]'\n",
        "requires:\n - lookup_order\n  - refund_x\n",
        "allowed_tools: [on, 010]\n",
        "allowed_tools:\n  - issue_refund: money\n",
        "allowed_tools: [issue_refund: money]\n",
        "requires:\n  - lookup_order: first\n",
        "allowed_tools:\n  - issue_refund:\n",
        "allowed_tools:\n  - issue_refund:\tmoney\n",
        # libyaml does not read [a:]; the pure-Python loader reads a mapping.
        "allowed_tools: [issue_refund:]\n",
        "allowed_tools: [lookup, ?admin]\n",
    ],
    ids=[
        "nested empty key",
        "nested value",
        "nested requires",
        "double-quoted key",
        "single-quoted key",
        "quoted value with a hash",
        "quoted item with a hash",
        "quoted item with a comma",
        "nested sequence",
        "complex key",
        "comment item",
        "comment after a value",
        "indented document",
        "flow mapping over lines",
        "flow sequence over lines",
        "double-quoted text over lines",
        "single-quoted text over lines",
        "explicit key",
        "tagged key",
        "anchored key",
        "merge key",
        "escaped key",
        "escaped item",
        "escaped flow item",
        "quote in a quoted text",
        "item YAML folds into the one before",
        "names YAML retypes",
        "item holding a key",
        "flow item holding a key",
        "requires item holding a key",
        "item ending in a colon",
        "item holding a key after a tab",
        "flow item ending in a colon",
        "flow item YAML reads as a key",
    ],
)
def test_the_line_reader_never_gates_less_than_yaml(lines: str) -> None:
    # RFC §24.1: of a frontmatter that is also YAML, the line-by-line reading
    # gates and requires at least what YAML does, or refuses the skill.
    text = "---\nname: refunds\ndescription: d\n" + lines + "---\n"
    try:
        as_yaml, _ = parse_frontmatter(text)
    except SkillParseError:
        as_yaml = None
    try:
        by_lines = _by_lines(text)
    except SkillParseError:
        return
    assert as_yaml is not None
    for key in ("allowed_tools", "requires"):
        missing = set(parse_tool_names(as_yaml.get(key))) - set(
            parse_tool_names(by_lines.get(key))
        )
        assert not missing, key


def test_the_rfc_example_reads_the_same_by_lines() -> None:
    as_yaml, _ = parse_frontmatter(RFC_EXAMPLE)
    by_lines = _by_lines(RFC_EXAMPLE)
    assert by_lines == as_yaml
    assert by_lines["allowed_tools"] == ["search_*", "fetch_*"]
