"""The public block reader accepts its default, custom and empty tag sets."""

from __future__ import annotations

import pytest

from roomkit.tools import FENCED_TAGS, named_blocks


@pytest.mark.parametrize("tag", FENCED_TAGS)
def test_default_tags_replace_the_whole_block(tag: str) -> None:
    text = f"before <{tag}>untrusted text</{tag}> after"

    assert named_blocks(text) == f"before [{tag}] after"
    assert named_blocks(text, FENCED_TAGS) == f"before [{tag}] after"


def test_custom_tags_replace_only_the_selected_blocks() -> None:
    text = "a <custom>secret</custom> b <tool_result>kept</tool_result> c"

    assert named_blocks(text, ("custom",)) == "a [custom] b <tool_result>kept</tool_result> c"


def test_empty_tags_preserve_the_text() -> None:
    text = "a <tool_result>kept</tool_result> b <custom>also kept"

    assert named_blocks(text, ()) == text


@pytest.mark.parametrize(
    ("tags", "expected"),
    [(("CUSTOM", "custom"), "[CUSTOM]"), (("custom", "CUSTOM"), "[custom]")],
)
def test_custom_tags_are_applied_in_the_given_order(tags: tuple[str, ...], expected: str) -> None:
    assert named_blocks("<custom>x</custom>", tags) == expected
