"""A SKILL.md frontmatter read line by line, for one that is not YAML
(RFC §24.1): ``key: value`` lines, and the two YAML list forms, read whole or
refused for the keys that gate tools."""

from __future__ import annotations

import re

from roomkit.skills.errors import SkillParseError
from roomkit.skills.models import parse_tool_names

GATING_KEYS = ("allowed_tools", "requires")
"""What a skill gates and what it requires: whichever reader, their value is
read whole, a text or a list of texts, or the skill is refused (RFC §24.1)."""

_KEY_LINE = re.compile(r"""^(["']?)(\w[\w-]*)\1\s*:\s*(.*)$""")
_ITEM_LINE = re.compile(r"^\s*-(?:\s+(.*))?$")
_COMMENT = re.compile(r"(?:^|\s)#")
_HIDDEN_KEY = re.compile(r"""^(?:\?(?:\s|$)|[!&]|<<|"[^"]*\\)""")
"""How a top-level line starts when YAML may read a key there that this
reading cannot name: an explicit key, a tag or an anchor on the key, a merge
key, or an escape in a double-quoted key."""
_NULLS = ("~", "null", "Null", "NULL")
"""The plain texts YAML reads as no value."""
_NOT_A_PLAIN_TEXT = re.compile(r"^(?:[\[\]{}|>&!%@`,]|[-?:](?:\s|$))")
"""How a value starts when YAML reads it as more than a plain text: a
collection, a block scalar, an anchor, a tag, a reserved indicator, or a
nested item, key or entry. A leading ``*`` stays text: no line defines an
anchor for it to name, and ``*`` or ``*_admin`` is a glob a skill gates."""


def parse_lines(fm_text: str) -> dict[str, str | list[str]]:
    """Read the top-level ``key: value`` lines of a frontmatter, a key alone
    on its line followed by ``- item`` lines as a list (RFC §24.1).

    Raises:
        SkillParseError: ``allowed_tools`` or ``requires`` is not read whole,
            or a line holds a key YAML reads where this reading cannot:
            reading the skill without it would leave open what it gates.
    """
    data: dict[str, str | list[str]] = {}
    open_key: str | None = None  # a key alone on its line, its items to follow
    list_key: str | None = None  # allowed_tools or requires, read last at the top
    item_indent: int | None = None  # where the open key's first item starts
    for line in fm_text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        item = _ITEM_LINE.match(line)
        if item and open_key is not None:
            indent = len(line) - len(line.lstrip())
            # YAML folds an item indented otherwise into the one before it.
            if open_key in GATING_KEYS and item_indent not in (None, indent):
                raise _unread(open_key)
            item_indent = indent
            _append_item(data, open_key, item[1])
            continue
        # A line indented under a key belongs to that key's value: read as a
        # key of its own, it could replace what the top level gates.
        read = None if line[:1].isspace() else _read_key_line(data, line)
        if list_key is not None and read is None:
            raise _unread(list_key)
        key, is_open = read or (None, False)
        open_key = key if is_open else None
        list_key = key if key in GATING_KEYS else None
        item_indent = None
    return data


def _read_key_line(data: dict[str, str | list[str]], line: str) -> tuple[str, bool] | None:
    """The key a ``key: value`` *line* names, and whether ``- item`` lines
    hold its value; a value on the line itself is recorded in *data*.

    Raises:
        SkillParseError: The line holds a key YAML reads where this reading
            cannot, or a gating value it cannot read whole.
    """
    match = _KEY_LINE.match(line)
    if not match:
        if _HIDDEN_KEY.match(line):
            raise SkillParseError("the frontmatter holds a key it cannot read line by line")
        return None
    key, value = match[2], match[3].strip()
    if key not in GATING_KEYS:
        if value:
            data[key] = value.strip("\"'")
        else:
            data.pop(key, None)
        return key, not value
    value = _entry_text(key, value)
    if value:
        _add_gating(data, key, _list_value(key, value))
    return key, not value


def _add_gating(data: dict[str, str | list[str]], key: str, value: str | list[str]) -> None:
    """Record *value* for a gating *key*, joined to what an earlier line of
    that key gave: a line YAML reads as the rest of another value (a flow
    collection or a quoted text over several lines) never takes back what
    the top level gates."""
    earlier = data.get(key)
    if earlier is None:
        data[key] = value
        return
    data[key] = list(dict.fromkeys([*parse_tool_names(earlier), *parse_tool_names(value)]))


def _list_value(key: str, text: str) -> str | list[str]:
    """The value of an ``allowed_tools`` or ``requires`` line: ``[a, b]`` as
    a list, a plain or quoted text as text, a YAML null as nothing.

    Raises:
        SkillParseError: The value is YAML a line cannot hold whole.
    """
    if text in _NULLS:
        return []
    if text.startswith("[") and text.endswith("]") and not any(c in text[1:-1] for c in "[]{}"):
        return [item for part in text[1:-1].split(",") for item in _list_item(key, part)]
    if _NOT_A_PLAIN_TEXT.match(text):
        raise _unread(key)
    return _unquoted(text)


def _append_item(data: dict[str, str | list[str]], key: str, text: str | None) -> None:
    """Add what a ``- item`` line holds to the list of *key*."""
    items = data.get(key)
    if not isinstance(items, list):
        items = parse_tool_names(items) if items else []
        data[key] = items
    items.extend(_list_item(key, text))


def _list_item(key: str, text: str | None) -> list[str]:
    """An item of *key*'s list as text, its trailing ``# comment`` and quotes
    stripped: none when nothing is left.

    Raises:
        SkillParseError: An item of ``allowed_tools`` or ``requires`` is YAML
            a line cannot hold (a collection, an anchor, a tag, a nested item).
    """
    item = _entry_text(key, text or "")
    if key in GATING_KEYS and _NOT_A_PLAIN_TEXT.match(item):
        raise _unread(key)
    item = "" if item in _NULLS else _unquoted(item)
    return [item] if item else []


def _unquoted(text: str) -> str:
    """*text* without the one pair of quotes around it, as YAML reads a
    quoted text (``''`` inside single quotes standing for one quote)."""
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        inner = text[1:-1]
        return inner.replace("''", "'") if text[0] == "'" else inner
    return text


def _unread(key: str) -> SkillParseError:
    """The refusal of a gating key a line-by-line reading cannot read whole."""
    return SkillParseError(f"the frontmatter's {key} cannot be read line by line")


def _entry_text(key: str, text: str) -> str:
    """The text a value or an item holds: up to its YAML comment (a ``#``
    first or after a space), or a quoted text whole whatever it holds; a
    gating one this reading cannot read whole is refused.

    Raises:
        SkillParseError: A quoted value or item of ``allowed_tools`` or
            ``requires`` does not close, holds a ``\\`` escape this reading
            does not decode, or has more than a comment after it.
    """
    text = text.strip()
    if text[:1] in ("'", '"'):
        closing = text.find(text[0], 1)
        after = text[closing + 1 :].strip() if closing > 0 else "unclosed"
        whole = not after or after.startswith("#")
        escaped = text[0] == '"' and "\\" in text[: closing + 1]
        if key in GATING_KEYS and (escaped or not whole):
            raise _unread(key)
        return text[: closing + 1] if whole else text
    comment = _COMMENT.search(text)
    return (text[: comment.start()] if comment else text).strip()
