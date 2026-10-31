"""Of a SKILL.md frontmatter that is also YAML, the line-by-line reading gates
and requires at least what YAML does, or refuses the skill (RFC §24.1).

The frontmatters are generated: every gating form and quoting style, keys
behind quotes, tags and anchors, values that YAML reads over several lines
around a line that looks like a gating key, then the same with random
insertions and deletions of YAML's own characters. A seed keeps them fixed.
"""

from __future__ import annotations

import random

import pytest

from roomkit.skills._frontmatter_lines import parse_lines
from roomkit.skills.errors import SkillError
from roomkit.skills.models import parse_tool_names
from roomkit.skills.parser import parse_frontmatter, read_frontmatter_yaml, split_frontmatter

KEYS = ("allowed_tools", "requires")
NAMES = (
    "issue_refund",
    "void_*",
    "*_admin",
    "a-b",
    "it's",
    "x y",
    "x#y",
    "x, y",
    "[ab]",
    "[sw]earch_*",
)
SPECIALS = [*" :-#'\"[]{},\n\t&!?|>\\", "\n  ", "\n- ", ": ", " #"]


def _quoted(rng: random.Random, name: str) -> str:
    style = rng.choice(["plain", "single", "double", "escaped"])
    if style == "plain":
        return name
    if style == "single":
        return "'" + name.replace("'", "''") + "'"
    if style == "double":
        return '"' + name.replace('"', '\\"') + '"'
    return '"' + name.replace("_", "\\x5f").replace('"', '\\"') + '"'


def _comment(rng: random.Random) -> str:
    return rng.choice(["", "", " # note", "  #x"])


def _gating(rng: random.Random, key: str) -> list[str]:
    names = rng.sample(NAMES, rng.randint(1, 3))
    head = rng.choice([key, key, f'"{key}"', f"'{key}'", f"!!str {key}", f"&k {key}"])
    form = rng.choice(["flow", "flow over lines", "block", "block at 0", "text", "quoted lines"])
    if form == "flow":
        return [f"{head}: [" + ", ".join(_quoted(rng, n) for n in names) + "]" + _comment(rng)]
    if form == "flow over lines":
        return [f"{head}: [", *(f"  {_quoted(rng, n)}," for n in names), "]"]
    if form == "block":
        items = [f"  - {_quoted(rng, n)}{_comment(rng)}" for n in names]
        return [f"{head}:{_comment(rng)}", *items]
    if form == "block at 0":
        return [f"{head}:", *(f"- {_quoted(rng, n)}" for n in names)]
    if form == "quoted lines":
        return [f'{head}: "{names[0]},', f'  {names[-1]}"']
    return [f"{head}: {_quoted(rng, ', '.join(names))}{_comment(rng)}"]


def _other(rng: random.Random) -> list[str]:
    key = rng.choice(["license", "notes", "metadata", "compatibility"])
    hidden = rng.choice(KEYS)
    return rng.choice(
        [
            [f"{key}: MIT"],
            [f'{key}: "a', f"{hidden}:", 'end"'],
            [f"{key}: 'a", f"{hidden}: zz'"],
            [f"{key}: {{a: 1,", f"{hidden}: zz}}"],
            [f"{key}:", f"  {hidden}:", "    - zz"],
            [f"{key}: |", f"  {hidden}: zz"],
        ]
    )


def _document(rng: random.Random) -> str:
    parts = [_gating(rng, key) for key in KEYS if rng.random() < 0.8]
    parts += [_other(rng) for _ in range(rng.randint(0, 3))]
    rng.shuffle(parts)
    lines = ["name: refunds", "description: d", *(line for part in parts for line in part)]
    return "---\n" + "\n".join(lines) + "\n---\n"


def _mutated(rng: random.Random, text: str) -> str:
    body = text[4:-5]
    for _ in range(rng.randint(1, 4)):
        at = rng.randrange(len(body) + 1)
        if rng.random() < 0.6:
            body = body[:at] + rng.choice(SPECIALS) + body[at:]
        else:
            body = body[:at] + body[at + 1 :]
    return "---\n" + body + "\n---\n"


def _read_as_yaml(text: str) -> bool:
    """Whether YAML itself reads *text*, rather than the line reader it
    falls back to."""
    try:
        read_frontmatter_yaml(split_frontmatter(text)[0])
    except SkillError:
        return False
    return True


def _gating_of(text: str, *, by_lines: bool) -> dict[str, set[str]] | None:
    """What *text* gates and requires, read as SKILL.md is or by the line
    reader alone; ``None`` when the reading refuses it."""
    try:
        data = parse_lines(split_frontmatter(text)[0]) if by_lines else parse_frontmatter(text)[0]
    except SkillError:
        return None
    return {
        "allowed_tools": set(parse_tool_names(data.get("allowed_tools"))),
        "requires": set(parse_tool_names(data.get("requires"))),
    }


@pytest.mark.parametrize("mutate", [False, True], ids=["generated", "mutated"])
def test_generated_frontmatters_never_gate_less_on_the_line_reader(mutate: bool) -> None:
    rng = random.Random(693)
    texts = [_document(rng) for _ in range(3000)]
    if mutate:
        texts = [_mutated(rng, text) for text in texts]
    compared = 0
    for text in filter(_read_as_yaml, texts):
        yaml_gating = _gating_of(text, by_lines=False)
        line_gating = _gating_of(text, by_lines=True)
        if yaml_gating is None or line_gating is None:
            continue
        compared += any(yaml_gating.values())
        for key in KEYS:
            assert yaml_gating[key] <= line_gating[key], (key, text)
    assert compared > 100
