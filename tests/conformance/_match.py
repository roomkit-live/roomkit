"""How a scenario's expectation matches what a run produced.

The rules are the suite's, written in roomkit-specs ``conformance/README.md``
("Matching"): an object names the members it checks, a list matches element
by element, and a few operators (``"$any"``, ``$unordered``, ``$absent``,
``$present``, ``$contains``, ``$lacks``) loosen a member. ``$as`` names a
value, which every ``$as`` of that name in the scenario matches (``Names``).
"""

from __future__ import annotations

from typing import Any


class _Missing:
    """A member the actual object does not hold."""

    def __repr__(self) -> str:
        return "<missing>"


MISSING = _Missing()


class Names:
    """The values the ``$as`` of one scenario's comparison named so far: a
    name holds the first value it matched, and two names never hold equal
    values (README, "Matching")."""

    def __init__(self, held: dict[str, Any] | None = None) -> None:
        self.held: dict[str, Any] = dict(held or {})

    def copy(self) -> Names:
        return Names(self.held)

    def take(self, other: Names) -> None:
        """Holds what *other*, a copy that matched, holds."""
        self.held = dict(other.held)

    def match(self, name: str, actual: Any, path: str) -> list[str]:
        if actual is None or actual is MISSING:
            return [f"{path}: absent"]
        if name in self.held:
            if _equal(self.held[name], actual):
                return []
            return [f"{path}: {_short(actual)} is not ${name}, {_short(self.held[name])}"]
        for other, value in self.held.items():
            if _equal(value, actual):
                return [f"{path}: {_short(actual)} is ${other} already, not ${name}"]
        self.held[name] = actual
        return []


def _equal(held: Any, actual: Any) -> bool:
    """Whether two values a run gave are the same: scalars as matching
    compares them, a list or an object member by member."""
    if isinstance(held, dict | list) or isinstance(actual, dict | list):
        return held == actual
    return _same_scalar(held, actual)


def _short(value: Any) -> str:
    """*value* as a mismatch shows it, cut to a readable length."""
    shown = repr(value)
    return shown if len(shown) <= 240 else shown[:240] + "..."


def mismatches(
    expected: Any, actual: Any, path: str = "$", names: Names | None = None
) -> list[str]:
    """Every place where *actual* does not match *expected*, as ``path: why``;
    *names* holds what the comparison's ``$as`` named so far."""
    names = names if names is not None else Names()
    if expected == "$any":
        return []
    if isinstance(expected, dict):
        return _object(expected, actual, path, names)
    if isinstance(expected, list):
        return _list(expected, actual, path, names)
    if expected is None:
        return (
            []
            if actual is None or actual is MISSING
            else [f"{path}: {_short(actual)} is not null"]
        )
    if _same_scalar(expected, actual):
        return []
    return [f"{path}: expected {expected!r}, got {_short(actual)}"]


def _same_scalar(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(expected) is type(actual) and expected == actual
    if isinstance(expected, int | float) and isinstance(actual, int | float):
        return float(expected) == float(actual)
    return type(expected) is type(actual) and expected == actual


def _object(expected: dict[str, Any], actual: Any, path: str, names: Names) -> list[str]:
    if "$unordered" in expected:
        return _unordered(expected["$unordered"], actual, path, names)
    if "$as" in expected:
        return names.match(expected["$as"], actual, path)
    if expected.get("$absent") is True:
        return (
            [] if actual is None or actual is MISSING else [f"{path}: {_short(actual)} is present"]
        )
    if expected.get("$present") is True:
        return [f"{path}: absent"] if actual is None or actual is MISSING else []
    if "$lacks" in expected:
        if isinstance(actual, str) and expected["$lacks"] not in actual:
            return []
        return [f"{path}: {_short(actual)} contains {expected['$lacks']!r}"]
    if "$contains" in expected:
        if isinstance(actual, str) and expected["$contains"] in actual:
            return []
        return [f"{path}: {_short(actual)} does not contain {expected['$contains']!r}"]
    if not isinstance(actual, dict):
        return [f"{path}: expected an object, got {_short(actual)}"]
    found: list[str] = []
    for key, value in expected.items():
        found.extend(mismatches(value, actual.get(key, MISSING), f"{path}.{key}", names))
    return found


def _list(expected: list[Any], actual: Any, path: str, names: Names) -> list[str]:
    if not isinstance(actual, list):
        return [f"{path}: expected a list, got {_short(actual)}"]
    if len(expected) != len(actual):
        return [f"{path}: expected {len(expected)} items, got {len(actual)}: {_short(actual)}"]
    found: list[str] = []
    for position, (want, got) in enumerate(zip(expected, actual, strict=True)):
        found.extend(mismatches(want, got, f"{path}[{position}]", names))
    return found


def _unordered(expected: list[Any], actual: Any, path: str, names: Names) -> list[str]:
    if not isinstance(actual, list):
        return [f"{path}: expected a list, got {_short(actual)}"]
    if len(expected) != len(actual):
        return [f"{path}: expected {len(expected)} items in any order, got {_short(actual)}"]
    matched = _assign(expected, actual, 0, set(), names)
    if matched is not None:
        names.take(matched)
        return []
    return [f"{path}: no order of {_short(actual)} matches {_short(expected)}"]


def _assign(
    expected: list[Any], actual: list[Any], position: int, used: set[int], names: Names
) -> Names | None:
    """The names an order matching *expected* from *position* on holds, or
    ``None`` when no order matches: each try names on a copy."""
    if position == len(expected):
        return names
    for index, item in enumerate(actual):
        if index in used:
            continue
        trial = names.copy()
        if mismatches(expected[position], item, "$", trial):
            continue
        used.add(index)
        matched = _assign(expected, actual, position + 1, used, trial)
        if matched is not None:
            return matched
        used.discard(index)
    return None


def framework_mismatches(
    expected: list[Any], actual: list[Any], path: str, names: Names | None = None
) -> list[str]:
    """A room's framework events: in order, a ``$unordered`` group matching as
    many consecutive events in any order, and a bare string naming the type."""
    names = names if names is not None else Names()
    found: list[str] = []
    cursor = 0
    for position, item in enumerate(expected):
        if isinstance(item, dict) and "$unordered" in item:
            group = [_as_event(entry) for entry in item["$unordered"]]
            window = actual[cursor : cursor + len(group)]
            found.extend(_unordered(group, window, f"{path}[{position}]", names))
            cursor += len(group)
            continue
        if cursor >= len(actual):
            found.append(f"{path}[{position}]: expected {item!r}, got nothing more")
            continue
        found.extend(mismatches(_as_event(item), actual[cursor], f"{path}[{position}]", names))
        cursor += 1
    if cursor < len(actual):
        found.append(f"{path}: unexpected events {_short(actual[cursor:])}")
    return found


def _as_event(item: Any) -> Any:
    return {"event": item} if isinstance(item, str) else item
