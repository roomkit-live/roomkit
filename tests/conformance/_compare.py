"""How a run's records are compared with a scenario's expectations.

roomkit-specs ``conformance/README.md`` ("Expectations", "Normalization",
"Matching") defines it: each record is normalized, then matched against what
the scenario's ``expect`` names; what it does not name is not compared.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tests.conformance._match import MISSING, Names, framework_mismatches, mismatches
from tests.conformance._normalize import Normalizer, Observed, is_lifecycle_record

# A call's outcome, checked against what its step or its action expects.
Check = tuple[str, Mapping[str, Any] | None, dict[str, Any]]


def compare(expect: Mapping[str, Any], observed: Observed, checks: list[Check]) -> list[str]:
    """Every mismatch between what the run produced and what it should have."""
    normalizer = Normalizer(observed)
    # What the ``$as`` of the steps and of ``expect`` named (README, "Matching").
    names = Names()
    found: list[str] = []
    for path, expected, outcome in checks:
        found.extend(outcome_mismatches(expected, outcome, normalizer, path, names))
    found.extend(_lifecycle_never_delivered(observed))
    found.extend(_indexes_increase(observed))
    comparison = _Comparison(normalizer, observed, expect, names)
    comparison.records()
    comparison.hooks()
    comparison.channel_calls()
    comparison.side_effects()
    comparison.models()
    return found + comparison.found


class _Comparison:
    """The comparison of a run's records, the rooms named by their aliases."""

    def __init__(
        self,
        normalizer: Normalizer,
        observed: Observed,
        expect: Mapping[str, Any],
        names: Names,
    ):
        self.normalizer = normalizer
        self.names = names
        self.observed = observed
        self.expect = expect
        self.rooms = {normalizer.room(room_id): room_id for room_id in observed.timelines}
        self.found: list[str] = []

    def _room(self, alias: str) -> str:
        return self.rooms.get(alias, alias)

    def _created(self) -> list[Any]:
        """The records of the rooms the run created and the scenario does not
        name, in the order of their ``room_created``: ``$created`` in
        ``rooms`` (README, "Normalization")."""
        minted = sorted(
            (alias for alias in self.rooms if alias.startswith("$r")),
            key=lambda alias: int(alias[2:]),
        )
        return [self.observed.rooms.get(self.rooms[alias], MISSING) for alias in minted]

    def _check(self, expected: Any, actual: Any, path: str) -> None:
        self.found.extend(mismatches(expected, self.normalizer.value(actual), path, self.names))

    def records(self) -> None:
        for alias, expected in self.expect.get("timeline", {}).items():
            events = self.observed.timelines.get(self._room(alias), [])
            kept = [event for event in events if not is_lifecycle_record(event)]
            self._check(expected, kept, f"timeline.{alias}")
        for alias, expected in self.expect.get("rooms", {}).items():
            actual = (
                self._created()
                if alias == "$created"
                else self.observed.rooms.get(self._room(alias), MISSING)
            )
            self._check(expected, actual, f"rooms.{alias}")
        for alias, channels in self.expect.get("bindings", {}).items():
            held = self.observed.bindings.get(self._room(alias), {})
            for channel_id, expected in channels.items():
                actual = held.get(channel_id, MISSING)
                self._check(expected, actual, f"bindings.{alias}.{channel_id}")
        for alias, expected in self.expect.get("lifecycle", {}).items():
            events = self.observed.timelines.get(self._room(alias), [])
            records = [event for event in events if is_lifecycle_record(event)]
            # An implementation that writes no lifecycle record passes (§5.1).
            if records:
                self._check(expected, records, f"lifecycle.{alias}")
        grouped: dict[str, list[Any]] = {}
        for event in self.normalizer.value(self.observed.framework_events):
            grouped.setdefault(event.get("room_id", "*"), []).append(event)
        for alias, events in self.expect.get("framework_events", {}).items():
            actual = grouped.get(alias, [])
            path = f"framework_events.{alias}"
            self.found.extend(framework_mismatches(events, actual, path, self.names))
        ai_grouped: dict[str, list[Any]] = {}
        for event in self.normalizer.value(self.observed.ai_events):
            ai_grouped.setdefault(event.get("room_id", "*"), []).append(event)
        for alias, events in self.expect.get("ai_events", {}).items():
            actual = ai_grouped.get(alias, [])
            path = f"ai_events.{alias}"
            self.found.extend(framework_mismatches(events, actual, path, self.names))

    def hooks(self) -> None:
        for name, expected in self.expect.get("hooks", {}).items():
            self._check(expected, self.observed.hooks.get(name, []), f"hooks.{name}")
        if "hook_order" in self.expect:
            named = set(self.expect["hook_order"])
            order = [name for name in self.observed.hook_order if name in named]
            self._check(self.expect["hook_order"], order, "hook_order")

    def channel_calls(self) -> None:
        kinds = ("deliveries", "contexts", "streamed", "turns", "ingested", "compacted")
        for kind in (*kinds, "room_callbacks"):
            recorded = getattr(self.observed, kind)
            for alias, channels in self.expect.get(kind, {}).items():
                per_room = recorded.get(self._room(alias), {})
                for channel_id, expected in channels.items():
                    actual = per_room.get(channel_id, [])
                    self._check(expected, actual, f"{kind}.{alias}.{channel_id}")

    def models(self) -> None:
        if "resolver" in self.expect:
            self._check(self.expect["resolver"], self.observed.resolver, "resolver")
        if "router_calls" in self.expect:
            self._check(self.expect["router_calls"], self.observed.router_calls, "router_calls")
        if "media_resolver" in self.expect:
            expected = self.expect["media_resolver"]
            self._check(expected, self.observed.media_resolver, "media_resolver")
        for strategy_id, expected in self.expect.get("dispatch_turns", {}).items():
            actual = self.observed.dispatch_turns.get(strategy_id, [])
            self._check(expected, actual, f"dispatch_turns.{strategy_id}")
        for channel_id, expected in self.expect.get("models", {}).items():
            actual = self.observed.models.get(channel_id, [])
            self._check(expected, actual, f"models.{channel_id}")
        for channel_id, expected in self.expect.get("ai_routes", {}).items():
            actual = self.observed.ai_routes.get(channel_id, [])
            self._check(expected, actual, f"ai_routes.{channel_id}")
        for channel_id, expected in self.expect.get("acp_wire", {}).items():
            actual = self.observed.acp_wire.get(channel_id, [])
            self._check(expected, actual, f"acp_wire.{channel_id}")
        for executor_id, expected in self.expect.get("executors", {}).items():
            actual = self.observed.executors.get(executor_id, [])
            self._check(expected, actual, f"executors.{executor_id}")
        for kind in ("tool_audit", "session_audit"):
            if kind in self.expect:
                self._check(self.expect[kind], getattr(self.observed, kind), kind)

    def side_effects(self) -> None:
        for kind in (
            "observations",
            "tasks",
            "participants",
            "thinking_events",
            "text_events",
            "tool_events",
            "presence_events",
        ):
            recorded = getattr(self.observed, kind)
            for alias, expected in self.expect.get(kind, {}).items():
                self._check(expected, recorded.get(self._room(alias), []), f"{kind}.{alias}")
        for label, expected in self.expect.get("actions", {}).items():
            outcomes = self.observed.actions.get(label, [])
            path = f"actions.{label}"
            self.found.extend(_outcomes_of(expected, outcomes, self.normalizer, path, self.names))


def outcome_mismatches(
    expected: Mapping[str, Any] | None,
    outcome: dict[str, Any],
    normalizer: Normalizer,
    path: str,
    names: Names | None = None,
) -> list[str]:
    """How a call's outcome differs from what is expected of it."""
    names = names if names is not None else Names()
    error = outcome["error"]
    expected = expected or {}
    if "error" in expected:
        want = expected["error"]
        want = want if isinstance(want, dict) else {"name": want}
        if error is None:
            return [f"{path}: expected the error {want}, the call returned {outcome['result']!r}"]
        return mismatches(want, error, f"{path}.error", names)
    if error is not None:
        return [f"{path}: unexpected error {error}"]
    if "result" in expected:
        return mismatches(
            expected["result"], normalizer.value(outcome["result"]), f"{path}.result", names
        )
    return []


def _outcomes_of(
    expected: Any,
    outcomes: list[dict[str, Any]],
    normalizer: Normalizer,
    path: str,
    names: Names,
) -> list[str]:
    """A label's outcomes, in order, or in any order under ``$unordered``."""
    unordered = isinstance(expected, dict)
    wanted = expected["$unordered"] if unordered else expected
    if len(outcomes) != len(wanted):
        return [f"{path}: expected {len(wanted)} runs, got {len(outcomes)}"]
    if not unordered:
        found: list[str] = []
        for position, (want, got) in enumerate(zip(wanted, outcomes, strict=True)):
            found.extend(outcome_mismatches(want, got, normalizer, f"{path}[{position}]", names))
        return found
    remaining = list(outcomes)
    for want in wanted:
        match = None
        for got in remaining:
            trial = names.copy()
            if not outcome_mismatches(want, got, normalizer, path, trial):
                match = got
                names.take(trial)
                break
        if match is None:
            return [f"{path}: no outcome matches {want!r}"]
        remaining.remove(match)
    return []


def _indexes_increase(observed: Observed) -> list[str]:
    """A room's indexes are given out once each (§8.1)."""
    found: list[str] = []
    for room_id, events in observed.timelines.items():
        indexes = [event["index"] for event in events]
        if any(later <= earlier for earlier, later in zip(indexes, indexes[1:], strict=False)):
            found.append(f"timeline.{room_id}: an index is held twice: {indexes}")
    return found


def _lifecycle_never_delivered(observed: Observed) -> list[str]:
    """A lifecycle record (§5.1) is never delivered, nor handed to a turn."""
    records = {
        event["id"]
        for events in observed.timelines.values()
        for event in events
        if is_lifecycle_record(event)
    }
    found: list[str] = []
    for kind in ("deliveries", "turns"):
        for room_id, channels in getattr(observed, kind).items():
            for channel_id, events in channels.items():
                if any(event.get("id") in records for event in events):
                    found.append(f"{kind}.{room_id}.{channel_id}: lifecycle record delivered")
    return found
