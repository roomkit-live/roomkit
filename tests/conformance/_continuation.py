"""A host continuation callback and the exact text it receives."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


class ContinuationFailedError(Exception):
    """A configured failure from the host policy, distinct from fixture errors."""


def policy_problems(spec: Any) -> list[str]:
    """Reject malformed controls before entering the channel's error boundary."""
    if not isinstance(spec, Mapping):
        return ["expected an object"]
    problems = [f"unknown member {key}" for key in spec if key not in {"prefix", "nudge", "error"}]
    if not isinstance(spec.get("prefix"), str):
        problems.append("prefix must be a string")
    if ("nudge" in spec) == ("error" in spec):
        problems.append("exactly one of nudge and error is required")
    if "nudge" in spec and spec["nudge"] is not None and not isinstance(spec["nudge"], str):
        problems.append("nudge must be a string or null")
    if "error" in spec and not isinstance(spec["error"], str):
        problems.append("error must be a string")
    return problems


def scripted_continuation(
    spec: Mapping[str, Any], calls: list[str]
) -> Callable[[str], str | None]:
    """Record every invocation, then answer only when the text matches the prefix."""
    if problems := policy_problems(spec):
        raise ValueError("continuation: " + "; ".join(problems))
    prefix = spec["prefix"]
    nudge = spec.get("nudge")
    error = spec.get("error")

    def policy(text: str) -> str | None:
        calls.append(text)
        if not text.startswith(prefix):
            return None
        if error is not None:
            raise ContinuationFailedError(error)
        return nudge

    return policy


def continuation_options(
    channel: Mapping[str, Any], observations: dict[str, list[str]] | None
) -> dict[str, Any]:
    """The channel's constructor argument with its required observation sink."""
    if "continuation" not in channel:
        return {}
    if observations is None:
        raise AssertionError("continuation: no observation sink")
    calls = observations.setdefault(channel["id"], [])
    return {"continuation": scripted_continuation(channel["continuation"], calls)}


def channel_problems(channel: Mapping[str, Any], path: str) -> list[str]:
    """Fixture errors qualified by their channel's location in the scenario."""
    if "continuation" not in channel:
        return []
    return [f"{path}.continuation: {p}" for p in policy_problems(channel["continuation"])]
