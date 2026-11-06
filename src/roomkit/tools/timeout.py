"""How long a tool handler may take to answer one call (RFC §21.6).

Every channel waits on its handlers through :func:`answer_within`, so a
handler that never answers costs one call, never the turn: the text loops,
the speech-to-speech entries and the conference alike.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass, field

from roomkit.core.exceptions import ToolTimeoutError


@dataclass(frozen=True, slots=True)
class ToolTimeouts:
    """A channel's bound per call: its default, and a bound per tool name."""

    default: float | None
    """Seconds a call may take; ``None`` leaves calls unbounded."""

    per_tool: Mapping[str, float | None] = field(default_factory=dict)
    """A tool's own bound, above everything else; ``None`` leaves it unbounded."""

    def __post_init__(self) -> None:
        bounds = {"tool_timeout_seconds": self.default, **self.per_tool}
        for key, bound in bounds.items():
            if bound is not None and bound <= 0:
                raise ValueError(f"tool timeout for {key!r} must be positive or None, got {bound}")

    def overridden(self, per_tool: Mapping[str, float | None]) -> ToolTimeouts:
        """These bounds under *per_tool*'s, a name *per_tool* gives keeping
        its own; a bound that is not positive fails here."""
        return ToolTimeouts(self.default, {**self.per_tool, **per_tool})

    def for_call(self, name: str, *, waits: bool = False) -> float | None:
        """The bound of a call to *name*.

        A tool that waits on another agent or on a person by design (*waits*)
        keeps its own bound rather than the channel's default.
        """
        if name in self.per_tool:
            return self.per_tool[name]
        return None if waits else self.default


async def answer_within[T](timeout: float | None, name: str, answer: Awaitable[T]) -> T:
    """Wait for a handler's *answer* to a call to *name*, cancelled past *timeout*.

    Raises :class:`~roomkit.core.exceptions.ToolTimeoutError` when the bound
    expired. A ``TimeoutError`` the handler raised itself passes through as its
    own failure.
    """
    if timeout is None:
        return await answer
    bound = asyncio.timeout(timeout)
    try:
        async with bound:
            return await answer
    except TimeoutError:
        if bound.expired():
            raise ToolTimeoutError(name, timeout) from None
        raise
