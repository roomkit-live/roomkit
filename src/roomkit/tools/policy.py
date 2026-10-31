"""Tool access policy with allow/deny glob patterns and role-based overrides."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field, PrivateAttr


def matches_any_pattern(tool_name: str, patterns: Iterable[str]) -> bool:
    """Whether *tool_name* matches any of *patterns* (RFC §24.2 glob semantics).

    The single definition of what a tool pattern covers. Skill gating, skill
    ``allowed_tools`` and :class:`ToolPolicy` all decide which tools a caller
    may reach, so they have to agree on what ``search_*`` means — three
    independent copies of one matching loop is three chances to drift.

    A pattern's ``*`` stands for any characters, ``?`` for one, and every
    other character for itself (RFC §21.1): a bracket is no character class.
    """
    return any(_glob(pattern).fullmatch(tool_name) for pattern in patterns)


@lru_cache(maxsize=1024)
def _glob(pattern: str) -> re.Pattern[str]:
    """*pattern* as the expression :func:`matches_any_pattern` reads it."""
    parts = (".*" if c == "*" else "." if c == "?" else re.escape(c) for c in pattern)
    return re.compile("".join(parts), re.DOTALL)


def bracket_error(pattern: str) -> str | None:
    """Why *pattern*, a tool pattern, is refused: it holds a bracket, which
    a pattern reads as itself (RFC §21.1). Refused rather than read so, since
    a rule written as a character class (``rm_[a-z]*``) would cover none of
    the names it was written for, and a deny rule or a gate would open."""
    if "[" in pattern or "]" in pattern:
        return f"tool pattern {pattern!r} holds a bracket: a pattern reads * and ? alone"
    return None


def _without_brackets(pattern: str) -> str:
    if (error := bracket_error(pattern)) is not None:
        raise ValueError(error)
    return pattern


ToolPattern = Annotated[str, AfterValidator(_without_brackets)]
"""A tool name or pattern: ``*`` stands for any characters, ``?`` for one,
a bracket is refused (:func:`bracket_error`)."""


class RoleOverride(BaseModel):
    """Per-role tool policy override.

    ``mode`` controls how the override combines with the base policy:

    - ``"restrict"`` (default): deny lists are unioned, allow lists are
      intersected (a tool must pass **both** the base and override allow lists).
    - ``"replace"``: the override completely replaces the base policy.
    """

    allow: list[ToolPattern] = Field(default_factory=list)
    deny: list[ToolPattern] = Field(default_factory=list)
    mode: Literal["restrict", "replace"] = "restrict"


def alias_server(name: str) -> str | None:
    """The server an MCP alias ``mcp__<server>__<tool>`` names, ``None`` for a
    name that is not an alias: the label a labelled ``MCPToolProvider``
    serves aliases under (RFC §21.2)."""
    if name.startswith("mcp__") and "__" in name[5:]:
        return name.split("__", 2)[1]
    return None


def served_tool_name(name: str) -> str:
    """The tool a call to *name* runs: an MCP alias ``mcp__<server>__<tool>``
    runs ``<tool>`` (``MCPToolProvider.as_tool_handler``); any other name runs
    as itself."""
    if alias_server(name) is None:
        return name
    return name.split("__", 2)[2]


def judged_names(name: str) -> tuple[str, ...]:
    """Every name a gate judges for a call to *name*: the name called, and the
    tool an MCP alias runs, so a policy or a skill's gate cannot be passed by
    calling a tool under its alias (RFC §21.1)."""
    served = served_tool_name(name)
    return (name,) if served == name else (name, served)


def call_admitted(policy: ToolPolicy, name: str) -> bool:
    """Whether *policy* admits a call to *name* under every name it is judged
    by (:func:`judged_names`): the one reading for a channel's gate, an
    external handler's and a host's (RFC §21.1)."""
    return all(policy.is_allowed(judged) for judged in judged_names(name))


def call_gated(name: str, patterns: Iterable[str]) -> bool:
    """Whether a glob of *patterns* (a skill's gate, RFC §24.2) covers a call
    to *name* under any name it is judged by (:func:`judged_names`)."""
    return any(matches_any_pattern(judged, patterns) for judged in judged_names(name))


def policy_refusal(name: str) -> str:
    """What the model reads of a call the tool policy refused, whichever
    gate refused it (an AI channel, a realtime session, a conference, an
    external tool handler)."""
    return f"Tool '{name}' is not permitted by the agent's tool policy."


class ToolPolicy(BaseModel):
    """Per-agent allow/deny rules for tool access.

    Rules are glob patterns (e.g. ``"mcp_*"``, ``"search_*"``): ``*`` stands
    for any characters, ``?`` for one (RFC §21.1).

    Resolution order:

    1. Empty *allow* **and** empty *deny* → **permit all** (backward compatible).
    2. If the tool name matches **any** *deny* pattern → **blocked**.
    3. If *allow* is non-empty and the tool name matches **no** *allow* pattern
       → **blocked**.
    4. Otherwise → **permitted**.

    In short: **deny always wins**, and a non-empty *allow* list is a whitelist.

    Role overrides
    ~~~~~~~~~~~~~~

    ``role_overrides`` maps :class:`~roomkit.models.enums.ParticipantRole` string
    values (e.g. ``"observer"``, ``"member"``) to :class:`RoleOverride` instances.

    Call :meth:`resolve` with a role to obtain an effective ``ToolPolicy`` that
    merges the base rules with the role-specific override.
    """

    allow: list[ToolPattern] = Field(default_factory=list)
    deny: list[ToolPattern] = Field(default_factory=list)
    role_overrides: dict[str, RoleOverride] = Field(default_factory=dict)

    # Private: when set, is_allowed enforces dual-constraint (restrict mode)
    _restrict_allow: list[str] = PrivateAttr(default_factory=list)

    def resolve(self, role: str | None = None) -> ToolPolicy:
        """Return an effective ``ToolPolicy`` for the given *role*.

        If *role* is ``None`` or has no override entry, returns ``self``
        unchanged (backward compatible).
        """
        if not role or role not in self.role_overrides:
            return self

        override = self.role_overrides[role]

        if override.mode == "replace":
            return ToolPolicy(allow=list(override.allow), deny=list(override.deny))

        # restrict mode: union deny, dual-constraint allow
        merged_deny = list(self.deny) + [p for p in override.deny if p not in self.deny]

        # For allow intersection: we store the override allow as a secondary
        # constraint via _restrict_allow.  is_allowed() checks both.
        resolved = ToolPolicy(allow=list(self.allow), deny=merged_deny)
        if override.allow:
            resolved._restrict_allow = list(override.allow)
        return resolved

    def is_allowed(self, tool_name: str) -> bool:
        """Return ``True`` if *tool_name* passes the policy."""
        if not self.allow and not self.deny and not self._restrict_allow:
            return True

        # Deny always wins
        if matches_any_pattern(tool_name, self.deny):
            return False

        # Non-empty allow list acts as whitelist
        if self.allow and not matches_any_pattern(tool_name, self.allow):
            return False

        # Restrict-mode secondary allow constraint
        if not self._restrict_allow:
            return True
        return matches_any_pattern(tool_name, self._restrict_allow)

    def as_filter(self) -> Callable[[str], bool]:
        """Return a callable ``(tool_name) -> bool`` suitable for :func:`filter`."""
        return self.is_allowed
