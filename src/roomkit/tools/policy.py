"""Tool access policy with allow/deny glob patterns and role-based overrides."""

from __future__ import annotations

from collections.abc import Callable
from fnmatch import fnmatch
from typing import Literal

from pydantic import BaseModel, Field, PrivateAttr


class RoleOverride(BaseModel):
    """Per-role tool policy override.

    ``mode`` controls how the override combines with the base policy:

    - ``"restrict"`` (default): deny lists are unioned, allow lists are
      intersected (a tool must pass **both** the base and override allow lists).
    - ``"replace"``: the override completely replaces the base policy.
    """

    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)
    mode: Literal["restrict", "replace"] = "restrict"


class ToolPolicy(BaseModel):
    """Per-agent allow/deny rules for tool access.

    Rules use :func:`fnmatch.fnmatch` glob patterns (e.g. ``"mcp_*"``,
    ``"search_*"``).

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

    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)
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
        for pattern in self.deny:
            if fnmatch(tool_name, pattern):
                return False

        # Non-empty allow list acts as whitelist
        if self.allow and not any(fnmatch(tool_name, pattern) for pattern in self.allow):
            return False

        # Restrict-mode secondary allow constraint
        if not self._restrict_allow:
            return True
        return any(fnmatch(tool_name, pattern) for pattern in self._restrict_allow)

    def as_filter(self) -> Callable[[str], bool]:
        """Return a callable ``(tool_name) -> bool`` suitable for :func:`filter`."""
        return self.is_allowed
