"""Normalized models for Teams provider data.

Consumers should depend on these types rather than the raw SDK shapes —
keeping ``botbuilder`` an implementation detail of the provider.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TeamsMember:
    """A Teams chat / channel member.

    Fields mirror what the Bot Framework roster + Teams extensions return,
    normalized to plain strings so callers don't have to parse different
    field name variants (``email`` vs ``user_principal_name`` vs
    ``givenName``).
    """

    teams_user_id: str
    """Bot-framework user identifier (``"29:..."`` for AAD users)."""

    aad_object_id: str | None
    """Azure AD object ID. ``None`` for anonymous meeting joiners."""

    name: str
    """Display name as Teams renders it."""

    email: str | None
    """Email address reported by the roster, when available."""

    user_principal_name: str | None
    """AAD UPN. Often equals ``email`` for internal users; differs for guests."""

    role: str | None
    """Member role inside the team (``"owner"`` / ``"member"`` / ``"guest"``), if known."""
