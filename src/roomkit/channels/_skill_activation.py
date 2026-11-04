"""Per-room record of which skills the model activated — and where their bodies live.

A skill body reaching the model as a *tool result* only survives the turn that
fetched it: the history replays no call of the channel's own housekeeping
tools, ``activate_skill`` among them (``_tool_history``). So from
one turn to the next the model loses both the instructions and any trace that it
ever activated the skill — and, obeying the skills preamble, it re-activates and
re-pays the whole body on every turn.

This room-scoped record closes that loop. An activation is remembered for the
conversation, and the body moves to where a per-turn rebuild can carry it: the
system prompt (see ``_ai_context``). ``activate_skill`` then returns the body only
the first time — afterwards a short ACK, because the rules are already in the
model's context. That is the same architecture the realtime channel runs per
session (``_realtime_skills``); this is its text-channel counterpart.

The invariant that makes the ACK safe is structural: **an ACK is returned only
because the body is in the prompt, and the body is in the prompt for exactly the
skills recorded here**, but on a standalone turn, whose prompt carries none and
which always delivers the body (RFC §24.4). Losing this store (process restart, channel object
replaced) loses the prompt block too, so the next activation returns the body
again — the mechanism degrades to the old behaviour, it never leaves the model
holding an ACK with no rules.

Scoped per room on a channel object shared by every room it serves — same shape
and lifetime as :class:`ToolUsageMemory` and :class:`ToolEviction`, and hydrated
from the same persisted ``TOOL_CALL_END`` history so a mid-conversation channel
swap doesn't restart the model amnesic.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomkit._text import fence
from roomkit.channels._skill_constants import (
    SERVED_SKILLS_HEADING,
    SERVED_SKILLS_INTRO,
    TOOL_ACTIVATE_SKILL,
)
from roomkit.skills.mcp.skill import MCPSkill
from roomkit.tools._outcome import OutcomeKind

if TYPE_CHECKING:
    from roomkit.skills.registry import SkillRegistry

# Skills kept active per room. Each one costs its full body in every subsequent
# turn's system prompt, so the set is bounded by recency rather than left to grow:
# an agent that works through many skills keeps the ones it is actually using.
# Eviction is graceful — the evicted skill is simply not active any more, so the
# next ``activate_skill`` for it returns the body and re-records it.
_MAX_ACTIVE_SKILLS = 4
_MAX_ROOMS = 100  # FIFO cap across rooms a shared channel serves

_PROMPT_HEADING = "# Active skill instructions (binding rules)"


@dataclass
class _RoomActivation:
    # Active skill names in recency order (value unused). Names only: the bodies
    # are resolved from the registry at render time, so a skill edited on disk is
    # never served stale from here.
    names: OrderedDict[str, None] = field(default_factory=OrderedDict)
    # Whether persisted history was already loaded (or attempted) for this room —
    # hydration is a one-shot per room per process, even when it finds nothing.
    hydrated: bool = False


class SkillActivationMemory:
    """In-memory, room-scoped record of the skills active in a conversation."""

    def __init__(self, max_active: int = _MAX_ACTIVE_SKILLS) -> None:
        self._max_active = max_active
        self._by_room: OrderedDict[str, _RoomActivation] = OrderedDict()

    def activate(self, room_id: str | None, name: str) -> bool:
        """Record an activation. Returns True when the skill was NOT already active.

        A caller uses the return value to decide what the tool answers: ``True``
        means the model has no body in context yet (send it), ``False`` means the
        prompt already carries it (a short ACK is enough). Re-activating refreshes
        recency — a skill the model keeps reaching for outlives one it abandoned.
        """
        if not room_id or not name:
            return False
        room = self._by_room.setdefault(room_id, _RoomActivation())
        self._by_room.move_to_end(room_id)
        is_new = name not in room.names
        room.names.pop(name, None)
        room.names[name] = None
        while len(room.names) > self._max_active:
            room.names.popitem(last=False)
        while len(self._by_room) > _MAX_ROOMS:
            self._by_room.popitem(last=False)
        return is_new

    def deactivate(self, room_id: str | None, name: str) -> None:
        """Retire *name* in this room, as if it had never been activated: its
        next activation delivers the body again (RFC §24.4). A served skill
        whose file failed verification is retired so (RFC §24.7)."""
        room = self._by_room.get(room_id) if room_id else None
        if room is not None:
            room.names.pop(name, None)

    def is_active(self, room_id: str | None, name: str) -> bool:
        """Whether *name* is currently active in this room."""
        if not room_id:
            return False
        room = self._by_room.get(room_id)
        return room is not None and name in room.names

    def active_names(self, room_id: str | None) -> set[str]:
        """Names of the skills active in this room (order-insensitive view)."""
        if not room_id:
            return set()
        room = self._by_room.get(room_id)
        return set(room.names) if room else set()

    def forget(self, room_id: str) -> None:
        """Drop *room_id*'s activations; its next turn hydrates them again
        from the store (RFC §10.6.3)."""
        self._by_room.pop(room_id, None)

    def needs_hydration(self, room_id: str | None) -> bool:
        """Whether persisted history should be loaded for this room.

        True only while the room has no live activation and no prior hydration
        attempt — a room already carrying activations must not be re-seeded with
        stale history, and an empty history must not be re-queried every turn.
        """
        if not room_id:
            return False
        room = self._by_room.get(room_id)
        return room is None or (not room.hydrated and not room.names)

    def seed(
        self, room_id: str | None, calls: Any, *, skip: Callable[[str], bool] | None = None
    ) -> None:
        """Seed the room from persisted tool-call history (oldest → newest).

        Reads the same ``TOOL_CALL_END`` rows the tool-usage memory hydrates from,
        keeping the ``activate_skill`` calls that SUCCEEDED: served, as the row's
        outcome states (RFC §6.4), and answered with the skill. A call a hook
        withheld, or one that errored (unknown or unavailable skill), never put
        rules in front of the model, so replaying it as an activation would
        inject a body the model never asked for. Anything whose result isn't a
        readable JSON object — an old evicted placeholder, a hook override — is
        skipped for the same reason: an activation we cannot confirm is one the
        model re-does once, which costs a body and never lies. A name *skip*
        holds is never replayed: a skill an MCP server serves, whose body is
        no longer held, is activated again (RFC §24.7).
        """
        if not room_id:
            return
        room = self._by_room.setdefault(room_id, _RoomActivation())
        room.hydrated = True
        for call in calls:
            if call.get("name") != TOOL_ACTIVATE_SKILL:
                continue
            name = (call.get("arguments") or {}).get("name", "")
            if name and _succeeded(call) and not (skip is not None and skip(name)):
                self.activate(room_id, name)

    def render_prompt(self, room_id: str | None, skills: SkillRegistry) -> str | None:
        """Bodies of this room's active skills, as a system-prompt block.

        Returns ``None`` when nothing is active (or nothing resolves), so the
        caller appends nothing at all. Skills the registry no longer serves are
        skipped rather than rendered empty — they simply stop being binding.
        """
        if not room_id:
            return None
        room = self._by_room.get(room_id)
        if room is None or not room.names:
            return None
        sections: list[str] = []
        served: list[str] = []
        for name in list(room.names):
            skill = skills.get_skill(name)
            if skill is None:
                # No longer served (a served skill retired or dropped): its
                # body is not in this prompt, so it is not active either, and
                # its next activation delivers the body (RFC §24.4).
                room.names.pop(name, None)
                continue
            body = skill.instructions.strip() if skill.instructions else ""
            if not body:
                continue
            if isinstance(skill, MCPSkill):
                served.append(served_skill_section(skill, body))
            else:
                sections.append(f"## Skill: {name}\n{body}")
        blocks = []
        if sections:
            blocks.append(f"{_PROMPT_HEADING}\n\n" + "\n\n".join(sections))
        if served:
            blocks.append(served_skills_block(served))
        return "\n\n".join(blocks) or None


def served_skills_block(sections: list[str]) -> str:
    """Served skills' sections under the heading that says whose they are."""
    return f"{SERVED_SKILLS_HEADING}\n\n{SERVED_SKILLS_INTRO}\n\n" + "\n\n".join(sections)


def served_skill_section(skill: MCPSkill, body: str) -> str:
    """A served skill's content as the prompt carries it: data in a ``<skill>``
    block it cannot close, under a line naming its origin (RFC §6.4, §24.7).
    The name and the URI keep to a name's characters, which the registry
    checked when it named the skill."""
    return (
        f"## Skill: {skill.name} (served by MCP server {skill.origin}, {skill.held.uri})\n"
        + fence("skill", body)
    )


def _succeeded(call: Mapping[str, Any]) -> bool:
    """Whether a persisted ``activate_skill`` call reads as a successful one.

    A row stored before outcomes were recorded carries none: its body alone
    tells.
    """
    if call.get("outcome") not in (None, OutcomeKind.SERVED):
        return False
    result = call.get("result")
    if not isinstance(result, str):
        return False
    try:
        payload = json.loads(result)
    except (TypeError, ValueError):
        return False
    return isinstance(payload, dict) and "error" not in payload
