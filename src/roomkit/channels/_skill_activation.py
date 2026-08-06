"""Per-room record of which skills the model activated — and where their bodies live.

A skill body reaching the model as a *tool result* only survives the turn that
fetched it: the rebuilt AI context drops tool events (``get_conversation`` returns
MESSAGE events only) and the tool-usage digest excludes skill infra tools. So from
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
skills recorded here.** Losing this store (process restart, channel object
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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomkit.channels._skill_constants import TOOL_ACTIVATE_SKILL

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

    def seed(self, room_id: str | None, calls: Any) -> None:
        """Seed the room from persisted tool-call history (oldest → newest).

        Reads the same ``TOOL_CALL_END`` rows the tool-usage digest hydrates from,
        keeping the ``activate_skill`` calls that SUCCEEDED. A call that errored
        (unknown or unavailable skill) never put rules in front of the model, so
        replaying it as an activation would inject a body the model never asked
        for. Anything whose result isn't a readable JSON object — an old evicted
        placeholder, a hook override — is skipped for the same reason: an
        activation we cannot confirm is one the model re-does once, which costs a
        body and never lies.
        """
        if not room_id:
            return
        room = self._by_room.setdefault(room_id, _RoomActivation())
        room.hydrated = True
        for call in calls:
            if call.get("name") != TOOL_ACTIVATE_SKILL:
                continue
            name = (call.get("arguments") or {}).get("name", "")
            if name and _succeeded(call.get("result")):
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
        for name in room.names:
            skill = skills.get_skill(name)
            body = skill.instructions.strip() if skill and skill.instructions else ""
            if body:
                sections.append(f"## Skill: {name}\n{body}")
        if not sections:
            return None
        return f"{_PROMPT_HEADING}\n\n" + "\n\n".join(sections)


def _succeeded(result: Any) -> bool:
    """Whether a persisted ``activate_skill`` result reads as a successful call."""
    if not isinstance(result, str):
        return False
    try:
        payload = json.loads(result)
    except (TypeError, ValueError):
        return False
    return isinstance(payload, dict) and "error" not in payload
