"""Per-conversation tool-usage memory — what the agent has already done.

The store filters tool-call events out of the rebuilt AI context
(``get_conversation`` returns MESSAGE events only — providers track tool context
*within* a turn, not across turns). So from one turn to the next the model loses
all trace of the tools it invoked: it can't tell which tool or source it used,
and — under Tool Search — it can't re-call a tool it already used because the
catalogue is re-hidden every turn. This in-memory, per-room record closes both
gaps, which have DIFFERENT shapes and costs, so each is bounded on its own axis:

* a compact **digest** (tool name + arguments + a short result preview) is added
  to the system prompt so the model knows what it did — bounded by recent
  *calls* (``_DIGEST_MAX_CALLS``): a short, readable "what you did" block;
* the set of distinct **tool names** it called is re-revealed each turn (see
  ``_build_context``) so a tool used once stays callable while Tool Search hides
  the rest — bounded by recent distinct *tools* (``_REVEAL_MAX_TOOLS``): this is
  the part that costs full tool schemas, so it's bounded by the conversation's
  recent working set of tools, not by call count.

Scoped per room on a channel object shared by every room it serves — same shape
and lifetime as :class:`ToolEviction`. In-memory only: a process restart clears
it (the model simply rediscovers tools on next use), which is fine for
continuity within a live conversation. A durable, store-backed variant can come
later if cross-restart memory is needed.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from roomkit.channels._skill_constants import SKILL_INFRA_TOOL_NAMES
from roomkit.channels._tool_search_constants import TOOL_SEARCH_INFRA_TOOL_NAMES

# Discovery / housekeeping tools are not "work the agent did" and are always
# available anyway — recording them would only add noise to the digest and
# pointlessly re-reveal tools that are never hidden.
_INFRA_NAMES = (
    TOOL_SEARCH_INFRA_TOOL_NAMES | SKILL_INFRA_TOOL_NAMES | frozenset({"read_stored_result"})
)

# Recent calls shown in the digest. Cheap (~one short line each); the bound is
# readability — a "what you did" block longer than this is noise, not memory.
_DIGEST_MAX_CALLS = 8
# Distinct tools re-revealed under Tool Search. Expensive (each re-exposed tool
# carries its full schema), so bounded by the conversation's recent working set
# of tools — large enough to cover a real multi-tool task, small enough not to
# undo Tool Search on a small window.
_REVEAL_MAX_TOOLS = 12
_MAX_ROOMS = 100  # FIFO cap across rooms a shared channel serves
_RESULT_PREVIEW_CHARS = 120
_ARG_VALUE_CHARS = 48


@dataclass
class _Call:
    name: str
    arguments: dict[str, Any]
    result_preview: str


@dataclass
class _RoomMemory:
    # Recent calls, newest last — feeds the digest (bounded by _DIGEST_MAX_CALLS).
    calls: list[_Call] = field(default_factory=list)
    # Distinct tool names in recency order (value unused) — feeds re-reveal
    # (bounded by _REVEAL_MAX_TOOLS). Separate from ``calls`` because a tool used
    # early then not since must still stay callable even if newer calls pushed it
    # out of the digest window.
    tools: OrderedDict[str, None] = field(default_factory=OrderedDict)


class ToolUsageMemory:
    """In-memory, room-scoped record of recent tool calls for a channel."""

    def __init__(
        self,
        digest_max_calls: int = _DIGEST_MAX_CALLS,
        reveal_max_tools: int = _REVEAL_MAX_TOOLS,
    ) -> None:
        self._digest_max_calls = digest_max_calls
        self._reveal_max_tools = reveal_max_tools
        self._by_room: OrderedDict[str, _RoomMemory] = OrderedDict()

    def record(
        self, room_id: str | None, name: str, arguments: dict[str, Any], result: Any
    ) -> None:
        """Record one completed tool call. No-op for infra tools / missing room."""
        if not room_id or name in _INFRA_NAMES:
            return
        mem = self._by_room.setdefault(room_id, _RoomMemory())
        self._by_room.move_to_end(room_id)

        entry = _Call(name, dict(arguments), self._preview(result))
        # Collapse an immediately-preceding identical call (same name + args) so a
        # repeated poll (e.g. a playback "get") doesn't crowd out the digest.
        if mem.calls and mem.calls[-1].name == name and mem.calls[-1].arguments == entry.arguments:
            mem.calls[-1] = entry
        else:
            mem.calls.append(entry)
        if len(mem.calls) > self._digest_max_calls:
            del mem.calls[: len(mem.calls) - self._digest_max_calls]

        # Distinct-tool reveal set: mark this tool most-recent, evict the oldest.
        mem.tools.pop(name, None)
        mem.tools[name] = None
        while len(mem.tools) > self._reveal_max_tools:
            mem.tools.popitem(last=False)

        while len(self._by_room) > _MAX_ROOMS:
            self._by_room.popitem(last=False)

    def tool_names(self, room_id: str | None) -> set[str]:
        """Distinct tools called in this room — used to re-reveal them per turn."""
        if not room_id:
            return set()
        mem = self._by_room.get(room_id)
        return set(mem.tools) if mem else set()

    def render_digest(self, room_id: str | None) -> str | None:
        """Markdown block listing recent calls, or ``None`` when there are none."""
        if not room_id:
            return None
        mem = self._by_room.get(room_id)
        if mem is None or not mem.calls:
            return None
        lines = [
            "## Tools you've already used here",
            "Tools you've ALREADY CALLED this conversation — reuse them directly, "
            "don't re-search for them. This is NOT your full toolset, only what "
            "you happened to use: many more tools stay hidden behind find_tools, so "
            "never conclude you can't do something without searching for it first.",
        ]
        lines.extend(f"- {self._format_call(c)}" for c in mem.calls)
        return "\n".join(lines)

    @classmethod
    def _format_call(cls, call: _Call) -> str:
        return f"{call.name}({cls._format_args(call.arguments)}) → {call.result_preview}"

    @staticmethod
    def _format_args(arguments: dict[str, Any]) -> str:
        parts: list[str] = []
        for key, value in arguments.items():
            rendered = repr(value)
            if len(rendered) > _ARG_VALUE_CHARS:
                rendered = rendered[:_ARG_VALUE_CHARS] + "…"
            parts.append(f"{key}={rendered}")
        return ", ".join(parts)

    @staticmethod
    def _preview(result: Any) -> str:
        text = " ".join(str(result).split())
        if len(text) > _RESULT_PREVIEW_CHARS:
            return text[:_RESULT_PREVIEW_CHARS] + "…"
        return text or "(no result)"
