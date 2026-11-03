"""Per-conversation tool-usage memory — what the agent has already done.

The calls themselves ride the turn's history from the store (``_tool_history``,
RFC §6.4). What this in-memory, per-room record keeps is what the history does
not give, each bounded on its own axis:

* the set of distinct **tool names** it called — or that ``find_tools`` already
  revealed (``record_revealed``) — is re-revealed each turn (see
  ``_build_context``) so a tool used or found once stays callable while Tool
  Search hides the rest — bounded by recent distinct *tools*
  (``_REVEAL_MAX_TOOLS``): this is the part that costs full tool schemas, so
  it's bounded by the conversation's recent working set of tools, not by call
  count;
* the **images** the latest results returned (``render_images``) ride the
  turn's input, after its notes, for a model that sees: a replayed result
  says what an image showed in the words of its tool, never what it looks
  like, and a follow-up ("the colours of that picture") needs the picture.
  Bounded by the last images (``_IMAGES_REPLAYED``), each a thousand tokens and
  more on every turn, as the computer-use loop and other agent frameworks
  keep the last few. Only an inline image of a type and a size every vision
  provider takes is kept (``_replayable``), a live result's, never a seeded
  one's, and none once a turn fails on the provider's refusal: one refused
  would fail every turn that replays it;
* the room's kept **declaration** (``declaration``), where the provider holds
  tools unseen: the tools the room's turns show, so the tool block, the head
  of the cached prefix, stays the same from one turn to the next (RFC §6.4).
  Not rebuilt from history: a room that lost it takes its next turn's.

Scoped per room on a channel object shared by every room it serves — same shape
and lifetime as :class:`ToolEviction`. Kept in memory and rebuilt once per room
from the persisted ``TOOL_CALL_END`` events (:meth:`ToolUsageMemory.seed`).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from roomkit._media import VISION_IMAGE_TYPES, data_uri_media_type
from roomkit._text import identifier, quoted
from roomkit.channels._turn_notes import TURN_IMAGES_HEADER
from roomkit.providers.ai.base import AIImagePart, AITextPart

# Distinct tools re-revealed under Tool Search. Expensive (each re-exposed tool
# carries its full schema), so bounded by the conversation's recent working set
# of tools — large enough to cover a real multi-tool task, small enough not to
# undo Tool Search on a small window.
_REVEAL_MAX_TOOLS = 12
_MAX_ROOMS = 100  # FIFO cap across rooms a shared channel serves
_ARG_VALUE_CHARS = 48
# The latest images tool results returned, replayed with each later turn.
_IMAGES_REPLAYED = 2
# What an image replayed may be: inline, of a type and a size every vision
# provider takes. A link may expire or be out of the vendor's reach.
_REPLAYED_IMAGE_MAX_CHARS = 2_000_000


@dataclass
class _RoomMemory:
    # Distinct tool names in recency order (value unused) — feeds re-reveal
    # (bounded by _REVEAL_MAX_TOOLS): a tool used early then not since must
    # still stay callable.
    tools: OrderedDict[str, None] = field(default_factory=OrderedDict)
    # The latest images results returned, newest last, each with the call that
    # returned it (bounded by _IMAGES_REPLAYED).
    images: list[tuple[str, AIImagePart]] = field(default_factory=list)
    # Whether persisted history was already loaded (or attempted) for this room —
    # hydration is a one-shot per room per process, even when it finds nothing.
    hydrated: bool = False
    # The tools the room's turns show, where the provider holds the others
    # unseen (RFC §6.4); ``None`` until a turn declared them.
    declared: frozenset[str] | None = None


class ToolUsageMemory:
    """In-memory, room-scoped record of recent tool calls for a channel."""

    def __init__(
        self,
        reveal_max_tools: int = _REVEAL_MAX_TOOLS,
        recorded: Callable[[str], bool] | None = None,
    ) -> None:
        # Whether a call to a tool is work the agent did: the channel's
        # discovery and housekeeping tools are not, and are always available
        # anyway, so recording them would only pointlessly re-reveal tools that
        # are never hidden (their ``in_tool_memory`` trait). Everything is
        # recorded without a channel.
        self._recorded: Callable[[str], bool] = recorded or (lambda _name: True)
        self._reveal_max_tools = reveal_max_tools
        self._by_room: OrderedDict[str, _RoomMemory] = OrderedDict()

    def record(
        self,
        room_id: str | None,
        name: str,
        arguments: dict[str, Any],
        result: Any,
        *,
        keep_images: bool = True,
    ) -> None:
        """Record one completed tool call: its tool stays revealed, the images
        it returned replay. No-op for infra tools / missing room.

        Its images are kept to replay unless *keep_images* is false.
        """
        if not room_id or not self._recorded(name):
            return
        mem = self._by_room.setdefault(room_id, _RoomMemory())
        self._by_room.move_to_end(room_id)
        if keep_images and isinstance(result, list):
            head = _format_head(name, arguments)
            mem.images += [(head, p) for p in result if _replayable(p)]
            del mem.images[:-_IMAGES_REPLAYED]

        # Distinct-tool reveal set: mark this tool most-recent, evict the oldest.
        mem.tools.pop(name, None)
        mem.tools[name] = None
        while len(mem.tools) > self._reveal_max_tools:
            mem.tools.popitem(last=False)

        while len(self._by_room) > _MAX_ROOMS:
            self._by_room.popitem(last=False)

    def forget(self, room_id: str) -> None:
        """Drop *room_id*'s record; its next turn hydrates it again from the
        store (RFC §10.6.3)."""
        self._by_room.pop(room_id, None)

    def needs_hydration(self, room_id: str | None) -> bool:
        """Whether persisted history should be loaded for this room.

        True only while the room has no live entries and no prior hydration
        attempt — a room already carrying live calls must not be re-seeded
        with stale history, and an empty history must not be re-queried
        every turn.
        """
        if not room_id:
            return False
        mem = self._by_room.get(room_id)
        return mem is None or (not mem.hydrated and not mem.tools)

    def seed(self, room_id: str | None, calls: Any) -> None:
        """Seed the room from persisted history (oldest → newest).

        Marks the room hydrated even when ``calls`` is empty, so a room with
        no history is not re-queried on every turn. Entries flow through
        :meth:`record`, so infra filtering and bounds apply exactly as they do
        for live calls. Their images are not replayed: a
        store may hand back a result's parts as they were, its images too.
        """
        if not room_id:
            return
        mem = self._by_room.setdefault(room_id, _RoomMemory())
        mem.hydrated = True
        for call in calls:
            name = call.get("name", "")
            if not name:
                continue
            self.record(
                room_id,
                name,
                call.get("arguments") or {},
                call.get("result", ""),
                keep_images=False,
            )
        # A room with no rows records nothing, and counts toward the bound too.
        while len(self._by_room) > _MAX_ROOMS:
            self._by_room.popitem(last=False)

    def forget_images(self, room_id: str | None) -> None:
        """Replay none of the images the room's results returned so far: a
        turn failed on the provider's refusal, and an image it refused would
        fail every later turn that replays it."""
        mem = self._by_room.get(room_id) if room_id else None
        if mem is not None:
            mem.images.clear()

    def record_revealed(self, room_id: str | None, names: Any) -> None:
        """Mark tools revealed by ``find_tools`` as part of the room's working set.

        ``find_tools`` tells the model its matches are "invocable for the rest
        of the session" — honouring that requires the reveal to outlive the
        loop, not just the turn (a tool found in turn N is often only called
        in turn N+1, after the user confirms). Revealed-but-not-yet-called
        tools share the called-tools recency window (``_REVEAL_MAX_TOOLS``):
        a reveal burst can age older entries out, and a tool actually used
        re-enters on use.
        """
        if not room_id:
            return
        mem = self._by_room.setdefault(room_id, _RoomMemory())
        self._by_room.move_to_end(room_id)
        for name in names:
            if not name or not self._recorded(name):
                continue
            mem.tools.pop(name, None)
            mem.tools[name] = None
        while len(mem.tools) > self._reveal_max_tools:
            mem.tools.popitem(last=False)
        while len(self._by_room) > _MAX_ROOMS:
            self._by_room.popitem(last=False)

    def declaration(self, room_id: str | None) -> frozenset[str] | None:
        """The tools the room's turns show, kept from one turn to the next
        (RFC §6.4); ``None`` when no turn of this process declared them."""
        mem = self._by_room.get(room_id) if room_id else None
        return mem.declared if mem is not None else None

    def keep_declaration(self, room_id: str | None, names: frozenset[str]) -> None:
        """Keep *names* as the tools the room's next turns show."""
        if not room_id:
            return
        mem = self._by_room.setdefault(room_id, _RoomMemory())
        self._by_room.move_to_end(room_id)
        mem.declared = names
        while len(self._by_room) > _MAX_ROOMS:
            self._by_room.popitem(last=False)

    def tool_names(self, room_id: str | None) -> set[str]:
        """Distinct tools called or revealed in this room — re-revealed per turn."""
        if not room_id:
            return set()
        mem = self._by_room.get(room_id)
        return set(mem.tools) if mem else set()

    def render_images(self, room_id: str | None) -> list[AITextPart | AIImagePart]:
        """The latest images the room's tool results returned, each after the
        call that returned it, under a header framing them as data; empty when
        there are none (RFC §6.4)."""
        mem = self._by_room.get(room_id) if room_id else None
        if mem is None or not mem.images:
            return []
        parts: list[AITextPart | AIImagePart] = [AITextPart(text=TURN_IMAGES_HEADER)]
        for head, image in mem.images:
            parts += [AITextPart(text=f"{head} returned:"), image]
        return parts


def _format_head(name: str, arguments: dict[str, Any]) -> str:
    """The call as an image's label names it, ``name(key=“value”, n=3)``: the
    model wrote it, so the name and keys are identifiers and each text quoted
    (RFC §6.4)."""
    args = ", ".join(
        f"{identifier(key, 'arg')}={_arg_value(value)}" for key, value in arguments.items()
    )
    return f"{identifier(name, 'tool')}({args})"


def _arg_value(value: Any) -> str:
    """An argument's value in a label: a number, a flag or nothing as it is,
    any other value quoted."""
    if value is None or isinstance(value, bool | int | float):
        return repr(value)
    return quoted(value if isinstance(value, str) else repr(value), _ARG_VALUE_CHARS)


def _replayable(part: Any) -> bool:
    """Whether *part* is an image later turns may replay: inline (a data URI)
    of a type and a size every vision provider takes."""
    if not isinstance(part, AIImagePart) or len(part.url) > _REPLAYED_IMAGE_MAX_CHARS:
        return False
    return data_uri_media_type(part.url) in VISION_IMAGE_TYPES
