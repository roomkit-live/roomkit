"""Large tool result eviction and paginated re-reading."""

from __future__ import annotations

import json
import logging
from collections import Counter, OrderedDict
from collections.abc import Sequence
from typing import Any

from roomkit.channels._skill_constants import TOOL_ACTIVATE_SKILL
from roomkit.channels._stored_read import REREAD_DESCRIPTION, REREAD_PARAMETERS, page, search
from roomkit.core.exceptions import ChannelRefusalError
from roomkit.memory.token_estimator import estimate_tokens
from roomkit.providers.ai.base import AIImagePart, AITextPart, AITool
from roomkit.tools.context import _current_loop_ctx

logger = logging.getLogger("roomkit.channels.ai")

# Stored results a room keeps (its least recently read go first), and the
# bounds the whole store needs for memory, which take from the room holding
# the most first: a room holding few results keeps them while others evict
# (RFC §21.5).
_MAX_EVICTED = 50
_MAX_EVICTED_TOTAL = 200
_MAX_EVICTED_CHARS = 64 * 1024 * 1024

# Ids the store has handed out, remembered well past what any context can
# still name, so a released id is not given to another result while it is
# among the last this many issued.
_ISSUED_IDS_REMEMBERED = 10_000

# Chars reserved per read_stored_result page for the JSON envelope — fixed
# keys plus the partial-page warning prose (~350 chars escaped, worst case) —
# and a margin, so worst-case escaping of the content still leaves the page
# under the re-eviction bound (4 * threshold_tokens chars). See handle_read.
_PAGE_ENVELOPE_CHARS = 512

# Ceiling of the preview that stands in for an evicted result, in chars. The
# budget also shrinks with the threshold (see _preview_budget), so a preview
# never carries as much as the result it replaces.
_PREVIEW_MAX_CHARS = 8000
_PREVIEW_HEAD_LINES = 5
_PREVIEW_TAIL_LINES = 5

# How every eviction placeholder starts. The usage memory recognises one by it:
# a placeholder is what TOOL_CALL_END persists for an evicted result, and its
# ``evicted_…`` id dies with the process, so it must not be replayed as data.
EVICTION_PLACEHOLDER_PREFIX = "Result too large ("

# The tool that pages a stored result back.
REREAD_TOOL = "read_stored_result"


def kept_whole(tool_name: str) -> bool:
    """Whether the model reads *tool_name*'s result whole, never as a preview.

    A skill's instructions are binding rules the model must hold whole
    (RFC §24.4), never a head and tail behind a ``read_stored_result``
    pointer. Every other result is data, and paginating data is what
    eviction is for.
    """
    return tool_name == TOOL_ACTIVATE_SKILL


def is_eviction_placeholder(text: str) -> bool:
    """Whether *text* is the stand-in :class:`ToolEviction` gave an oversized result."""
    return text.startswith(EVICTION_PLACEHOLDER_PREFIX)


def eviction_placeholder_size(text: str) -> str:
    """The size head of the placeholder in *text*, ``Result too large (N tokens)``,
    without the stored id, which does not outlive the process."""
    start = text.find(EVICTION_PLACEHOLDER_PREFIX)
    return text[start:].split(")", 1)[0] + ")"


def _omission_marker(count: int) -> str:
    return f"[... {count} lines omitted ...]"


def _truncation_marker(count: int) -> str:
    return f" [... {count} chars truncated ...]"


def _clip_line(line: str, room: int) -> str | None:
    """*line* cut to fit *room* chars, the cut stated in a marker; ``None``
    when the marker leaves no room for any of the line."""
    # Sized for the widest count, so the real marker is never longer.
    keep = room - len(_truncation_marker(len(line)))
    if keep < 1:
        return None
    return line[:keep] + _truncation_marker(len(line) - keep)


def _fit_lines(lines: list[str], budget: int) -> list[str]:
    """The leading *lines* that fit in *budget* chars once joined by newlines.

    The first line that does not fit is clipped, if its marker leaves room for
    some of it, and ends the run: the lines after it are not shown.
    """
    kept: list[str] = []
    used = 0
    for line in lines:
        sep = 1 if kept else 0
        room = budget - used - sep
        if len(line) > room:
            clipped = _clip_line(line, room)
            if clipped is not None:
                kept.append(clipped)
            break
        kept.append(line)
        used += sep + len(line)
    return kept


def _preview(result: str, budget: int) -> str:
    """Head/tail preview of *result* within *budget* chars.

    A bound on lines alone would let one giant line (minified HTML, a JSON
    blob) reach the provider whole, which is what eviction exists to prevent.
    The head and the tail split the lines, so even a short result shows its
    last line. The head takes at most half the budget, or what the whole
    tail leaves; the tail takes the rest, filled from the last line. A line
    that does not fit is clipped with a marker, and every line not shown is
    counted in the omission marker.
    """
    lines = result.splitlines()
    # Reserved at its widest, which also covers the newline joining head and
    # tail when nothing is omitted.
    reserve = len(_omission_marker(len(lines))) + 4
    if budget < reserve:
        return ""
    head_n = min(_PREVIEW_HEAD_LINES, (len(lines) + 1) // 2)
    head_src = lines[:head_n]
    tail_src = lines[max(head_n, len(lines) - _PREVIEW_TAIL_LINES) :]
    lines_budget = budget - reserve
    whole_tail = len("\n".join(tail_src))
    head = _fit_lines(head_src, max(lines_budget // 2, lines_budget - whole_tail))
    head_text = "\n".join(head)
    tail = _fit_lines(tail_src[::-1], lines_budget - len(head_text))[::-1]
    tail_text = "\n".join(tail)

    omitted = len(lines) - len(head) - len(tail)
    if not omitted:
        return "\n".join(head + tail)
    return "\n\n".join(part for part in (head_text, _omission_marker(omitted), tail_text) if part)


class ToolEviction:
    """Stores large tool results and provides paginated re-reading.

    When a tool result exceeds ``threshold_tokens``, the full result is
    stored, each room keeping its most recently stored or read results
    within the store's bounds, and replaced with a head/tail preview
    bounded in lines and in chars.
    The ``read_stored_result`` tool (``with_reread_tool``) lets the agent
    paginate back through the full output.

    The store is scoped per room: the eviction buffer lives on a channel
    object shared by every room the channel serves, so an unscoped buffer
    would leak one conversation's tool output into another. The room comes
    from the tool-loop context; paths outside a loop share one fallback
    scope.
    """

    def __init__(self, threshold_tokens: int = 5000) -> None:
        self.threshold_tokens = threshold_tokens
        self._store: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._chars = 0
        self._issued: OrderedDict[tuple[str, str], None] = OrderedDict()

    @staticmethod
    def _room_scope() -> str:
        ctx = _current_loop_ctx.get()
        return (ctx.room_id if ctx is not None else None) or ""

    @property
    def has_evicted(self) -> bool:
        room = self._room_scope()
        return any(key[0] == room for key in self._store)

    def forget(self, room_id: str) -> None:
        """Let go of the results *room_id* stored, as the channel lets the
        room go (RFC §10.6.2): a room joined again under that id reads back
        nothing of the earlier conversation. Its issued ids stay taken, so a
        placeholder left in its history never reads another result."""
        for key in [key for key in self._store if key[0] == room_id]:
            self._drop(key)

    def holds(self, placeholder: str) -> bool:
        """Whether the room still stores the result *placeholder* points to:
        a later turn may then read it back as the turn that got it could."""
        _, saved, rest = placeholder.partition(" saved as '")
        result_id = rest.split("'", 1)[0]
        return bool(saved and result_id) and (self._room_scope(), result_id) in self._store

    def maybe_evict(self, result: str, tool_call_id: str = "") -> str:
        """Evict large results to the store, returning a preview.

        A placeholder is returned as it is: evicted again (a hook handing the
        bounded copy back, at a threshold below the placeholder's own size) it
        would overwrite the text it points to.
        """
        if estimate_tokens(result) <= self.threshold_tokens or is_eviction_placeholder(result):
            return result
        return self.evict(result, tool_call_id)

    def evict(self, result: str, tool_call_id: str = "", preview_chars: int | None = None) -> str:
        """Store *result* for re-reading and return the placeholder that
        previews it in *preview_chars*, whatever its size."""
        room = self._room_scope()
        result_id = self._free_id(
            room, f"evicted_{tool_call_id}" if tool_call_id else f"evicted_{id(result)}"
        )
        self._store[(room, result_id)] = result
        self._chars += len(result)
        self._bound(room)

        preview = _preview(result, preview_chars or self._preview_budget())
        return (
            f"{EVICTION_PLACEHOLDER_PREFIX}{estimate_tokens(result)} tokens). Full output "
            f"saved as '{result_id}'. Use {REREAD_TOOL} to read it with pagination.\n\n"
            f"Preview:\n{preview}"
        )

    def _free_id(self, room: str, base: str) -> str:
        """*base*, numbered when the room was already given it.

        An id is not given again while it is among the last
        ``_ISSUED_IDS_REMEMBERED`` issued, even once its result left the
        store: a placeholder still in the model's context would otherwise
        read another call's data. A call id reused in a later turn gets
        ``base_2``.
        """
        result_id, n = base, 1
        while (room, result_id) in self._issued or (room, result_id) in self._store:
            n += 1
            result_id = f"{base}_{n}"
        self._issued[(room, result_id)] = None
        while len(self._issued) > _ISSUED_IDS_REMEMBERED:
            self._issued.popitem(last=False)
        return result_id

    def _bound(self, room: str) -> None:
        """Keep :data:`_MAX_EVICTED` per room, then the store's bounds.

        Over a store bound, the room holding the most gives up its least
        recently read result first, so a room holding few keeps them. The
        newest result always stays, however large.
        """
        in_room = [key for key in self._store if key[0] == room]
        for key in in_room[: max(0, len(in_room) - _MAX_EVICTED)]:
            self._drop(key)
        while len(self._store) > 1 and (
            len(self._store) > _MAX_EVICTED_TOTAL or self._chars > _MAX_EVICTED_CHARS
        ):
            self._drop(self._oldest_of_fullest_room())

    def _oldest_of_fullest_room(self) -> tuple[str, str]:
        counts = Counter(scope for scope, _ in self._store)
        fullest = max(counts, key=counts.__getitem__)
        return next(key for key in self._store if key[0] == fullest)

    def _drop(self, key: tuple[str, str]) -> None:
        self._chars -= len(self._store.pop(key))

    def maybe_evict_parts(
        self, parts: list[AITextPart | AIImagePart], tool_call_id: str = ""
    ) -> list[AITextPart | AIImagePart]:
        """Evict the text of a content-part result, keeping its images in place.

        The text parts are measured joined, so many medium parts cannot add up
        past the threshold unseen. Over it, they are stored as one text and
        replaced by a single placeholder part where the first text part was.
        """
        texts = [p.text for p in parts if isinstance(p, AITextPart)]
        if estimate_tokens("\n".join(texts)) <= self.threshold_tokens:
            return parts
        if any(is_eviction_placeholder(t) for t in texts):
            return parts
        return self.evict_parts(parts, tool_call_id)

    def evict_parts(
        self,
        parts: list[AITextPart | AIImagePart],
        tool_call_id: str = "",
        preview_chars: int | None = None,
    ) -> list[AITextPart | AIImagePart]:
        """*parts* with their text stored as one result and replaced by its
        placeholder where the first text part was, their images kept."""
        texts = [p.text for p in parts if isinstance(p, AITextPart)]
        if not texts:
            return parts
        text = "\n".join(texts)
        placeholder: AITextPart | None = AITextPart(
            text=self.evict(text, tool_call_id, preview_chars)
        )
        kept: list[AITextPart | AIImagePart] = []
        for part in parts:
            if not isinstance(part, AITextPart):
                kept.append(part)
            elif placeholder is not None:
                kept.append(placeholder)
                placeholder = None
        return kept

    def _preview_budget(self) -> int:
        """Chars the preview may use: the ceiling, or half of what evicts (the
        estimator is len // 4 tokens) when the threshold is small."""
        return min(_PREVIEW_MAX_CHARS, 2 * self.threshold_tokens)

    def handle_read(self, arguments: dict[str, Any]) -> str:
        """Paginate a previously evicted result, or search it for ``query``.

        Pages are size-bounded below the eviction threshold: a page that grew
        past it would itself be evicted on return, re-stored under a new id,
        and the agent would chase evicted results forever. Lines longer than
        the page budget (single-line JSON tool results) are split into chunks
        so they paginate instead of coming back whole.
        """
        result_id = arguments.get("result_id", "")
        offset = arguments.get("offset", 0)
        limit = arguments.get("limit", 800)

        room = self._room_scope()
        full_result = self._store.get((room, result_id))
        if full_result is None:
            # A refusal, not a result: the call failed, and the room's tool
            # memory must not keep it as the answer to this read.
            available = [rid for scope, rid in self._store if scope == room]
            raise ChannelRefusalError(
                json.dumps({"error": f"Result '{result_id}' not found", "available": available})
            )
        # Read back, it is in use: the room's least recently read go first.
        self._store.move_to_end((room, result_id))
        # A blank query, an optional field a model filled in, reads a page.
        query = str(arguments.get("query") or "")
        if query.strip():
            return search(full_result, query, offset, self._page_budget())
        return page(full_result, offset, limit, self._page_budget())

    def _page_budget(self) -> int:
        """The chars a page (or a search's matches) may carry.

        The page returns as a JSON string (the content
        re-escaped, wrapped in an envelope) and is re-measured against
        threshold_tokens on the way back, so it must never itself be evicted
        — else the agent chases evicted results forever. The estimator is
        len // 4 tokens, so the page must stay under 4 * threshold_tokens
        chars. JSON escaping can double the content (already-escaped tool
        output re-escapes worst-case ~2x) and the envelope adds a fixed
        overhead, so bound the raw content at 2 * threshold_tokens minus an
        envelope allowance: worst case 2*budget + envelope stays under
        4 * threshold_tokens.
        """
        return max(1, self.threshold_tokens * 2 - _PAGE_ENVELOPE_CHARS)

    def with_reread_tool(self, tools: Sequence[AITool]) -> list[AITool]:
        """*tools* with the re-read tool last, when they declare a tool or the
        room holds a stored result.

        Declared from the first round of such a turn, so the declaration does
        not change when a result is stored (RFC §6.4): a declaration that gains
        a tool invalidates everything a provider cached after the tools.
        """
        if not tools and not self.has_evicted:
            return []
        reread = [t for t in tools if t.name == REREAD_TOOL] or [self.tool_definition()]
        return [*(t for t in tools if t.name != REREAD_TOOL), reread[0]]

    @staticmethod
    def tool_definition() -> AITool:
        """Return the AITool definition for read_stored_result."""
        return AITool(
            name=REREAD_TOOL,
            description=REREAD_DESCRIPTION,
            parameters=REREAD_PARAMETERS,
        )
