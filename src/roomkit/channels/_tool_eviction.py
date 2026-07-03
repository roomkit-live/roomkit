"""Large tool result eviction and paginated re-reading."""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from typing import Any

from roomkit.providers.ai.base import AITool

logger = logging.getLogger("roomkit.channels.ai")

_MAX_EVICTED = 50


class ToolEviction:
    """Stores large tool results and provides paginated re-reading.

    When a tool result exceeds ``threshold_tokens``, the full result is
    stored in a FIFO-bounded buffer and replaced with a head/tail preview.
    The ``read_tool_result`` tool definition is injected into the AI
    context so the agent can paginate back through the full output.

    The store is scoped per room: the eviction buffer lives on a channel
    object shared by every room the channel serves, so an unscoped buffer
    would leak one conversation's tool output into another (and inject
    the re-read tool into rooms that evicted nothing). The room comes
    from the tool-loop context; paths outside a loop share one fallback
    scope.
    """

    def __init__(self, threshold_tokens: int = 5000) -> None:
        self.threshold_tokens = threshold_tokens
        self._store: OrderedDict[tuple[str, str], str] = OrderedDict()

    @staticmethod
    def _room_scope() -> str:
        from roomkit.channels.ai import _current_loop_ctx

        ctx = _current_loop_ctx.get()
        return (ctx.room_id if ctx is not None else None) or ""

    @property
    def has_evicted(self) -> bool:
        room = self._room_scope()
        return any(key[0] == room for key in self._store)

    def maybe_evict(self, result: str, tool_call_id: str = "") -> str:
        """Evict large results to the store, returning a preview."""
        estimated = len(result) // 4 + 1
        if estimated <= self.threshold_tokens:
            return result

        result_id = f"evicted_{tool_call_id}" if tool_call_id else f"evicted_{id(result)}"
        self._store[(self._room_scope(), result_id)] = result
        while len(self._store) > _MAX_EVICTED:
            self._store.popitem(last=False)

        lines = result.splitlines()
        head_n, tail_n = 5, 5
        if len(lines) <= head_n + tail_n:
            preview = result[:8000]
        else:
            head = "\n".join(lines[:head_n])
            tail = "\n".join(lines[-tail_n:])
            omitted = len(lines) - head_n - tail_n
            preview = f"{head}\n\n[... {omitted} lines omitted ...]\n\n{tail}"

        return (
            f"Result too large ({estimated} tokens). Full output saved as "
            f"'{result_id}'. Use read_stored_result to read it with pagination.\n\n"
            f"Preview:\n{preview}"
        )

    def handle_read(self, arguments: dict[str, Any]) -> str:
        """Paginate a previously evicted result.

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
            available = [rid for scope, rid in self._store if scope == room]
            return json.dumps({"error": f"Result '{result_id}' not found", "available": available})

        # Char budget per page: just under the eviction threshold so a page
        # can never be re-evicted on return. threshold_tokens is in TOKENS
        # (~4 chars each); the previous budget of threshold_tokens CHARS made
        # pages 4x smaller than allowed — a 50k-char result took ~11 paging
        # rounds, each re-sending the whole conversation context. 3 chars per
        # threshold token keeps ~25% headroom for JSON escaping + envelope.
        budget = max(1, self.threshold_tokens * 3)
        lines = self._paginable_lines(full_result, budget)
        total_lines = len(lines)

        page: list[str] = []
        used = 0
        for line in lines[offset : offset + limit]:
            if page and used + len(line) > budget:
                break
            page.append(line)
            used += len(line) + 1

        has_more = (offset + len(page)) < total_lines
        return json.dumps(
            {
                "content": "\n".join(page),
                "offset": offset,
                "lines_returned": len(page),
                "total_lines": total_lines,
                "has_more": has_more,
                "next_offset": offset + len(page) if has_more else None,
            }
        )

    @staticmethod
    def _paginable_lines(text: str, budget: int) -> list[str]:
        """Lines of ``text``, with lines longer than ``budget`` split into
        budget-sized chunks so every line fits within one page."""
        lines: list[str] = []
        for line in text.splitlines():
            if len(line) <= budget:
                lines.append(line)
            else:
                lines.extend(line[i : i + budget] for i in range(0, len(line), budget))
        return lines

    @staticmethod
    def tool_definition() -> AITool:
        """Return the AITool definition for read_stored_result."""
        return AITool(
            name="read_stored_result",
            description=(
                "Read a previously evicted large tool result. "
                "Supports line-based pagination via offset and limit; pages "
                "are size-bounded, so follow next_offset until has_more is "
                "false to read everything."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "result_id": {
                        "type": "string",
                        "description": "The evicted result ID shown in the preview.",
                    },
                    "offset": {
                        "type": "integer",
                        "default": 0,
                        "description": "Line number to start reading from.",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 200,
                        "description": "Maximum number of lines to return.",
                    },
                },
                "required": ["result_id"],
            },
        )
