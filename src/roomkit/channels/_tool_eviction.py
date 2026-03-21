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
    """

    def __init__(self, threshold_tokens: int = 5000) -> None:
        self.threshold_tokens = threshold_tokens
        self._store: OrderedDict[str, str] = OrderedDict()

    @property
    def has_evicted(self) -> bool:
        return bool(self._store)

    def maybe_evict(self, result: str, tool_call_id: str = "") -> str:
        """Evict large results to the store, returning a preview."""
        estimated = len(result) // 4 + 1
        if estimated <= self.threshold_tokens:
            return result

        result_id = f"evicted_{tool_call_id}" if tool_call_id else f"evicted_{id(result)}"
        self._store[result_id] = result
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
            f"'{result_id}'. Use _read_tool_result to read it with pagination.\n\n"
            f"Preview:\n{preview}"
        )

    def handle_read(self, arguments: dict[str, Any]) -> str:
        """Paginate a previously evicted result."""
        result_id = arguments.get("result_id", "")
        offset = arguments.get("offset", 0)
        limit = arguments.get("limit", 200)

        full_result = self._store.get(result_id)
        if full_result is None:
            return json.dumps(
                {"error": f"Result '{result_id}' not found", "available": list(self._store.keys())}
            )

        lines = full_result.splitlines()
        total_lines = len(lines)
        page = lines[offset : offset + limit]
        return json.dumps(
            {
                "content": "\n".join(page),
                "offset": offset,
                "lines_returned": len(page),
                "total_lines": total_lines,
                "has_more": (offset + limit) < total_lines,
            }
        )

    @staticmethod
    def tool_definition() -> AITool:
        """Return the AITool definition for _read_tool_result."""
        return AITool(
            name="_read_tool_result",
            description=(
                "Read a previously evicted large tool result. "
                "Supports line-based pagination via offset and limit."
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
