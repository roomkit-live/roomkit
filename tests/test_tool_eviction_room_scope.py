"""ToolEviction per-room scoping.

The eviction buffer lives on a channel object shared by every room the
channel serves — without room scoping, one conversation's oversized tool
output is readable from another conversation, and the re-read tool is
injected into rooms that evicted nothing.
"""

from __future__ import annotations

import json

from roomkit.channels._tool_eviction import ToolEviction
from roomkit.channels.ai import _current_loop_ctx, _ToolLoopContext

_BIG = "line\n" * 20_000  # far past the default 5000-token threshold


def _in_room(room_id: str):
    return _current_loop_ctx.set(_ToolLoopContext(room_id=room_id))


class TestRoomScope:
    def test_read_is_scoped_to_the_evicting_room(self) -> None:
        ev = ToolEviction()

        token = _in_room("room-a")
        try:
            ev.maybe_evict(_BIG, "tc1")
        finally:
            _current_loop_ctx.reset(token)

        token = _in_room("room-b")
        try:
            out = json.loads(ev.handle_read({"result_id": "evicted_tc1"}))
        finally:
            _current_loop_ctx.reset(token)

        assert "error" in out
        # The other room's ids must not leak through the error hint either.
        assert out["available"] == []

        token = _in_room("room-a")
        try:
            out = json.loads(ev.handle_read({"result_id": "evicted_tc1"}))
        finally:
            _current_loop_ctx.reset(token)
        assert "content" in out

    def test_has_evicted_is_per_room(self) -> None:
        ev = ToolEviction()

        token = _in_room("room-a")
        try:
            ev.maybe_evict(_BIG, "tc1")
            assert ev.has_evicted
        finally:
            _current_loop_ctx.reset(token)

        token = _in_room("room-b")
        try:
            assert not ev.has_evicted
        finally:
            _current_loop_ctx.reset(token)

    def test_fallback_scope_outside_tool_loop(self) -> None:
        ev = ToolEviction()
        ev.maybe_evict(_BIG, "tc1")
        assert ev.has_evicted
        out = json.loads(ev.handle_read({"result_id": "evicted_tc1"}))
        assert "content" in out
