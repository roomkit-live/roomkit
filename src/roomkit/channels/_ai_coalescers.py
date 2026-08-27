"""Rate-capped buffers that project a turn's in-progress work onto the bus.

A model's reasoning and its tool-call arguments are both produced token by
token, and both are worth showing live. Publishing one realtime event per token
is one ephemeral event, one fan-out and one WS serialise per token, thousands
for a long trace, all on the shared event loop. These batch that onto a window.
"""

from __future__ import annotations

import time
from typing import Any

from roomkit.channels._ai_events import THINKING_PREVIEW_LIMIT
from roomkit.realtime.base import EphemeralEventType


class _Window:
    """The rate cap the coalescers share, and the only thing they share.

    Both bound how often one round's in-progress work reaches the bus, on the
    same two settings. What they publish has nothing in common: one joins text
    and splits it on a preview cap, the other carries per-call sizes. So the
    cap lives here and the payloads stay with their owners.
    """

    def __init__(self, *, flush_ms: float, flush_chars: int) -> None:
        self._flush_ms = flush_ms
        self._flush_chars = flush_chars
        self._pending = 0
        self._last_publish = time.monotonic()

    def add(self, chars: int) -> bool:
        """Account for ``chars``; ``True`` once the window is spent.

        A window of ``0`` ms disables batching: every delta publishes as it
        arrives.
        """
        self._pending += chars
        if self._flush_ms <= 0:
            return True
        elapsed_ms = (time.monotonic() - self._last_publish) * 1000.0
        return self._pending >= self._flush_chars or elapsed_ms >= self._flush_ms

    def reset(self) -> None:
        """Start a fresh window. Called by a coalescer once it has published."""
        self._pending = 0
        self._last_publish = time.monotonic()


class _ThinkingCoalescer:
    """Batches per-token thinking deltas into one ``THINKING_DELTA`` publish per window.

    Reasoning models emit one ``StreamThinkingDelta`` per token, and publishing
    each on the realtime bus is one ephemeral event + fan-out + WS serialise per
    token — thousands for a long trace, all on the shared event loop. Buffering
    and publishing once per time/size window (~80 ms / ~256 chars) cuts that
    10-100x while keeping the reasoning visibly real-time: the UI appends deltas,
    so a coalesced delta renders identically to many small ones. Closing the
    window still publishes its block at ``THINKING_END`` — that block only, and
    capped at ``THINKING_PREVIEW_LIMIT``, so a client that renders the deltas
    already holds more than the terminal event carries.

    A window of ``0`` ms disables batching — every delta publishes immediately.
    Flushes larger than ``_publish_thinking_event``'s preview cap are split
    into multiple publishes so a coalesced delta is never truncated, whatever
    the configured size threshold.
    """

    def __init__(
        self,
        publish: Any,
        room_id: str | None,
        round_idx: int,
        *,
        flush_ms: float,
        flush_chars: int,
    ) -> None:
        self._publish = publish
        self._room_id = room_id
        self._round_idx = round_idx
        self._window = _Window(flush_ms=flush_ms, flush_chars=flush_chars)
        self._pending: list[str] = []

    async def add(self, delta: str) -> None:
        """Buffer a delta; publish the batch once the window is exceeded."""
        self._pending.append(delta)
        if self._window.add(len(delta)):
            await self.flush()

    async def flush(self) -> None:
        """Publish whatever is buffered, then reset the window."""
        if not self._pending:
            return
        text = "".join(self._pending)
        self._pending.clear()
        self._window.reset()
        for i in range(0, len(text), THINKING_PREVIEW_LIMIT):
            await self._publish(
                EphemeralEventType.THINKING_DELTA,
                self._room_id,
                text[i : i + THINKING_PREVIEW_LIMIT],
                self._round_idx,
            )


class _ToolCallDeltaCoalescer:
    """Batches tool-call argument fragments into one ``TOOL_CALL_DELTA`` per window.

    A model composing a large tool argument — a document, an SVG, base64 —
    spends minutes producing tokens that reach no one: the complete call, and
    with it ``TOOL_CALL_START``, only lands once the last fragment is in. This
    publishes the composition as it happens, so a host can say *what* is being
    composed and *how far along* instead of "working".

    It shares :class:`_ThinkingCoalescer`'s window and its two settings, but
    carries sizes rather than text: the payload is every call in flight this
    round with the number of argument characters composed so far, **never the
    argument content** — that can be megabytes or personal data, and
    ``TOOL_CALL_START`` delivers it whole at the end of the round.

    A call's first fragment publishes immediately, whatever the window: the
    tool's name is the signal a host is waiting for. The round closes with a
    terminal frame carrying an empty ``tool_calls`` — see :meth:`close`.
    """

    def __init__(
        self,
        publish: Any,
        room_id: str | None,
        round_idx: int,
        *,
        flush_ms: float,
        flush_chars: int,
    ) -> None:
        self._publish = publish
        self._room_id = room_id
        self._round_idx = round_idx
        self._window = _Window(flush_ms=flush_ms, flush_chars=flush_chars)
        self._calls: dict[int, dict[str, Any]] = {}
        self._published = False

    async def add(self, index: int, call_id: str, name: str, chars: int) -> None:
        """Fold one fragment in; publish if the call is new or the window is exceeded.

        Keyed on ``index`` — the provider's slot for the call — because that
        is the one identifier every provider carries on every fragment. An id
        can arrive late or not at all (PolarGrid's non-streaming path keeps an
        explicit fallback for a missing one), and keying on it would split a
        single call across two entries, or merge two parallel calls that have
        none yet into one frame with the wrong name and the sum of both sizes.
        """
        call = self._calls.get(index)
        if call is None:
            self._calls[index] = {"id": call_id, "name": name, "arguments_chars": chars}
            await self.flush()
            return
        if call_id:
            call["id"] = call_id
        if name:
            call["name"] = name
        call["arguments_chars"] += chars
        if self._window.add(chars):
            await self.flush()

    async def flush(self) -> None:
        """Publish every call in flight with its running size, then reset the window."""
        if not self._calls:
            return
        self._published = True
        self._window.reset()
        await self._publish(
            EphemeralEventType.TOOL_CALL_DELTA,
            self._room_id,
            [dict(call) for call in self._calls.values()],
            self._round_idx,
        )

    async def close(self) -> None:
        """Publish the terminal frame — an empty ``tool_calls`` — if anything was.

        A round can end without ever reaching ``TOOL_CALL_START``: cancelled
        mid-composition, out of rounds, out of time, or handed to an external
        tool provider that publishes its own events. A host shown a
        composition and never told it ended is stuck on "composing
        publish_artifact (3.2 kB)" exactly as it used to be stuck on
        "working" — the defect this class exists to remove, moved rather than
        fixed. So the round always closes, and an empty list is what says so.
        """
        if not self._published:
            return
        self._calls.clear()
        self._published = False
        await self._publish(
            EphemeralEventType.TOOL_CALL_DELTA,
            self._room_id,
            [],
            self._round_idx,
        )
