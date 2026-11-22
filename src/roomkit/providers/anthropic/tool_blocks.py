"""The ``tool_use`` blocks of one streamed Anthropic response, folded into calls.

A block's call is announced when the block opens, its arguments stream as
deltas, and it is handed to the loop when the block closes. Every composition
event names the call by the id it ends with, and a call keeps that id and the
arguments that streamed even when the response ended before its block closed
(RFC §6.4).
"""

from __future__ import annotations

from typing import Any

from roomkit.providers.ai.base import StreamToolCall, StreamToolCallDelta
from roomkit.providers.ai.tool_calls import (
    CallIds,
    call_garbled,
    call_partial,
    tool_arguments,
    unreadable_arguments,
)
from roomkit.providers.anthropic.stop_reasons import finish_reason_of


class ToolUseBlocks:
    """The ``tool_use`` blocks of one streamed response."""

    def __init__(self) -> None:
        self._open: dict[int, dict[str, str]] = {}
        self._closed: set[str] = set()  # the server's ids of the closed blocks
        self._ids = CallIds()

    def open(self, index: int, block: Any) -> StreamToolCallDelta:
        """Announce a block's call at once, before a single argument byte, so
        a host can say what is being composed for the whole composition."""
        call_id = self._ids(block.id, block.name)
        self._open[index] = {
            "id": call_id,
            "server_id": block.id or "",
            "name": block.name,
            "input_json": "",
        }
        return StreamToolCallDelta(id=call_id, name=block.name, index=index, arguments_delta="")

    def add(self, index: int, fragment: str) -> StreamToolCallDelta | None:
        """Fold a fragment of a block's arguments in; its composition event."""
        held = self._open.get(index)
        if held is None:
            return None
        held["input_json"] += fragment
        return StreamToolCallDelta(
            id=held["id"], name=held["name"], index=index, arguments_delta=fragment
        )

    def close(self, index: int) -> StreamToolCall | None:
        """The call a closed block is: one of its own, even under a server id
        another block carried. A closed block always parses: one that does not
        was cut by ``max_tokens``."""
        held = self._open.pop(index, None)
        if held is None:
            return None
        self._closed.add(held["server_id"])
        raw = held["input_json"]
        return StreamToolCall(
            id=held["id"],
            name=held["name"],
            arguments=tool_arguments(raw),
            partial=unreadable_arguments(raw),
        )

    def remaining(self, final: Any) -> list[StreamToolCall]:
        """The calls the stream did not close, read off the final message.

        A block the stream opened keeps the id its composition announced and
        the arguments that streamed, not the SDK's parse of what arrived. It
        is partial by the rule every provider follows (RFC §6.4): its
        arguments do not read, or the response was cut over it (the output
        cap, the context window, a refusal, or no stop reason at all) before
        argument text that reads arrived. Anthropic sends no stop for a block
        a cut stops, so a cut call ends here, often with nothing streamed yet.
        """
        opened = {held["server_id"]: held for held in self._open.values()}
        stop = finish_reason_of(final.stop_reason)
        calls: list[StreamToolCall] = []
        for block in final.content:
            if block.type != "tool_use" or block.id in self._closed:
                continue
            held = opened.get(block.id)
            raw: Any = held["input_json"] if held is not None else block.input
            calls.append(
                StreamToolCall(
                    id=held["id"] if held is not None else self._ids(block.id, block.name),
                    name=block.name,
                    arguments=tool_arguments(raw),
                    partial=call_partial(raw, stop),
                    garbled=call_garbled(raw, stop),
                )
            )
        return calls
