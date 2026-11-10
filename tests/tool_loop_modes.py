"""Drive an AIChannel's tool loop and read what it produced.

An AIChannel runs one tool loop whatever its provider streams (RFC §6.4): a
provider that does not stream is read through its ``generate()``. A tool-loop
test takes the ``streaming`` fixture (``tests/conftest.py``) to run against
both kinds of provider, and drives the turn through the helpers below, which
read the loop's outcome into a :class:`LoopRun`:

- :func:`respond` goes through ``on_event``: pass the fixture's value to
  ``MockAIProvider(streaming=...)``. Under the fixture it checks that the
  provider streams in the mode the test runs in.
- :func:`run_tool_loop` runs the loop directly on a context.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from roomkit.channels import FallbackRouter
from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.enums import EventType
from roomkit.models.event import RoomEvent, ToolCallContent
from roomkit.models.streaming import (
    LoopEndMarker,
    LoopEndReason,
    ToolCallEndMarker,
    ToolCallStartMarker,
)
from roomkit.providers.ai.base import AIContext

# The mode the running test was parametrised with, set by the ``streaming``
# fixture for the length of one test; ``None`` outside it.
_expected_streaming: bool | None = None


def expect_streaming(value: bool | None) -> None:
    """Record the mode the running test runs in (the ``streaming`` fixture)."""
    global _expected_streaming
    _expected_streaming = value


@dataclass
class LoopCall:
    """One tool call the loop ran, as it reported it.

    ``requested`` is what the model asked for (the call's start); ``arguments``
    is what the loop reports as executed (the call's end), after folds and a
    ``BEFORE_TOOL_USE`` rewrite.
    """

    name: str
    id: str
    requested: dict[str, Any]
    arguments: dict[str, Any]
    result: Any
    failed: bool
    error: str | None = None
    structured_content: dict[str, Any] | None = None


@dataclass
class LoopRun:
    """What one turn of a tool loop produced, whichever loop ran it.

    ``said`` is every non-empty text segment in order: what the model said
    before each tool round, then its answer. ``text`` is the answer alone.
    ``reason`` is the loop's end reason (``LoopEndMarker.reason``); ``None``
    when the loop did not reach its end. ``metadata`` is the reply's response
    metadata (the turn's live record), read once the turn has ended.
    """

    text: str = ""
    said: list[str] = field(default_factory=list)
    calls: list[LoopCall] = field(default_factory=list)
    reason: LoopEndReason | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def open_tool_calls(events: list[RoomEvent]) -> list[str]:
    """Ids of the stored TOOL_CALL_START events that no TOOL_CALL_END closes."""
    ended = {
        e.content.tool_id
        for e in events
        if e.type == EventType.TOOL_CALL_END and isinstance(e.content, ToolCallContent)
    }
    return [
        e.content.tool_id
        for e in events
        if e.type == EventType.TOOL_CALL_START
        and isinstance(e.content, ToolCallContent)
        and e.content.tool_id not in ended
    ]


async def run_tool_loop(channel: AIChannel, context: AIContext) -> LoopRun:
    """Run the channel's tool loop for ``context``."""
    return await _read_stream(channel._run_streaming_tool_loop(context))


async def respond(
    channel: AIChannel, event: RoomEvent, binding: ChannelBinding, context: RoomContext
) -> LoopRun:
    """Deliver ``event`` to the channel and read its reply. With a router,
    the check reads the primary of a fallback router, the provider a turn
    starts on."""
    provider = channel.provider
    if isinstance(channel.router, FallbackRouter):
        provider = channel.router.primary
    streams = provider.supports_structured_streaming if provider is not None else None
    if _expected_streaming is not None and streams not in (None, _expected_streaming):
        raise AssertionError(
            f"the test runs streaming={_expected_streaming} but the provider "
            f"streams={streams}: pass the fixture to the provider"
        )
    return await read_reply(await channel.on_event(event, binding, context))


async def read_reply(output: ChannelOutput) -> LoopRun:
    """Read a reply ``on_event`` handed back: its stream runs the turn."""
    run = await _read_stream(output.response_stream) if output.response_stream else LoopRun()
    run.metadata = dict(output.response_metadata or {})
    return run


async def _read_stream(stream: AsyncIterator[Any]) -> LoopRun:
    run = LoopRun()
    pending: list[str] = []
    requested: dict[str, dict[str, Any]] = {}
    in_round = False
    async for delta in stream:
        if isinstance(delta, str):
            pending.append(delta)
            in_round = False
        elif isinstance(delta, ToolCallStartMarker):
            if not in_round:
                _flush(run, pending)
                in_round = True
            requested[delta.tool_id] = delta.arguments
        elif isinstance(delta, ToolCallEndMarker):
            in_round = False
            run.calls.append(
                LoopCall(
                    name=delta.tool_name,
                    id=delta.tool_id,
                    requested=requested.get(delta.tool_id, {}),
                    arguments=delta.arguments,
                    result=delta.result,
                    failed=delta.status == "failed",
                    error=delta.error,
                    structured_content=delta.structured_content,
                )
            )
        elif isinstance(delta, LoopEndMarker):
            run.reason = delta.reason
    run.text = "".join(pending)
    _flush(run, pending)
    return run


def _flush(run: LoopRun, pending: list[str]) -> None:
    text = "".join(pending)
    pending.clear()
    if text:
        run.said.append(text)
