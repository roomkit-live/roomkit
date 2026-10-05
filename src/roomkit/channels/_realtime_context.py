"""Context variables for the tool calls of a channel hosting a realtime model."""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.channels._realtime_tool_calls import RealtimeToolCall
    from roomkit.voice.base import VoiceSession

_current_voice_session: contextvars.ContextVar[VoiceSession | None] = contextvars.ContextVar(
    "_current_voice_session",
    default=None,
)


def get_current_voice_session() -> VoiceSession | None:
    """Get the voice session for the current tool call.

    Available inside the tool handlers of every channel hosting a realtime
    model (a realtime voice channel, an audio-video one, a conference with
    a realtime model plugged in), on every door a call is served on.
    Returns None outside of a tool call context.
    """
    return _current_voice_session.get()


class _ServedCall:
    """The provider call a tool-call task is serving.

    A reconnect its own handler causes (a handoff reconfiguring its session)
    orphans it like every other call, but the model did not abandon it: the
    handler runs on, its id released, since the new socket never issued it
    (RFC §9.3, §12.4). A task the handler starts inherits the record, and one
    can outlive the call (a provider's new receive loop does); ``finished``
    keeps it from naming a call that has ended. The record names the call
    itself, not its id: a vendor may issue the id again once the call's
    result went out (RFC §12.4), and the call it then names is another one.
    """

    __slots__ = ("call", "finished")

    def __init__(self, call: RealtimeToolCall) -> None:
        self.call = call
        self.finished = False


_served_call: contextvars.ContextVar[_ServedCall | None] = contextvars.ContextVar(
    "_served_call",
    default=None,
)


@contextlib.contextmanager
def serving_call(call: RealtimeToolCall) -> Iterator[None]:
    """Run a provider call's handling as that call's own context."""
    served = _ServedCall(call)
    token = _served_call.set(served)
    try:
        yield
    finally:
        served.finished = True
        _served_call.reset(token)


def _this_task_serves(call: RealtimeToolCall) -> _ServedCall | None:
    served = _served_call.get()
    if served is None or served.finished or served.call is not call:
        return None
    return served


def spare_own_orphaned_call(call: RealtimeToolCall) -> bool:
    """Whether the orphaned call is the one this task's handler is serving.

    Called where a provider reports orphaned calls. When the report runs
    inside the call's own handler, or a task it started, that handler caused
    the reconnect: the call is not to be interrupted, and it releases its id,
    so its result is not sent (RFC §12.4).
    """
    if _this_task_serves(call) is None:
        return False
    call.released = True
    return True
