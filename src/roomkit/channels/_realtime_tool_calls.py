"""The realtime tool calls in flight on a channel, one record per call (RFC §12.4).

Every door of a speech-to-speech channel (the provider's function call, a call
recovered from speech, a reasoning backend's call) opens a record here when a
call arrives and closes it when the call ends. The record is where the call's
one delivery and one report are claimed, so a reconfiguration that fails once
the result went out, a cancellation that lands after it, or the session's end
cannot add a second outcome to the same call.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomkit.providers.ai.tool_calls import (
    CutArguments,
    cut_call_error,
    nameless_call_error,
    tool_arguments,
    unreadable_call_error,
)
from roomkit.tools.result import call_id_in_flight_error

if TYPE_CHECKING:
    from roomkit.tools._outcome import ToolOutcome
    from roomkit.voice.base import VoiceSession

logger = logging.getLogger("roomkit.channels.realtime_tools")


@dataclass(eq=False)
class RealtimeToolCall:
    """One realtime tool call: what was called, the task serving it, and
    whether its result went out and its outcome was reported."""

    session: VoiceSession
    call_id: str
    name: str
    arguments: dict[str, Any]
    room_id: str | None = None
    """The room the session served when the call ran."""
    unreadable: str | None = None
    """What the model reads when the call cannot be read (its arguments, or a
    ``call_tool`` transport's, are not an object, or it named no tool):
    refused before the gate."""
    unanswerable: str | None = None
    """Why no result can be sent for the call (it came without an id, or under
    an id that still names a call in flight): refused on the normal path,
    reported, and nothing sent (RFC §12.4)."""
    mutes: bool = False
    """The call holds the session's input muted while it runs."""
    structured_content: dict[str, Any] | None = None
    """The structured copy its handler left on the tool call context (MCP
    ``structuredContent``), carried to ON_TOOL_CALL (RFC §9.3)."""
    task: asyncio.Task[Any] | None = field(default=None, repr=False)
    carrier: asyncio.Task[Any] | None = field(default=None, repr=False)
    """The task that holds the call's task, when another one does (a
    backend's delegation): an ending the call caused spares it too."""
    caused_ending: bool = False
    """Its handler caused an ending that took it off the books, which spared
    it: its delegation, if any, ends once it returns."""
    delivered: bool = False
    released: bool = False
    """The provider no longer waits for its result (it abandoned the call, or
    a reconnect the call's own handler caused orphaned it): its id is free,
    and nothing is sent for it (RFC §12.4)."""
    reported: bool = False
    owed: ToolOutcome | None = None
    """The failure the model reads, kept from its delivery, which the call's
    report follows: a report an ending cuts still owes it (RFC §9.3)."""

    @classmethod
    def from_provider(
        cls,
        session: VoiceSession,
        call_id: str,
        name: str | None,
        arguments: dict[str, Any] | str,
        **fields: Any,
    ) -> RealtimeToolCall:
        """The call a provider handed ``on_tool_call``: one that named no tool,
        or whose arguments came as the model's text that does not read as an
        object, is unreadable, its arguments kept under ``raw`` for its
        reports; one its response cut (:class:`CutArguments`) reads as cut off
        (RFC §6.4, §12.4)."""
        if not name:
            logger.warning(
                "Provider sent tool call %s naming no tool: it does not run", call_id or "(no id)"
            )
            readable = arguments if isinstance(arguments, dict) else tool_arguments(arguments)
            refusal = json.dumps(nameless_call_error())
            return cls(session, call_id, "", readable, unreadable=refusal, **fields)
        if isinstance(arguments, dict):
            return cls(session, call_id, name, arguments, **fields)
        cut = isinstance(arguments, CutArguments)
        logger.warning(
            "Provider sent %s arguments for tool call %s (%s): it does not run",
            "cut" if cut else "unreadable",
            name,
            call_id,
        )
        refusal = json.dumps(cut_call_error(name) if cut else unreadable_call_error(name))
        return cls(session, call_id, name, tool_arguments(arguments), unreadable=refusal, **fields)

    @property
    def interruptible(self) -> bool:
        """Whether a cancellation still has something to interrupt: the
        call's task runs and its outcome was not reported (RFC §9.3)."""
        return not self.reported and self.task is not None and not self.task.done()

    @property
    def holds_id(self) -> bool:
        """Whether the call's id still names it: its result has not gone out
        and the provider still waits for it (RFC §12.4)."""
        return not (self.delivered or self.released)

    def claim_report(self) -> bool:
        """Claim the call's one report: False when it was already made."""
        if self.reported:
            return False
        self.reported = True
        return True


class ToolCallBook:
    """The realtime tool calls in flight, per session.

    An id names its call from the call until its result goes out or the
    provider abandons it (RFC §12.4): a second call under it meanwhile is
    refused, while one after it is a new call, recorded beside the first,
    which may still be finishing its report or its interrupted handler. The
    provider frees the id at the same step, when it sends the result or
    reports the abandonment.
    """

    def __init__(self) -> None:
        self._calls: dict[str, dict[str, list[RealtimeToolCall]]] = {}

    def open(self, call: RealtimeToolCall) -> bool:
        """Record *call*: False, the call marked unanswerable, when it came
        without an id or under one that still names a call in flight, which
        then keeps the id (RFC §12.4)."""
        if not call.call_id:
            what = f"Tool call '{call.name}'" if call.name else "A tool call"
            call.unanswerable = json.dumps({"error": f"{what} came without an id"})
            return False
        calls = self._calls.setdefault(call.session.id, {})
        held = calls.get(call.call_id, [])
        if any(earlier.holds_id for earlier in held):
            call.unanswerable = call_id_in_flight_error(call.call_id)
            return False
        calls[call.call_id] = [*held, call]
        return True

    def close(self, call: RealtimeToolCall) -> None:
        """Forget *call*, whatever call its id names now."""
        calls = self._calls.get(call.session.id)
        held = calls.get(call.call_id) if calls is not None else None
        if calls is None or held is None or call not in held:
            return
        held = [other for other in held if other is not call]
        if held:
            calls[call.call_id] = held
        else:
            del calls[call.call_id]
        if not calls:
            del self._calls[call.session.id]

    def holds(self, call: RealtimeToolCall) -> bool:
        """Whether *call* is still on the books."""
        held = (self._calls.get(call.session.id) or {}).get(call.call_id, [])
        return any(other is call for other in held)

    def get(self, session_id: str, call_id: str) -> RealtimeToolCall | None:
        """The latest call under *call_id*: the one its id names now."""
        held = (self._calls.get(session_id) or {}).get(call_id)
        return held[-1] if held else None

    def release(self, session_id: str, call_id: str) -> RealtimeToolCall | None:
        """Free *call_id* from the call the provider abandoned under it, and
        return that call; ``None`` when the id names no call that holds it.

        The provider freed the id when it reported the abandonment: nothing is
        sent for the call from then on, its observers told or not, and a call
        the provider issues under the id is a new call, answered (RFC §12.4).
        """
        call = self.get(session_id, call_id)
        if call is None or not call.holds_id:
            return None
        call.released = True
        return call

    def busy(self, session_id: str) -> bool:
        """Whether a call is in flight on the session."""
        return bool(self._calls.get(session_id))

    def muting(self, session_id: str) -> bool:
        """Whether a call holding the input muted is in flight on the session."""
        return any(call.mutes for call in self._all(session_id))

    def take(self, session_id: str) -> list[RealtimeToolCall]:
        """Every call in flight on the session, off the books: the session ends."""
        calls = self._all(session_id)
        self._calls.pop(session_id, None)
        return calls

    def _all(self, session_id: str) -> list[RealtimeToolCall]:
        return [call for held in (self._calls.get(session_id) or {}).values() for call in held]
