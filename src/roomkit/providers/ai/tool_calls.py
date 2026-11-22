"""What a provider hands the tool loop for a call, the same on every provider.

Each provider reads its own wire, but the loop must receive one thing for one
call (RFC §6.4): a mapping of arguments and never an error, an id no other
call of the response carries, and a mark on a call whose arguments do not
read, which never runs. The rules live here once, so a provider cannot drift
from the others by reading its wire its own way.
"""

from __future__ import annotations

import json
import math
from typing import Any
from uuid import uuid4

from roomkit.providers.ai.finish_reason import FinishReason

# The endings the core reads, in the contract's one vocabulary: each provider
# maps its vendor's words onto it, so no rule here knows a vendor's word.
_CALL_CUTTING_FINISH_REASONS = frozenset(
    {
        FinishReason.LENGTH,
        FinishReason.ERROR,
        FinishReason.CONTENT_FILTER,
        FinishReason.REFUSAL,
    }
)
"""Endings that can stop a call mid-arguments: running out of room, a failed
generation, a filter or a refusal stopping the stream."""

_MALFORMED_CALL_FINISH_REASONS = frozenset(
    {FinishReason.MALFORMED_CALL, FinishReason.UNEXPECTED_CALL}
)
"""Endings where the model called a tool and the provider handed no call over,
so none reached the loop."""

_REFUSAL_FINISH_REASONS = frozenset({FinishReason.REFUSAL, FinishReason.CONTENT_FILTER})
"""Endings where the provider withheld the answer: the model refused, or a
filter declined the request."""


MALFORMED_CALL_NUDGE = (
    "Your last tool call could not be parsed, so it did not run. Call the tool "
    "again with arguments that are valid JSON matching its parameters, or answer "
    "in plain text."
)
"""What a model is told when its provider could not parse its tool call, on
the text loops and on a speech-to-speech session alike (RFC §6.4, §12.4)."""

UNEXPECTED_CALL_NUDGE = (
    "The tool you called is not available in this request, so it did not run. "
    "Use one of the declared tools, or answer in plain text."
)
"""What a model is told when it called a tool its request did not enable
(``unexpected_call``): asking it to call the same tool again would end the
same way (RFC §6.4)."""


def malformed_call_nudge(finish_reason: FinishReason | None) -> str:
    """What a model is told for a call its provider would not hand over, by
    the ending's cause: one it could not parse, or one to a tool the request
    did not enable."""
    if finish_reason == FinishReason.UNEXPECTED_CALL:
        return UNEXPECTED_CALL_NUDGE
    return MALFORMED_CALL_NUDGE


def is_malformed_call(finish_reason: FinishReason | None) -> bool:
    """Whether a response ended on a tool call its provider would not hand
    over: one it could not parse, or one to a tool the request did not enable."""
    return finish_reason in _MALFORMED_CALL_FINISH_REASONS


def is_natural_stop(finish_reason: FinishReason | None) -> bool:
    """Whether the model ended the response itself: ``stop``."""
    return finish_reason == FinishReason.STOP


def is_refusal(finish_reason: FinishReason | None) -> bool:
    """Whether the provider withheld the answer: ``refusal`` or ``content_filter``."""
    return finish_reason in _REFUSAL_FINISH_REASONS


def is_truncation(finish_reason: FinishReason | None) -> bool:
    """Whether a response ended by running out of room, ``length``: its output
    cap, or its context window filling up mid-answer."""
    return finish_reason == FinishReason.LENGTH


def tool_arguments(raw: Any) -> dict[str, Any]:
    """A call's arguments as a mapping, never an error.

    A mapping passes through. No arguments (nothing, blank text, JSON
    ``null``) are ``{}``. Text that parses to a JSON object is that object;
    anything else (invalid JSON, an array, a fragment the output cap cut) is
    kept whole under ``raw``, and the call is ``partial``
    (:func:`unreadable_arguments`).
    """
    read = _read_arguments(raw)
    return {"raw": raw} if read is None else read


def readable_arguments(raw: Any) -> dict[str, Any] | str:
    """A call's arguments as a mapping, or the text the model wrote when they
    do not read as one: how a realtime provider hands a call to
    ``on_tool_call``, so the channel refuses it rather than run a tool on a
    mapping that only passes for arguments (RFC §12.4)."""
    read = _read_arguments(raw)
    if read is not None:
        return read
    return raw if isinstance(raw, str) else json.dumps(raw, default=str)


class CutArguments(str):
    """The argument text of a realtime call its response cut before they
    were whole: the model's text, which the channel refuses as cut off
    (RFC §6.4, §12.4), where text that merely does not read is refused as
    unreadable."""


def realtime_call_arguments(raw: Any, *, cut: bool) -> dict[str, Any] | str:
    """A realtime call's arguments as its provider hands them to
    ``on_tool_call``, the response *cut* over the call or not: a mapping when
    the call may run, else the model's text, a :class:`CutArguments` when the
    cut stopped it (:func:`partial_when`, the rule every provider applies)."""
    if not partial_when(raw, cut=cut):
        return readable_arguments(raw)
    text = readable_arguments(raw) if unreadable_arguments(raw) else ""
    return CutArguments(text) if cut else str(text)


def unreadable_arguments(raw: Any) -> bool:
    """Whether a call's arguments do not read as an object, which makes the
    call ``partial``: it never runs, whatever the provider and whatever stop
    reason the response gave (RFC §6.4)."""
    return _read_arguments(raw) is None


def _finite_number(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite tool argument: {text}")
    return value


def _load_arguments(text: str) -> Any:
    # JSON names no non-finite numbers. A floating exponent can also exceed
    # the finite range while it is being decoded.
    return json.loads(text, parse_float=_finite_number, parse_constant=_finite_number)


def _read_arguments(raw: Any) -> dict[str, Any] | None:
    """A call's arguments as a mapping, or ``None`` when they do not read as one."""
    if isinstance(raw, dict):
        return raw
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}
    if not isinstance(raw, str):
        return None
    try:
        parsed = _load_arguments(raw)
    except (ValueError, RecursionError):
        return None
    if parsed is None:
        return {}
    return parsed if isinstance(parsed, dict) else None


def arguments_cut(raw: Any) -> bool:
    """Whether argument text is not valid JSON: what a cut leaves of a call
    still being written. Complete JSON that is not an object (an array) is
    whole, however unusable."""
    if not isinstance(raw, str) or not raw.strip():
        return False
    try:
        _load_arguments(raw)
    except (ValueError, RecursionError):
        return True
    return False


def call_partial(raw: Any, finish_reason: FinishReason | None, *, last: bool = True) -> bool:
    """Whether a call must not run: its arguments do not read, or it is the
    response's *last* call and the response ended on something that stops a
    call mid-arguments (running out of room, a content filter, a refusal) or on
    nothing at all, a stream that stopped without a stop reason, before
    argument text that reads arrived (RFC §6.4)."""
    return partial_when(raw, cut=_cuts_calls(finish_reason, last=last))


def partial_when(raw: Any, *, cut: bool) -> bool:
    """Whether a call must not run, the response *cut* short over it or not.

    Its arguments do not read, or the response was cut and they did not
    arrive as text that reads: nothing under a cut is no evidence of no
    arguments, and a parse of what arrived (a vendor SDK reads a fragment
    leniently) no evidence that they are whole.
    """
    if unreadable_arguments(raw):
        return True
    return cut and not (isinstance(raw, str) and raw.strip())


def call_cut(raw: Any, finish_reason: FinishReason | None, *, last: bool = True) -> bool:
    """Whether the response was cut short over a call's arguments: it is the
    response's *last* call, the response ended on something that stops a call
    mid-arguments, and they did not arrive whole (:func:`call_partial`). The
    call is ``partial`` and ``cut``."""
    return _cuts_calls(finish_reason, last=last) and partial_when(raw, cut=True)


def _cuts_calls(finish_reason: FinishReason | None, *, last: bool) -> bool:
    """Whether a response that ended so can have stopped this call mid-arguments.

    Only its last call: the model writes calls one after the other, so a call
    another followed was closed by it, whatever then cut the response.
    """
    if not last:
        return False
    return finish_reason is None or finish_reason in _CALL_CUTTING_FINISH_REASONS


def call_garbled(raw: Any, finish_reason: FinishReason | None, *, last: bool = True) -> bool:
    """Whether the model wrote a call's arguments unreadable: they do not
    read, and the response did not cut them short (:func:`call_cut`). The call
    is ``partial`` and ``garbled``."""
    return unreadable_arguments(raw) and not call_cut(raw, finish_reason, last=last)


def partial_call_error(name: str, *, garbled: bool) -> dict[str, Any]:
    """What the model reads for a ``partial`` call: nothing ran, and why."""
    return unreadable_call_error(name) if garbled else cut_call_error(name)


def cut_call_error(name: str) -> dict[str, Any]:
    """What the model reads for a call the response cut: nothing ran."""
    return {
        "error": "Tool call cut off",
        "tool": name,
        "hint": (
            "This call was cut off before its arguments were complete, so it did "
            "not run. Call it again if it is still needed, with shorter arguments "
            "if the response ran out of room."
        ),
    }


def unreadable_call_error(name: str) -> dict[str, Any]:
    """What the model reads for a call written with unreadable arguments:
    nothing ran."""
    return {
        "error": "Tool call arguments unreadable",
        "tool": name,
        "hint": (
            "This call's arguments are not a JSON object, so it did not run. Call "
            "it again with arguments that are a valid JSON object matching the "
            "tool's parameters."
        ),
    }


def nameless_call_error() -> dict[str, Any]:
    """What the model reads for a call that named no tool: nothing ran."""
    return {
        "error": "Tool call named no tool",
        "hint": "This call named no tool, so it did not run. Call a tool by its name.",
    }


def minted_call_id(name: str) -> str:
    """An id for a call its server gave none, unique across turns."""
    return f"call_{name}_{uuid4().hex[:12]}"


class CallIds:
    """Hands each call of one response its id.

    The server's id when it gave one no earlier call of the response took,
    a minted one otherwise: results, eviction and cancellation all find a
    call by its id, so two calls must never share one.
    """

    def __init__(self) -> None:
        self._taken: set[str] = set()

    def __call__(self, server_id: Any, name: str) -> str:
        given = str(server_id) if server_id else ""
        call_id = given if given and given not in self._taken else minted_call_id(name)
        self._taken.add(call_id)
        return call_id
