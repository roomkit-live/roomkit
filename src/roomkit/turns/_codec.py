"""What a turn's objects become on the wire, and back (RFC §10.6.3).

A stream item (a text delta or one of the stream markers) and a channel's
output each have one wire form (an exception's is ``_error_codec``'s).
Decoding builds new objects: nothing the executor holds is handed to the
owner. What the owner sends of a room (a request, a channel control message)
crosses as the response-metadata record does: a value JSON has no form for
is written as its text. In memory, a request's events cross as they are
(:class:`CarriedRequest`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic_core import from_json, to_json, to_jsonable_python

from roomkit.models.channel import ChannelOutput
from roomkit.models.response_metadata import ResponseMetadata
from roomkit.models.streaming import (
    InstructionReadMarker,
    LoopEndMarker,
    SegmentBreakMarker,
    ThinkingDeltaMarker,
    ToolCallEndMarker,
    ToolCallStartMarker,
    UsageMarker,
)
from roomkit.providers.ai.base import AIImagePart, AITextPart
from roomkit.providers.ai.tool_intent import with_intent_field
from roomkit.turns._error_codec import ErrorClasses, error_record
from roomkit.turns.wire import ChannelCall, ChannelControlEnvelope, TurnRequest, WireOutput

if TYPE_CHECKING:
    from roomkit.models.event import RoomEvent
    from roomkit.turns.wire import ChannelControl

# -- Stream items ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class _ItemKind:
    """One kind of stream item: its type, and its wire form both ways."""

    name: str
    type: type
    encode: Callable[[Any, int | None], dict[str, Any]]
    """The item's fields; the second argument is a call end's start frame."""
    decode: Callable[[dict[str, Any]], Any]
    delta: bool = False
    """Read ahead of the owner's reader within a pull (RFC §10.6.4)."""


def _encode_start(item: ToolCallStartMarker, start: int | None) -> dict[str, Any]:
    fields = {
        "tool_name": item.tool_name,
        "tool_id": item.tool_id,
        "arguments": item.arguments,
        "ran_with": item.ran_with,
    }
    return with_intent_field(fields, item.intent)


def _decode_start(data: dict[str, Any]) -> ToolCallStartMarker:
    return ToolCallStartMarker(
        tool_name=data["tool_name"],
        tool_id=data["tool_id"],
        arguments=data["arguments"],
        ran_with=data["ran_with"],
        intent=data.get("intent"),
    )


def _decode_loop_end(data: dict[str, Any]) -> LoopEndMarker:
    fields = {f.name for f in dataclasses.fields(LoopEndMarker)}
    return LoopEndMarker(**{k: v for k, v in data.items() if k in fields})


_ITEM_KINDS = (
    _ItemKind(
        "text", str, lambda item, start: {"text": item}, lambda data: data["text"], delta=True
    ),
    _ItemKind(
        "thinking",
        ThinkingDeltaMarker,
        lambda item, start: {"thinking": item.thinking},
        lambda data: ThinkingDeltaMarker(thinking=data["thinking"]),
        delta=True,
    ),
    _ItemKind(
        "instruction_read",
        InstructionReadMarker,
        lambda item, start: {"text": item.text},
        lambda data: InstructionReadMarker(text=data["text"]),
    ),
    _ItemKind("tool_start", ToolCallStartMarker, _encode_start, _decode_start),
    _ItemKind(
        "tool_end",
        ToolCallEndMarker,
        lambda item, start: {"start": start, **encode_end(item)},
        lambda data: decode_end(data),
    ),
    _ItemKind(
        "segment_break",
        SegmentBreakMarker,
        lambda item, start: {},
        lambda data: SegmentBreakMarker(),
    ),
    _ItemKind(
        "usage",
        UsageMarker,
        lambda item, start: dataclasses.asdict(item),
        lambda data: UsageMarker(
            usage=data["usage"], model=data.get("model"), provider=data.get("provider")
        ),
        # Read ahead as text is, or the loop would wait at it for its reader
        # where it went on to decide the round: what a stop finds would change.
        delta=True,
    ),
    _ItemKind(
        "loop_end", LoopEndMarker, lambda item, start: dataclasses.asdict(item), _decode_loop_end
    ),
)
_KIND_OF_TYPE = {kind.type: kind for kind in _ITEM_KINDS}
_KIND_NAMED = {kind.name: kind for kind in _ITEM_KINDS}
_DELTA_TYPES = tuple(kind.type for kind in _ITEM_KINDS if kind.delta)
_DELTA_NAMES = frozenset(kind.name for kind in _ITEM_KINDS if kind.delta)


def is_delta(item: Any) -> bool:
    """Whether *item* is a text or reasoning delta or what a generation used,
    which the executor reads ahead of the owner's reader (RFC §10.6.4)."""
    return isinstance(item, _DELTA_TYPES)


def carries_deltas(data: dict[str, Any]) -> bool:
    """Whether the wire item *data* carries deltas: one, or a joined burst."""
    return data.get("kind") in _DELTA_NAMES


def encode_item(item: Any, *, start: int | None = None) -> dict[str, Any]:
    """The wire form of one stream item; *start* is the sequence number of the
    frame that carried a call end's start."""
    kind = _KIND_OF_TYPE.get(type(item)) or next(
        (k for k in _ITEM_KINDS if isinstance(item, k.type)), None
    )
    if kind is None:
        raise TypeError(f"A turn's stream cannot carry a {type(item).__name__} across the backend")
    return {"kind": kind.name, **kind.encode(item, start)}


def decode_item(data: dict[str, Any]) -> Any:
    """A new stream item from its wire form (a call end without its start link)."""
    kind = _KIND_NAMED.get(data["kind"])
    if kind is None:
        raise ValueError(f"Unknown stream item kind {data['kind']!r}")
    return kind.decode(data)


def encode_end(end: ToolCallEndMarker) -> dict[str, Any]:
    """A call end's fields, its result in a form that keeps its type."""
    fields = {
        "tool_name": end.tool_name,
        "tool_id": end.tool_id,
        "arguments": end.arguments,
        "result": encode_result(end.result),
        "status": end.status,
        "duration_ms": end.duration_ms,
        "error": end.error,
        "structured_content": end.structured_content,
        "outcome": end.outcome,
        "refused_but_ran": end.refused_but_ran,
    }
    return with_intent_field(fields, end.intent)


def decode_end(data: dict[str, Any]) -> ToolCallEndMarker:
    """A new call end from its fields."""
    return ToolCallEndMarker(
        tool_name=data["tool_name"],
        tool_id=data["tool_id"],
        arguments=data["arguments"],
        result=decode_result(data["result"]),
        status=data["status"],
        duration_ms=data["duration_ms"],
        error=data["error"],
        structured_content=data["structured_content"],
        outcome=data["outcome"],
        refused_but_ran=data["refused_but_ran"],
        intent=data.get("intent"),
    )


def encode_result(result: Any) -> dict[str, Any]:
    """A tool result: text, content parts, or a JSON value."""
    if (
        isinstance(result, list)
        and result
        and all(isinstance(part, AITextPart | AIImagePart) for part in result)
    ):
        return {"parts": [part.model_dump(mode="json") for part in result]}
    return {"value": result}


def decode_result(data: dict[str, Any]) -> Any:
    if "parts" in data:
        return [
            AITextPart.model_validate(p)
            if p.get("type") == "text"
            else AIImagePart.model_validate(p)
            for p in data["parts"]
        ]
    return data["value"]


# -- Channel output -------------------------------------------------------------


def record_payload(record: Any) -> dict[str, Any]:
    """A response-metadata record as it crosses: a deep copy JSON carries,
    a value JSON has no form for written as its text. One rule for the
    record wherever it crosses, at the output and on every later change."""
    return to_jsonable_python(dict(record), fallback=str)


def encode_output(output: ChannelOutput) -> WireOutput:
    """A channel's output without its stream; its error as a record."""
    return WireOutput(
        responded=output.responded,
        response_events=output.response_events,
        response_metadata=record_payload(output.response_metadata),
        tasks=output.tasks,
        observations=output.observations,
        metadata_updates=output.metadata_updates,
        error=None if output.error is None else error_record(output.error),
    )


def decode_output(wire: WireOutput, errors: ErrorClasses) -> ChannelOutput:
    """A new channel output, its error of a class *errors* rebuilds; its
    record is the one the turn's frames update."""
    return ChannelOutput(
        responded=wire.responded,
        response_events=wire.response_events,
        response_metadata=ResponseMetadata(wire.response_metadata),
        tasks=wire.tasks,
        observations=wire.observations,
        metadata_updates=wire.metadata_updates,
        error=None if wire.error is None else errors.rebuild(wire.error),
    )


# -- What the owner sends -------------------------------------------------------


def request_json(request: TurnRequest) -> str:
    """A turn's request as it crosses. The room's data it carries (the room's
    metadata, an event's) may hold a value an application in the owner's
    process left there that JSON has no form for: it crosses as its text,
    the record's rule (:func:`record_payload`), and the turn runs."""
    return _request_wire(request).decode()


def _request_wire(request: TurnRequest, exclude: dict[str, Any] | None = None) -> bytes:
    """*request*'s wire form, the parts *exclude* names left out."""
    # pydantic_core's to_json: model_dump_json takes no fallback in pydantic
    # 2.9, the floor RoomKit declares.
    return to_json(request, by_alias=False, fallback=str, exclude=exclude)


# The parts of a request the in-memory backend hands over as they are: the
# event the turn answers and the room's history.
_REQUEST_EVENTS: dict[str, Any] = {"event": True, "context": {"recent_events": True}}


@dataclasses.dataclass(frozen=True, slots=True)
class CarriedRequest:
    """A turn's request as the in-memory backend carries it (RFC §10.6.1):
    its events as they are, the rest in its wire form.

    The event the turn answers and the room's history are records nothing
    changes once made: committed events, which the store shares with every
    reader (§14.4), or ones the owner made for the turn (an instruction, a
    strategy's stand-in, a copy with content withheld), which neither the
    owner, once it submitted the request, nor the executor changes. They are
    most of the request: 51 events of history made 87 % of its JSON. The rest
    crosses as :func:`request_json` writes it, so the executor reads its own
    copy of the room, the bindings and the binding, a value JSON has no form
    for as its text.
    """

    rest: bytes
    event: RoomEvent
    history: tuple[RoomEvent, ...]

    @classmethod
    def of(cls, request: TurnRequest) -> CarriedRequest:
        """*request*, its events kept as they are."""
        rest = _request_wire(request, exclude=_REQUEST_EVENTS)
        return cls(rest, request.event, tuple(request.context.recent_events))

    def request(self) -> TurnRequest:
        """The request built anew on the executor's side, around the same events."""
        data = from_json(self.rest)
        data["event"] = self.event
        data["context"]["recent_events"] = list(self.history)
        return TurnRequest.model_validate(data)


def control_json(message: ChannelControl) -> str:
    """A channel control message as it crosses, in its envelope: the room's
    data it carries (a binding, a committed event) as a request carries it.
    A channel call's arguments are its caller's, as a host verb's are: one
    JSON has no form for is refused (``ValueError``), never sent as text."""
    envelope = ChannelControlEnvelope(message=message)
    if isinstance(message, ChannelCall):
        return envelope.model_dump_json()
    return to_json(envelope, by_alias=False, fallback=str).decode()
