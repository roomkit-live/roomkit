"""The messages of a turn's link (RFC §10.6.3).

Every message a turn backend carries is one of these records, serialized to
JSON whatever the transport, the in-memory one included: no object of the turn
crosses between the room owner and the executor, but the events a request
carries in memory, which nothing changes (``_codec.CarriedRequest``). A link
carries an :class:`Envelope` per message, numbered per direction, so a gap or
a duplicate is seen on arrival.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, PrivateAttr

from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.models.steering import SteeringDirective
from roomkit.models.task import Observation, Task
from roomkit.providers.ai.base import AITool


class OwnerTool(BaseModel):
    """A tool the room owner serves a turn, as the request lists it (RFC §10.6.5)."""

    tool: AITool
    waits_by_design: bool = False
    """Bounded by its own bound, not the channel's default call bound (RFC §21.6)."""


_SCOPES_KEPT = 32
"""How many turns' answer scopes a previous-turn record keeps, newest last."""


class PreviousTurn(BaseModel):
    """The room's previous turn of the channel, the one the owner submitted
    last, ended or not (RFC §10.6.2); or its last channel call, which leaves
    the room's state as a turn does."""

    turn_id: str
    """The turn's id, or the channel call's."""
    executor_id: str | None = None
    """The executor that took it; ``None`` when none had said so yet."""
    _scopes: dict[str, bool] = PrivateAttr(default_factory=dict)
    """Per turn of this record's executor in the room, newest last: whether
    its trigger scoped its answers nowhere. Kept on the owner, never sent."""

    def _record_scope(self, turn_id: str, *, unscoped: bool) -> None:
        """Keep whether *turn_id*'s trigger scoped its answers nowhere (RFC
        §5.12), for what its work publishes after it (RFC §8.4)."""
        self._scopes[turn_id] = unscoped
        while len(self._scopes) > _SCOPES_KEPT:
            del self._scopes[next(iter(self._scopes))]

    def _inherit_scopes(self, previous: PreviousTurn | None) -> None:
        """Keep the earlier turns' scopes: a publication may name an earlier
        turn of the executor that still holds the room (RFC §10.6.3)."""
        if previous is not None:
            self._scopes.update(previous._scopes)

    def _unscoped(self, turn_id: str) -> bool:
        """Whether *turn_id*'s trigger scoped its answers nowhere; ``False``
        for a turn this record does not know, a channel call included."""
        return self._scopes.get(turn_id, False)


class _HandedOver(BaseModel):
    """A message that leaves the room's state once an executor may hold it
    (RFC §10.6.2). Its backend marks it as it gives it to an executor, so
    the owner knows whether a failure came before: one that did leaves the
    room's previous turn as it was, one after may follow the message's
    start there. The mark never crosses."""

    _handed: bool = PrivateAttr(default=False)

    def hand_over(self) -> None:
        """Mark the message given to an executor, here or across the
        backend's transport."""
        self._handed = True

    def refused(self) -> None:
        """The executor it was given to refused it unapplied (its process
        closing): no executor holds it, until it is handed again."""
        self._handed = False

    @property
    def handed(self) -> bool:
        """Whether the backend gave the message to an executor."""
        return self._handed


class TurnRequest(_HandedOver):
    """What the owner sends an executor to start one turn."""

    turn_id: str
    room_id: str
    channel_id: str
    event: RoomEvent
    binding: ChannelBinding
    context: RoomContext
    strategy_turn: bool = False
    """A generation a strategy's runner asks: the channel answers, past its
    speak policy and its thinker (RFC §10.6.1)."""
    notes: list[str] = Field(default_factory=list)
    """The runtime's own blocks for the turn (a discussion's turn notes)."""
    in_place: bool = False
    """A discussion's turn: the event keeps its place in the history."""
    owner_tools: list[OwnerTool] = Field(default_factory=list)
    """The tools the owner serves for the channel in the room: the channel
    declares them to its model, and a call to one is a ``tool_call``."""
    room_hooks: list[str] = Field(default_factory=list)
    """The names of the room's hooks on the triggers the turn fires (RFC
    §10.6.6): the executor runs those it defines under them."""
    previous_turn: PreviousTurn | None = None
    """The room's previous turn of the channel; ``None`` for its first, or
    its first since the channel was detached from the room. An executor
    holding the room's state for the channel that this turn did not leave
    drops it before the turn runs (RFC §10.6.2)."""
    channel_rows: list[RoomEvent] = Field(default_factory=list)
    """The channel's own rows of the room its turn reads (its tool calls, the
    instructions it read), each once, oldest first: the owner holds the store
    (RFC §10.6.3). Empty for a turn the owner's own executor runs."""
    trace: dict[str, str] | None = None


class ErrorRecord(BaseModel):
    """An exception as it crosses: its class's name, its message, what
    classifies it, and its causes in order (RFC §10.6.8)."""

    kind: str
    message: str
    args: list[Any] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    """The error's attributes whose values JSON carries; the others stay
    on the executor."""
    causes: list[ErrorRecord] = Field(default_factory=list)
    traceback: str | None = None
    """Where the error was raised on the executor, formatted."""
    bases: list[str] = Field(default_factory=list)
    """The classes the error's class derives from below ``Exception``,
    nearest first: a side that does not rebuild the class reads the error as
    the nearest of these it rebuilds."""


class WireOutput(BaseModel):
    """A ``ChannelOutput`` without its stream."""

    responded: bool = False
    response_events: list[RoomEvent] = Field(default_factory=list)
    response_metadata: dict[str, Any] = Field(default_factory=dict)
    tasks: list[Task] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    metadata_updates: dict[str, Any] = Field(default_factory=dict)
    error: ErrorRecord | None = None


# -- Frames: executor to owner ------------------------------------------------


class Accepted(BaseModel):
    type: Literal["accepted"] = "accepted"
    executor_id: str


class Heartbeat(BaseModel):
    type: Literal["heartbeat"] = "heartbeat"


class Output(BaseModel):
    type: Literal["output"] = "output"
    output: WireOutput
    streams: bool = False


class Item(BaseModel):
    """One stream item; ``item["kind"]`` names it (see ``_codec``)."""

    type: Literal["item"] = "item"
    item: dict[str, Any]


class ToolArgs(BaseModel):
    """The arguments a call runs with once its gate fixed them."""

    type: Literal["tool_args"] = "tool_args"
    start: int
    arguments: dict[str, Any]


class ToolEnded(BaseModel):
    """A call's end, known before its end item was read."""

    type: Literal["tool_ended"] = "tool_ended"
    start: int
    end: dict[str, Any]


class Record(BaseModel):
    type: Literal["record"] = "record"
    record: dict[str, Any]


class Error(BaseModel):
    type: Literal["error"] = "error"
    error: ErrorRecord


class End(BaseModel):
    type: Literal["end"] = "end"


class InputRequest(BaseModel):
    """A tool of the turn asked a human for input (RFC §10.6.5): sent before
    ``ON_USER_INPUT_REQUIRED`` fires, so the owner can answer it at once, with
    what the tool knows of it, which the owner's host reads by its id
    (``RoomKit.input_request``). A field an older executor leaves out reads as
    the turn's own (its room, its channel) or as unknown."""

    type: Literal["input_request"] = "input_request"
    request_id: str
    tool_name: str
    arguments: dict[str, Any]
    room_id: str | None = None
    channel_id: str | None = None
    tool_call_id: str | None = None
    actor_id: str | None = None
    created_at: datetime | None = None


class InputClosed(BaseModel):
    """A request settled on the executor, where its tool waits, by other than
    the owner's answer (an answer the channel's handler took, a hook's block,
    its bound, a release)."""

    type: Literal["input_closed"] = "input_closed"
    request_id: str
    reason: str


class InputTaken(BaseModel):
    """Whether the owner's ``input_answer`` settled its request: one settled
    first where its tool waits did not take it."""

    type: Literal["input_taken"] = "input_taken"
    request_id: str
    taken: bool


class Ephemeral(BaseModel):
    """An ephemeral event the turn published (thinking, a tool call's
    composition, a plan): the owner publishes it on its realtime backend
    (RFC §10.6.3)."""

    type: Literal["ephemeral"] = "ephemeral"
    channel: str
    """The realtime channel it was published to (``room:<room_id>``)."""
    event: dict[str, Any]
    """The event, as ``EphemeralEvent.to_dict()`` gives it."""


class FrameworkEventFrame(BaseModel):
    """A framework event the turn emitted (RFC §8.2): the owner emits it on
    its own bus (RFC §10.6.3)."""

    type: Literal["framework_event"] = "framework_event"
    event_type: str
    room_id: str | None = None
    channel_id: str | None = None
    event_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class SteerResult(BaseModel):
    """How many loops a ``steer`` reached; ``None`` for a channel without
    steering, whose owner then cuts the turn on a ``Cancel``."""

    type: Literal["steer_result"] = "steer_result"
    steer_id: str
    loops: int | None


# -- Calls: executor to owner, each answered once by its result --------------


class CallContext(BaseModel):
    """What a call's handler reads of the turn it serves (RFC §10.6.5, §21.4).

    The owner reads the room and the channel of the call from the request it
    submitted; a call that names others is refused. An intelligence turn
    serves no voice session, so none is carried.
    """

    tool_call_id: str
    room_id: str
    channel_id: str
    chain_depth: int = 0
    actor_id: str | None = None
    allowed_names: list[str] | None = None
    """The names the turn admits; ``None`` before its toolset was resolved."""
    requester: str | None = None


class ToolCall(BaseModel):
    """A call to a tool the owner serves: its handler runs there."""

    type: Literal["tool_call"] = "tool_call"
    call_id: str
    """Unique on the link: two calls under one model id stay two calls."""
    name: str
    arguments: dict[str, Any]
    """The arguments the call's gate left."""
    context: CallContext


class HostCall(BaseModel):
    """A verb of the turn's ``TurnHost`` (``roomkit.turns.host.VERBS``), run
    on the owner with its arguments as data."""

    type: Literal["host_call"] = "host_call"
    call_id: str
    verb: str
    arguments: dict[str, Any]
    context: CallContext
    """What the verb reads of the turn: its chain depth, who asked; a call
    made from a hook names no tool call."""


class CallCancel(BaseModel):
    """The executor withdraws a call (a ``tool_call`` or a ``host_call``):
    its turn was cut or its bound expired."""

    type: Literal["call_cancel"] = "call_cancel"
    call_id: str


# -- Control: owner to executor -----------------------------------------------


class Pull(BaseModel):
    """Grant the reading of the stream's next *count* items, the owner's
    reader having read *read* of them (a joined burst counting its deltas):
    a pull sent before the reader read an item that is no delta grants
    nothing past it (RFC §10.6.4)."""

    type: Literal["pull"] = "pull"
    count: int = Field(default=1, ge=1)
    read: int = Field(default=0, ge=0)


class Close(BaseModel):
    type: Literal["close"] = "close"


class Cancel(BaseModel):
    type: Literal["cancel"] = "cancel"
    reason: str


class Steer(BaseModel):
    """A steering directive for the turn's loop (RFC §21.3)."""

    type: Literal["steer"] = "steer"
    steer_id: str
    directive: SteeringDirective = Field(discriminator="type")


class InputAnswer(BaseModel):
    """The outcome the owner recorded for an ``input_request``."""

    type: Literal["input_answer"] = "input_answer"
    request_id: str
    outcome: Literal["answered", "rejected"]
    value: str | None = None
    reason: str | None = None


class ToolResult(BaseModel):
    """The owner's answer to one ``tool_call``."""

    type: Literal["tool_result"] = "tool_result"
    call_id: str
    outcome: Literal["served", "refused", "failed", "unserved"]
    """``refused``: the handler refused the call (``ToolRefusedError``);
    ``unserved``: nothing on the owner serves it (RFC §21.4)."""
    result: dict[str, Any] | None = None
    """The handler's answer as the model reads it, when served."""
    error: ErrorRecord | None = None
    structured_content: dict[str, Any] | None = None
    """The structured copy the handler wrote on its call (RFC §21.4)."""


class HostResult(BaseModel):
    """The owner's answer to one ``host_call``: the verb's result as data,
    or what it raised."""

    type: Literal["host_result"] = "host_result"
    call_id: str
    result: Any = None
    error: ErrorRecord | None = None


# -- Channel control: owner to executor, outside any turn ----------------------


class RoomAttached(BaseModel):
    """The channel was attached to the room (RFC §10.6.3)."""

    type: Literal["room_attached"] = "room_attached"
    channel_id: str
    room_id: str
    binding: ChannelBinding
    organization_id: str | None = None
    """The room's organization (RFC §17.2): it goes only to an executor
    serving it (RFC §10.6.7)."""


class RoomDetached(BaseModel):
    """The channel was detached from the room: its state for the room is
    dropped (RFC §10.6.3)."""

    type: Literal["room_detached"] = "room_detached"
    channel_id: str
    room_id: str
    organization_id: str | None = None
    """The room's organization (RFC §17.2): it goes only to an executor
    serving it (RFC §10.6.7)."""


class CommittedEvent(BaseModel):
    """An event committed in the room that the channel's memory is handed (a
    discussion's, RFC §19.7.5 rule 3), answered by nothing (RFC §10.6.3)."""

    type: Literal["committed_event"] = "committed_event"
    channel_id: str
    room_id: str
    event: RoomEvent
    organization_id: str | None = None
    """The room's organization (RFC §17.2): it goes only to an executor
    serving it (RFC §10.6.7)."""
    room_hooks: list[str] = Field(default_factory=list)
    """The room's hooks on executor triggers, as a turn's request names them:
    the event goes to an executor that would take the room's turns."""


class ChannelCall(_HandedOver):
    """A per-room verb of the channel (an ACP session's ``set_config_option``),
    answered by a :class:`ChannelResult` (RFC §10.6.3)."""

    type: Literal["channel_call"] = "channel_call"
    call_id: str = Field(default_factory=lambda: f"call-{uuid4().hex}")
    channel_id: str
    room_id: str
    verb: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    organization_id: str | None = None
    """The room's organization (RFC §17.2): the call goes only to an executor
    serving it (RFC §10.6.7)."""
    room_state: dict[str, Any] = Field(default_factory=dict)
    """What the room keeps for the channel, as the owner stores it: the
    channel takes it back when it holds none for the room."""
    previous_turn: PreviousTurn | None = None
    """The room's previous turn or call of the channel: the executor drops a
    state it did not leave before the verb runs, as before a turn."""


class ChannelResult(BaseModel):
    """The executor's answer to one ``channel_call``: the verb's result as
    data, or what it raised."""

    type: Literal["channel_result"] = "channel_result"
    result: Any = None
    error: ErrorRecord | None = None
    executor_id: str | None = None
    """The executor that ran the verb."""
    room_state: dict[str, Any] | None = None
    """What the room keeps for the channel once the verb ran, when it
    changed (empty: nothing); ``None`` when it did not."""


Frame = Annotated[
    Accepted
    | Heartbeat
    | Output
    | Item
    | ToolArgs
    | ToolEnded
    | Record
    | Error
    | End
    | ToolCall
    | HostCall
    | CallCancel
    | SteerResult
    | InputRequest
    | InputClosed
    | InputTaken
    | Ephemeral
    | FrameworkEventFrame,
    Field(discriminator="type"),
]
Control = Annotated[
    Pull | Close | Cancel | Heartbeat | Steer | InputAnswer | ToolResult | HostResult,
    Field(discriminator="type"),
]


class ChannelReleased(BaseModel):
    """The channel's build is stale (its host changed what it was built
    from): an executor holding it lets it go, closed once the turns running
    on it end, and builds it anew for the next one (RFC §10.6.3)."""

    type: Literal["channel_released"] = "channel_released"
    channel_id: str


ChannelControl = Annotated[
    RoomAttached | RoomDetached | CommittedEvent | ChannelCall | ChannelReleased,
    Field(discriminator="type"),
]


class ChannelControlEnvelope(BaseModel):
    message: ChannelControl


# -- Outside a turn: executor to owner ---------------------------------------


class RoomCall(BaseModel):
    """A verb of a turn's host made after its turn ended (RFC §10.6.3,
    §10.6.5), answered by a :class:`HostResult`: the turn it belongs to names
    the room. The work starts fresh once its turn is over (RFC §8.3): what
    it sends is at the depth its arguments name."""

    type: Literal["room_call"] = "room_call"
    call_id: str = Field(default_factory=lambda: f"call-{uuid4().hex}")
    channel_id: str
    room_id: str
    executor_id: str
    turn_id: str
    """The turn the work making the call began in."""
    chain_depth: int = 0
    """Read by no owner: kept so that an executor sending it is still
    understood. What the call sends is at the depth its arguments name."""
    verb: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class RoomPublication(BaseModel):
    """A realtime publication or a framework event made after its turn ended
    (RFC §10.6.3), answered by nothing."""

    type: Literal["room_publication"] = "room_publication"
    channel_id: str
    room_id: str
    executor_id: str
    turn_id: str
    frame: Annotated[Ephemeral | FrameworkEventFrame, Field(discriminator="type")]


OwnerMessage = Annotated[RoomCall | RoomPublication, Field(discriminator="type")]


class OwnerMessageEnvelope(BaseModel):
    message: OwnerMessage


class FrameEnvelope(BaseModel):
    turn_id: str
    seq: int
    message: Frame


class ControlEnvelope(BaseModel):
    turn_id: str
    seq: int
    message: Control
