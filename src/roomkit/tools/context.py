"""Per-call tool execution context, and the public accessors for it.

An ``AIChannel`` object is registered once per ``channel_id`` and shared
by every room it serves, so any room-specific state stored on the channel
(or on a host's tool handler) goes stale the moment another room attaches.
The tool loop scopes its per-invocation state in a contextvar instead;
this module holds that contextvar and exposes the parts of the state a
host's tool handler may need to resolve the call's origin.

Contextvars propagate through the async call chain, so a handler invoked
from inside a tool loop sees the loop's context without any signature
change. The realtime voice channel installs the same context around each
tool call it serves, with the session's room and participant as the turn's
room and actor, so a handler shared between an ``AIChannel`` and a
``RealtimeVoiceChannel`` reads the room id, the Room, the actor and the
call's record on both paths, and :func:`current_tool_allowed_names` answers
the tools the session declares (``None`` when it declares no catalogue);
:func:`current_response_metadata` alone answers ``None`` there, since no turn
merges that record. Outside a tool call (a direct call) every accessor
returns ``None`` — hosts keep their own fallback there. A test that calls a handler
directly describes the turn it runs under with :func:`tool_turn_context`.
"""

from __future__ import annotations

import asyncio
import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit.models.response_metadata import ResponseMetadata
from roomkit.models.room import Room
from roomkit.tools._turn_calls import TurnCalls

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

    from roomkit.channels._turn_budget import TurnBudget
    from roomkit.models.steering import SteeringDirective
    from roomkit.models.tool_call import DeclaredTool
    from roomkit.providers.ai.base import AIMessage, AITool


@dataclass
class ToolCallContext:
    """Contextvar payload carrying tool-call metadata.

    The ToolHandler protocol is ``(name, arguments) → result`` — it does not
    receive ``room_id``, ``tool_call_id`` or ``channel_id``.  This payload
    bridges the gap: ``_ai_tools._run_one()`` sets it before calling the
    handler, and a handler that needs the call's origin reads it.  Safe
    with :func:`asyncio.gather`, which creates Tasks with copied contexts.

    ``structured_content`` is the reverse channel: the ToolHandler contract
    returns only text or content parts, but MCP tools can produce a structured result
    (``CallToolResult.structuredContent``) that UI surfaces need as it is —
    the LLM-facing string may be truncated/evicted when large. A handler
    that has one sets it here; ``_run_one()`` reads it back after the call
    and carries it on the tool-call events untouched by eviction, its binary
    payloads bounded like the result's.
    """

    room_id: str = ""
    tool_call_id: str = ""
    channel_id: str = ""
    structured_content: dict[str, Any] | None = None


_current_tool_call: contextvars.ContextVar[ToolCallContext | None] = contextvars.ContextVar(
    "_current_tool_call", default=None
)


@dataclass(frozen=True)
class TurnFootprint:
    """What a turn takes of the context window besides its history (RFC §20).

    Measured by the AI channel as round 0 will send it, before it reads its
    memory. What a hook adds to the context afterwards is not in it.

    Attributes:
        input_tokens: The system prompt, the tools declared (Tool Search's
            collapse and the re-read tool included) and the notes the channel
            adds to the turn's input (the room's plan, the digest of the tools
            already used, the speaker attribution).
        reply_tokens: The reply budget the turn requests; 0 when the channel
            sets none and the provider applies its own.
    """

    input_tokens: int
    reply_tokens: int


@dataclass
class _ToolLoopContext:
    """Per-invocation state for a tool loop, scoped via contextvar."""

    activated_skills: set[str] = field(default_factory=set)
    # Tool Search: names revealed by ``find_tools`` this loop. Accrues during
    # the loop (NOT inherited across for_loop) exactly like ``activated_skills``
    # — round 0 starts empty, a find_tools call reveals matches, and the next
    # round's tool re-filter shows them.
    revealed_tools: set[str] = field(default_factory=set)
    # The tool rounds this loop ran, and the one whose served search last swapped
    # the reveal window: searches of one round add up, a later round's swaps it.
    tool_rounds: int = 0
    revealed_in_round: int | None = None
    # Anti-loop guard: count of identical (tool, canonical-args) calls this
    # turn. Read by ``_repeated_call_guard`` to short-circuit a model stuck
    # re-issuing the same call instead of answering.
    repeated_calls: dict[tuple[str, str], int] = field(default_factory=dict)
    # The same guard's other axis: count of identical (tool, result-hash)
    # RESULTS this turn. ``repeated_calls`` keys on the ARGUMENTS, so a model
    # that permutes them walks straight past it while learning nothing —
    # measured on a stuck turn at 44 distinct argument sets for 25 distinct
    # results, one of which came back 23 times. Read by
    # ``_repeated_result_note`` to tell the model the thing it cannot see:
    # this answer already arrived.
    repeated_results: dict[tuple[str, str], int] = field(default_factory=dict)
    # Set by ``_repeated_call_guard`` once a model keeps re-issuing an
    # identical blocked call (it ignores the advisory error). The tool loop
    # reads it and force-ends the turn with a plain-text answer instead of
    # letting the model hammer the same call to the round limit.
    force_stop: bool = False
    # Tools the agent already CALLED — or that find_tools already REVEALED —
    # this conversation, seeded once per turn in ``_build_context`` from
    # ToolUsageMemory. Unlike ``revealed_tools`` (per-loop find_tools swap
    # window, starts empty), these are INHERITED across for_loop and unioned
    # into the Tool Search keep-set by ``_apply_tool_filters`` — so a tool
    # used or found once stays callable without re-running find_tools.
    sticky_tools: set[str] = field(default_factory=set)
    # ``None`` means context construction has not run. An empty list is a
    # completed, deny-all toolset and must remain distinguishable from it.
    all_context_tools: list[Any] | None = None
    # Names withdrawn for the rest of the turn (``withdraw``).
    # ``all_context_tools`` has already lost them; this keeps the channel's
    # per-round injections (the eviction re-read) from bringing one back.
    # Inherited across for_loop like the toolset it amends.
    withdrawn_tools: frozenset[str] = frozenset()
    # Tools of a driven turn's catalogue its model is not offered, each with
    # the refusal its call reads, as the gate that drives the turn words it (a
    # reasoning backend's voice session, RFC §21.1). Inherited like the above.
    unavailable_tools: Mapping[str, str] = field(default_factory=dict)
    # Names BEFORE_AI_GENERATION added: declared at every round of the turn,
    # never deferred by Tool Search (RFC §6.4). Inherited like the above.
    hook_pinned: frozenset[str] = frozenset()
    # The tools the turn's first round showed the model, when its provider
    # holds the others unseen: the room's kept declaration
    # (``_open_turn_declaration``), fixed for the loop, so a reveal or a skill
    # activation references a tool instead of declaring it.
    first_shown: frozenset[str] | None = None
    # A standalone turn (RFC §10.1.1), which reads none of the room's working
    # state, the room's kept declaration included. Inherited like the above.
    standalone: bool = False
    # Held tools a result has referenced this loop, so each is referenced once.
    referenced: set[str] = field(default_factory=set)
    # The turn's input, notes included, as _build_context and then the
    # BEFORE_AI_GENERATION hook left it: what an emergency compaction keeps
    # whole (RFC §6.4), found among the messages by identity.
    turn_input: AIMessage | None = None
    # What the turn may spend, resolved with its other settings (RFC §6.4).
    turn_budget: TurnBudget | None = None
    # ``activate_skill`` calls whose activation waits for the call's outcome,
    # by tool_call_id: committed once the call is served, dropped when
    # ON_TOOL_CALL blocks it or it fails, so a refused activation opens no gate.
    pending_activations: dict[str, str] = field(default_factory=dict)
    # The tools a ``find_tools`` call matched or an ``activate_skill`` hint
    # named, by tool_call_id: revealed once the call is served, as an
    # activation is committed.
    pending_reveals: dict[str, list[str]] = field(default_factory=dict)
    # The catalogue tool a call named without ``find_tools``, by tool_call_id:
    # revealed once the tool answered the call (the room's tool memory then
    # keeps it for later turns); a call refused before it ran, or that
    # nothing served, reveals nothing (RFC §6.4).
    pending_recoveries: dict[str, str] = field(default_factory=dict)
    # The tools served recoveries revealed this loop: used, so a find_tools
    # swap of the reveal window keeps them.
    recovered_tools: set[str] = field(default_factory=set)
    # Whether Tool Search is active for this turn (catalogue over threshold).
    # Decided once in ``_build_context`` and read by ``_apply_tool_filters`` on
    # every round, so it is inherited across for_loop like ``all_context_tools``.
    tool_search_active: bool = False
    current_participant_role: str | None = None
    # Participant id of whoever's turn this is — the author of the event that
    # woke the channel. A channel object is registered once per channel_id and
    # shared by every room it serves, so anything a tool handler wants to know
    # about *this* turn has to ride the contextvar: the room does
    # (``room_id``), and so must the person, or a handler acting "for the
    # user" acts for whoever the channel was built with. ``None`` when the
    # event carries no participant (a system injection, a webhook) — a caller
    # must then decide for itself rather than assume the last speaker. It
    # names the turn without authenticating it: what the id is worth is the
    # participant's ``identification``, which is why
    # ``current_tool_actor_id()`` documents the resolution a host owes it.
    actor_id: str | None = None
    room_id: str | None = None
    # The channel whose loop runs the turn: the calls it announced are its
    # own, claimed by its reports alone (RFC §9.3).
    channel_id: str | None = None
    # The chain depth of the response this turn produces (RFC §8.3, §21.4):
    # a result delivered later on the turn's behalf, a background
    # delegation's, inherits it, so a cycle of delegations ends at
    # ``max_chain_depth`` like any chain. 0 outside a turn.
    chain_depth: int = 0
    # The Room of the turn, as ``on_event`` received it in its ``RoomContext``:
    # the room as the store loaded it when the turn began, carried by
    # reference so a tool handler reads the same object the turn's hooks,
    # memory provider and config provider hold, instead of re-reading it by
    # ``room_id``. A patch written to the store during the turn is not in
    # it, and a handler must not write on it (``current_tool_room`` says
    # why). ``None`` for a loop started without a turn above it.
    room: Room | None = None
    # Whether a turn above this context merges ``response_metadata`` into the
    # MESSAGE events it produces. False for the context the realtime voice
    # channel builds around a tool call: no turn runs there, so
    # ``current_response_metadata()`` answers ``None`` rather than a record
    # nothing will carry.
    has_turn: bool = True
    steering_queue: asyncio.Queue[SteeringDirective] = field(default_factory=asyncio.Queue)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    # Set by the channel's close (RFC §9.3): the calls the turn runs are
    # cancelled, each reported cancelled, and no further round is asked.
    closing: bool = False
    # The tasks running the turn's calls (its round's, an external handler's
    # pending decision), each registered once running: the close cancels them.
    cancellable: set[asyncio.Task[Any]] = field(default_factory=set)
    # Set once the turn ended and the calls it cut were reported.
    ended: asyncio.Event = field(default_factory=asyncio.Event)
    loop_id: str = ""
    # The turn's one response-metadata record (see ``roomkit.models.response_metadata``).
    # Created here, at the start of the turn, so a memory provider writing during
    # ``_build_context`` writes the same object a tool handler reaches mid-loop;
    # ``_build_context`` hands it to ``AIContext`` and ``for_loop`` inherits the
    # reference, never a copy.
    response_metadata: ResponseMetadata = field(default_factory=ResponseMetadata)
    # What the turn takes of the window besides its history, measured as the
    # channel will send it before it reads its memory (RFC §20); ``None``
    # until a context build measured it. See ``current_turn_footprint()``.
    turn_footprint: TurnFootprint | None = None
    # The tools the provider received, over every round of the turn, keyed by
    # name in first-declaration order (see ``AIResponseEvent.declared_tools``).
    # Round 0 is declared under the turn's context and later rounds under the
    # loop's child, so ``for_loop`` shares this dict by reference like
    # ``response_metadata``: whichever context recorded a round, the emission
    # reads the whole turn.
    declared_tools: dict[str, DeclaredTool] = field(default_factory=dict)
    # The calls the loop announced, each held as a call of its own with its
    # one ON_TOOL_CALL report: a call announced and never reported, whatever
    # cut it (a stop, a cancellation, a transport that stopped reading), is
    # reported cancelled when the loop ends (RFC §9.3).
    calls: TurnCalls = field(default_factory=TurnCalls)
    # Whether the turn's tool policy, resolved for its actor, admits a name;
    # ``None`` when no policy applies. Read by ``current_tool_allowed_names()``
    # (RFC §21.4): the gate refuses what it denies, so it is not callable.
    admits: Callable[[str], bool] | None = None

    def offered_tools(self) -> list[Any]:
        """Every tool the turn offers the model: its resolved toolset, then
        what a round declared beyond it (the re-read of a stored result),
        from the round that first declared it (RFC §6.4, §21.4), nothing
        withdrawn. ``AITool`` and ``DeclaredTool`` entries,
        each with its name and description."""
        base: list[Any] = list(self.all_context_tools or [])
        known = {tool.name for tool in base} | self.withdrawn_tools
        return base + [tool for name, tool in self.declared_tools.items() if name not in known]

    def withdraw(self, names: Iterable[str]) -> None:
        """Take *names* out of the rest of the turn (RFC §6.4).

        Gone from the toolset every round is built from, and remembered: the
        gate refuses a call naming one (a tool the channel provides itself
        included), an external handler is never handed it, and the channel's
        per-round injections (the eviction re-read) do not bring it back.
        """
        gone = frozenset(names)
        if self.all_context_tools is not None:
            self.all_context_tools = [t for t in self.all_context_tools if t.name not in gone]
        self.withdrawn_tools = self.withdrawn_tools | gone

    def claim_report(self, call_id: str) -> bool:
        """Claim the one report of the call *call_id* names now: ``False``
        when it was made (see :class:`TurnCalls`)."""
        return self.calls.claim(call_id)

    def was_reported(self, call_id: str) -> bool:
        """Whether the call *call_id* names now has had its one report."""
        return self.calls.was_reported(call_id)

    @contextmanager
    def cut_by_close(self) -> Iterator[None]:
        """Run the enclosed code as one of the turn's calls: the channel's
        close cancels the current task while it is inside (RFC §9.3)."""
        task = asyncio.current_task()
        if task is None:
            yield
            return
        self.cancellable.add(task)
        try:
            yield
        finally:
            self.cancellable.discard(task)

    def absorb_close_cut(self) -> bool:
        """Absorb a cancellation the channel's close made: ``True`` when it
        was the close's alone, and the task goes on, its call cancelled;
        ``False`` for a cancellation of the turn itself, to re-raise."""
        task = asyncio.current_task()
        if task is None or not self.closing:
            return False
        return task.uncancel() == 0

    @classmethod
    def for_loop(
        cls,
        parent: _ToolLoopContext | None,
        room_id: str | None,
        room: Room | None = None,
    ) -> _ToolLoopContext:
        """Create a tool-loop context inheriting per-turn state from *parent*.

        _build_context ran under the parent (handle_event) ctx and stamped the
        turn's full toolset there — without this inheritance the per-round
        tools re-application never fires (skill-gated tools would stay hidden
        after activation) and per-call allowlist accessors see nothing.

        *room* names the loop's room for a loop started without a turn; with
        a parent, the parent's is inherited by reference. The id follows the
        room whenever one is known, so ``current_tool_room_id()`` and
        ``current_tool_room().id`` cannot disagree; *room_id* stands in only
        when no room is.
        """
        ctx = cls()
        # A uuid, not id(ctx): CPython recycles object ids after gc, and the
        # _active_loops registry keyed on a recycled id could cross-target.
        ctx.loop_id = uuid4().hex
        if parent is not None:
            ctx.current_participant_role = parent.current_participant_role
            ctx.actor_id = parent.actor_id
            ctx.chain_depth = parent.chain_depth
            ctx.all_context_tools = parent.all_context_tools
            ctx.admits = parent.admits
            ctx.withdrawn_tools = parent.withdrawn_tools
            ctx.unavailable_tools = parent.unavailable_tools
            ctx.hook_pinned = parent.hook_pinned
            ctx.tool_search_active = parent.tool_search_active
            # Carry the used-tools re-exposition seeded in _build_context into the
            # loop: the per-round re-filter runs under THIS child ctx, so without
            # this the seed is dropped at round 0 and the model must re-find_tools.
            ctx.sticky_tools = set(parent.sticky_tools)
            # By reference: the loop writes into the record the turn already
            # holds, and the output built before the loop ran reads the same one.
            ctx.response_metadata = parent.response_metadata
            # By reference too: round 0 was declared under the parent, the
            # rounds below run under this child, and the turn reports one union.
            ctx.declared_tools = parent.declared_tools
            # The input _build_context gave the turn, which a compaction in
            # the loop keeps whole.
            ctx.turn_input = parent.turn_input
            ctx.standalone = parent.standalone
            ctx.turn_budget = parent.turn_budget
        ctx.room = room if room is not None else (parent.room if parent else None)
        if ctx.room is not None:
            ctx.room_id = ctx.room.id
        else:
            ctx.room_id = room_id or (parent.room_id if parent else None)
        return ctx


_current_loop_ctx: contextvars.ContextVar[_ToolLoopContext | None] = contextvars.ContextVar(
    "_current_loop_ctx", default=None
)


def current_tool_call() -> ToolCallContext | None:
    """The per-call context of the tool call the caller is executing under.

    What ``_run_one`` set before invoking the handler — the call's id, its
    room, its channel — and the reverse channel the handler may fill:
    ``structured_content``, the MCP structured result the tool-call events
    carry for UI surfaces, as it is but for binary payloads past the event's
    bound. A host that rewrites a result before the
    model reads it (a provider's private address turned into its own relay
    link, say) reaches the structured copy here, so the persisted event does
    not keep what the text no longer says.

    ``None`` outside a tool call.
    """
    return _current_tool_call.get()


def current_tool_room_id() -> str | None:
    """Room id of the tool loop the caller is executing under.

    Returns ``None`` when called outside a tool loop.
    """
    ctx = _current_loop_ctx.get()
    return ctx.room_id if ctx is not None else None


def current_tool_room() -> Room | None:
    """The :class:`~roomkit.models.room.Room` of the turn the caller is executing under.

    The room as the store loaded it when the turn began: the same object
    ``RoomContext.room`` holds for that turn's hooks, memory provider and
    config provider, so a handler deciding whom a call acts for reads the
    room's ``organization_id``, ``metadata`` or ``status`` here instead of
    re-reading the room by :func:`current_tool_room_id` on every call. On a
    realtime tool call, which runs no turn, it is the room as the store
    loaded it for that call, shared with the ``BEFORE_TOOL_USE`` gate's
    context when a hook made the channel build one. A
    patch written to the store during the turn, by this handler or another,
    is not in it; re-read the room when the turn's own writes matter. Do not
    mutate it: the object is shared with the whole turn, a room changes
    through the store, and a write on this object would be read by the rest
    of the turn (the agent-response policy, the delivery plan) as if the
    room had changed.

    Like :func:`current_tool_actor_id`, it names the turn and authenticates
    nothing. Which organization the room belongs to is a fact of the room;
    whether the caller may act for it is the host's rule, applied by the
    host.

    ``None`` outside a tool loop, and ``None`` for a loop started without a
    turn above it.
    """
    ctx = _current_loop_ctx.get()
    return ctx.room if ctx is not None else None


def current_tool_actor_id() -> str | None:
    """Participant id of whoever's turn the caller is executing under.

    The author of the event that woke the channel this round. Read it rather
    than the identity a handler captured when it was built — one channel
    object serves every room and every speaker, so a captured identity is
    whoever happened to attach it.

    It names the turn; it does not authenticate it. The value is a room
    ``Participant.id``, and the inbound pipeline only substitutes the resolved
    ``Identity.id`` for it once identification succeeds — a turn still pending,
    ambiguous or unknown carries whatever the channel supplied, or a synthetic
    ``pending-…``, and reads back just as non-``None``. A handler that reaches
    a person's data with it resolves it first: load the participant, require
    ``Participant.identification`` to be ``IDENTIFIED``, and take
    ``Participant.identity_id`` as the principal.

    The author need not be human, either. In a multi-agent room the waking
    event may be another agent's, whose participant id reads back the same
    way — compare the participant's ``role`` against ``ParticipantRole.AGENT``
    when that distinction matters.

    ``None`` outside a tool loop, and ``None`` when the turn has no
    participant behind it (a system injection, a webhook, a scheduled run).
    A caller that needs a person then decides for itself — refuse, or fall
    back to a principal it configured on purpose — rather than borrow whoever
    spoke last.
    """
    ctx = _current_loop_ctx.get()
    return ctx.actor_id if ctx is not None else None


def _current_turn_chain_depth() -> int:
    """The chain depth of the response the current turn produces; 0 outside a turn.

    What a result delivered later on the turn's behalf inherits (RFC §21.4,
    §23.3). Internal: the delegation paths read it, no host needs to.
    """
    ctx = _current_loop_ctx.get()
    return ctx.chain_depth if ctx is not None else 0


def turn_report_claim(call_id: str, channel_id: str) -> Callable[[], bool] | None:
    """The claim on call *call_id*'s one report in *channel_id*'s turn running
    now, or ``None`` when no such turn announced it (RFC §9.3)."""
    ctx = _current_loop_ctx.get()
    if ctx is None or ctx.channel_id != channel_id or ctx.calls.entry_for(call_id) is None:
        return None
    return partial(ctx.claim_report, call_id)


def current_tool_allowed_names() -> set[str] | None:
    """Names of every tool in the current turn's resolved toolset that its
    tool policy admits, with what a round declared beyond it (the re-read of
    a stored result, from the round that declared it), nothing withdrawn.

    ``_build_context`` stamps the turn's full toolset (config-provider
    result plus channel-injected tools) into the loop context; a host's
    tool handler can validate an incoming call against it instead of an
    attach-time snapshot that goes stale on shared channels. A tool the
    policy denies the turn's actor is left out, on every channel: the gate
    refuses it before any handler. Includes skill-gated tools whose
    *visibility* is filtered per round — gating is presentation, not an
    execution boundary.

    On a realtime tool call (a voice session, a conference) it is every
    tool the session declares that its policy admits: its catalogue, what
    orchestration set up, the channel's own (RFC §21.4); ``None`` when the
    session declares no catalogue: it names no list, and its gate, policy
    included, still judges each call.

    Returns ``None`` outside a tool loop or before context build, so
    hosts can fall back to their own allowlist.
    """
    ctx = _current_loop_ctx.get()
    if ctx is None or ctx.all_context_tools is None:
        return None
    admits = ctx.admits or (lambda _name: True)
    return {
        name for t in ctx.offered_tools() if (name := getattr(t, "name", None)) and admits(name)
    }


def current_turn_footprint() -> TurnFootprint | None:
    """What the turn takes of the context window besides its history (RFC §20).

    Measured by the AI channel before it reads its memory: a memory reading
    the room for a turn sizes the history to what the window leaves. Readable
    from that read on, through the turn's ``BEFORE_AI_GENERATION`` hooks.
    ``None`` outside an AI channel's turn, before the channel measured it, and
    in a tool handler, whose loop context is the call's own.
    """
    ctx = _current_loop_ctx.get()
    return ctx.turn_footprint if ctx is not None else None


def current_response_metadata() -> ResponseMetadata | None:
    """The response-metadata record of the turn the caller is executing under.

    The one mapping RoomKit merges into every MESSAGE event the turn produces
    (see :mod:`roomkit.models.response_metadata`): a memory provider writing it
    during context build, a ``BEFORE_AI_GENERATION`` hook writing
    ``event.ai_context.response_metadata``, and a tool handler or a
    ``BEFORE_TOOL_USE`` hook writing here (the hook runs under the call's turn)
    all reach the same object, the one ``InboundResult.response_metadata``
    hands the caller, the turn answered or failed. A tool handler is the case this exists for — the
    ``ToolHandler`` protocol hands it nothing but ``(name, arguments)``, and a
    document it read is a fact about the turn, not about the tool's string
    result.

    Returns ``None`` when no loop context is set (a direct call) and on a
    realtime tool call, where no turn merges the record: the guard
    ``if record is not None`` then skips a write nothing would carry. A loop
    started without a turn — no ``handle_event`` above it — carries a record
    of its own that no MESSAGE event is built from; writes to it are harmless
    and go nowhere.
    """
    ctx = _current_loop_ctx.get()
    return ctx.response_metadata if ctx is not None and ctx.has_turn else None


@contextmanager
def _installed(loop_ctx: _ToolLoopContext, call: ToolCallContext | None) -> Iterator[None]:
    """Run the enclosed code under a tool call's context, restored on the way out."""
    call_token = _current_tool_call.set(call)
    loop_token = _current_loop_ctx.set(loop_ctx)
    try:
        yield
    finally:
        _current_loop_ctx.reset(loop_token)
        _current_tool_call.reset(call_token)


def _check_one_room(room_id: str | None, room: Room | None, call: ToolCallContext | None) -> None:
    """Refuse a described turn whose arguments name more than one room."""
    if room is not None and room_id is not None and room.id != room_id:
        raise ValueError(f"room {room.id!r} and room_id {room_id!r} name different rooms")
    turn_room = room.id if room is not None else room_id
    if call is not None and call.room_id != (turn_room or ""):
        raise ValueError(
            f"the call's room_id {call.room_id!r} is not the turn's room {turn_room!r}"
        )


@contextmanager
def tool_turn_context(
    *,
    room_id: str | None = None,
    room: Room | None = None,
    actor_id: str | None = None,
    tools: Iterable[AITool] | None = None,
    chain_depth: int = 0,
    call: ToolCallContext | None = None,
) -> Iterator[None]:
    """Run the enclosed code as a tool call of a turn described by the arguments.

    What a test calling a tool handler directly needs: inside the block the
    accessors of this module answer for the turn described here, as they do
    for a call the tool loop makes, and on the way out (an exception
    included) they answer what they answered before. A handler called
    outside a tool loop otherwise reads ``None`` everywhere.

    Args:
        room_id: The turn's room id, read by :func:`current_tool_room_id`.
        room: The turn's :class:`~roomkit.models.room.Room`, read by
            :func:`current_tool_room`; its id is the turn's room id, so
            *room_id* may be left out. Both given must name the same room.
        actor_id: Whose turn it is, read by :func:`current_tool_actor_id`;
            ``None`` for a turn with no author (a system injection, a webhook).
        tools: The turn's resolved toolset, read by
            :func:`current_tool_allowed_names`. ``None`` stands for a turn
            whose toolset was not resolved (the accessor answers ``None``);
            an empty list for a resolved, empty one.
        chain_depth: The chain depth of the response the turn produces
            (RFC §8.3), which a result delivered later on its behalf inherits.
            0, the default, is what outside a turn reads (RFC §21.4); an AI
            channel's turn answering a human's message runs at 1.
        call: The per-call record :func:`current_tool_call` answers; a handler
            writing its ``structured_content`` writes this object. Its
            ``room_id`` is the turn's, as the tool loop builds it (``""`` for
            a turn without a room). ``None`` leaves the block outside any
            call record.

    The turn carries a fresh response-metadata record, which
    :func:`current_response_metadata` answers inside the block.

    Raises:
        ValueError: The arguments name more than one room: *room* and
            *room_id*, or the turn's room and *call*'s ``room_id``.
    """
    _check_one_room(room_id, room, call)
    ctx = _ToolLoopContext.for_loop(None, room_id, room)
    ctx.actor_id = actor_id
    ctx.chain_depth = chain_depth
    ctx.all_context_tools = None if tools is None else list(tools)
    with _installed(ctx, call):
        yield
