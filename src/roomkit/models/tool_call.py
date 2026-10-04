"""Unified tool call event for all channel types."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from roomkit.models.enums import ChannelType
from roomkit.models.streaming import LoopEndReason

if TYPE_CHECKING:
    from roomkit.providers.ai.base import AIContext, AITool, AIToolCall, AIToolResultPart
    from roomkit.voice.base import VoiceSession

logger = logging.getLogger("roomkit.hooks")


def _utcnow() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(UTC)


@dataclass(frozen=True)
class ToolCallEvent:
    """Channel-agnostic tool call event.

    Fired through ON_TOOL_CALL hooks from both AIChannel and
    RealtimeVoiceChannel.  When ``result`` is None the hook is
    expected to provide a result; when set, the hook observes
    (and may override) the handler's result.

    A call that failed or was refused fires the hook too, with
    :attr:`is_error` set. That firing is **observational**: the hook's
    override is discarded, because nothing ran and a refusal a hook could
    rewrite into a result would not be a refusal.
    """

    channel_id: str
    """ID of the channel that triggered the tool call."""

    channel_type: ChannelType
    """Type of the originating channel."""

    tool_call_id: str
    """Provider-assigned ID for this tool call."""

    name: str
    """The function name being called."""

    arguments: dict[str, Any]
    """Parsed arguments for the function call."""

    result: str | list[Any] | None = None
    """Handler result (None = hook must provide).

    Usually a string; a list of content parts when a tool returns multimodal
    output (e.g. an image). Typed ``list[Any]`` to avoid a models→providers
    import cycle — the concrete part types live in ``providers.ai.base``.
    """

    room_id: str | None = None
    """Room where the tool call originated."""

    session: VoiceSession | None = None
    """Voice session (realtime channels only)."""

    timestamp: datetime = field(default_factory=_utcnow)
    """When the tool call was received."""

    is_error: bool = False
    """Whether the call failed or was refused — the producer's own verdict.

    ``result`` alone cannot answer it. A refusal is a body like any other:
    roomkit's own refusals are JSON error envelopes, a handler that raised
    leaves ``{"error": "Tool 'x' failed (<class>)"}``, and a failed external
    tool leaves whatever the
    provider printed. A consumer reading the body can only guess, and guessing
    reads a refusal as a completed call — which is how an audit trail ends up
    recording ``ok`` for a tool that never ran.

    True means nothing usable came back: a pre-execution refusal (undeclared
    tool, invalid arguments, denied by policy, gated behind a skill, denied by
    ``BEFORE_TOOL_USE``), a handler that raised, a call nothing served, or an
    external provider reporting its own failure (``is_error`` on
    :meth:`~roomkit.tools.external.ExternalToolHandler.on_tool_result`).

    The body stays verbatim in :attr:`result` — it is what the model is told,
    and its wording is tuned for that reader; an oversized one reaches the
    model through the same eviction as any result. This flag carries the one
    thing prose cannot: that the call did not succeed.
    """

    cancelled: bool = False
    """Whether the call was abandoned before its result was read.

    A realtime model interrupted while a call is outstanding may discard it
    (Gemini Live's ``tool_call_cancellation``), and a provider abandons the
    calls a reconnect orphans, its own wait on a result times out on, or a
    lost connection takes with it: the handler is interrupted, its result is
    never sent, and the call ends with no outcome the model ever saw. That is
    neither a refusal nor a failure, and an audit counting refusals must not
    count it as one, so it travels as its own marker beside :attr:`is_error`
    (``True`` too: nothing usable came back). The firing is observational,
    like a refusal's; :attr:`result` carries a short envelope saying so.
    """

    structured_content: dict[str, Any] | None = None
    """The call's structured copy, carried beside :attr:`result` on its
    tool-call event for UI surfaces (MCP ``structuredContent``); never read by
    the model.

    A SYNC hook replaces it with ``HookResult(metadata={"structured_content":
    ...})``, and ``None`` there clears it, or with a MODIFY whose event carries
    a new one; the next hook sees the replacement (RFC §9.3). A hook that replaces only the result
    keeps it (rewriting text is not withholding the payload), and a BLOCK drops
    it: the event of a withheld call must not publish what was withheld.
    """

    error_detail: str | None = None
    """What failed, in full, when the handler or a hook failed (RFC §9.3).

    A raised call's exception class and message, what the ON_TOOL_CALL hooks
    that failed said, or the error of a BEFORE_TOOL_USE hook that failed
    closed and so refused the call, for logs and observers only. The model reads
    :attr:`result`, the failure without the message: that message can hold
    anything the failing code held, a connection string with its password
    included. ``None`` for every other outcome.
    """

    refused: bool = False
    """Whether a gate or the handler refused the call before it ran (RFC §9.3).

    A pre-execution refusal (undeclared tool, invalid or cut arguments,
    denied by policy, gated behind a skill, denied by ``BEFORE_TOOL_USE``), a
    handler's refusal (:meth:`~roomkit.tools.external.ExternalToolHandler.on_tool_refused`,
    a rejected ACP permission). ``True`` with :attr:`is_error`, and apart
    from a failure (a handler that raised) or a cancellation, so an audit
    counts each. Like a cancellation, a refusal reaches ON_TOOL_CALL's
    observers only, on every door: a SYNC hook could serve the call, and a
    refused call is not to be served.
    """
    refused_but_ran: bool = False
    """Whether the call ran although RoomKit refused it: an ACP agent that
    executed a call whose permission was rejected and closed it completed.
    Reported as it ended, served (:attr:`refused` and :attr:`is_error`
    false), and marked so an audit sees the refusal the agent went past
    (RFC §9.3)."""


@dataclass(frozen=True)
class ToolCallVerdict:
    """What ON_TOOL_CALL's SYNC hooks decided about a call that was served.

    :attr:`result` replaces the tool's result when not ``None``.
    :attr:`blocked` withholds it: the model reads :attr:`result`, the block's
    reason, as a failed call, and the call keeps no structured copy.
    :attr:`structured_content` replaces the structured copy when
    :attr:`replaces_structured` is set, ``None`` then clearing it.
    """

    result: str | list[Any] | None = None
    blocked: bool = False
    replaces_structured: bool = False
    structured_content: dict[str, Any] | None = None
    error_detail: str | None = None
    """For a call nothing served: what the hooks that failed said, for the
    observers only (RFC §9.3)."""


def fold_tool_call_rewrite(event: Any, metadata: dict[str, Any]) -> Any:
    """An ON_TOOL_CALL hook's override, written into the event it leaves.

    The hook engine's ``fold`` for ON_TOOL_CALL (RFC §9.3): the next SYNC
    hook and the channel then read the outcome as the chain left it, whether
    a hook replaced it with ``modify`` or through ``metadata``, and the ASYNC
    observers read it as the model does. ``metadata["result"]`` replaces the
    result; ``metadata["structured_content"]`` replaces the structured copy
    (:func:`renderable_copy`).
    """
    if not isinstance(event, ToolCallEvent):
        return event
    changes: dict[str, Any] = {}
    if "result" in metadata:
        changes["result"] = metadata["result"]
    if "structured_content" in metadata:
        changes["structured_content"] = renderable_copy(metadata["structured_content"])
    return replace(event, **changes) if changes else event


def tool_call_chain_fold(event: ToolCallEvent) -> Callable[[Any, dict[str, Any]], Any]:
    """The ``fold`` for one ON_TOOL_CALL chain on *event* (RFC §9.3).

    :func:`fold_tool_call_rewrite`, keeping track of whether the call stands
    served: a served call that a hook empties stays served, its result JSON
    ``null``. ``None`` on the event means nothing served the call, and the
    next hook would otherwise serve it in place of the empty result the
    previous one decided.
    """
    served = event.result is not None

    def fold(latest: Any, metadata: dict[str, Any]) -> Any:
        nonlocal served
        folded = fold_tool_call_rewrite(latest, metadata)
        if not isinstance(folded, ToolCallEvent):
            return folded
        if served and folded.result is None:
            folded = replace(folded, result="null")
        served = folded.result is not None
        return folded

    return fold


def renderable_copy(copy: Any) -> dict[str, Any] | None:
    """*copy* as a structured copy a surface can render: a mapping, or ``None``.

    A hook's copy that is not a mapping is no copy a surface can render, so it
    clears the copy rather than publish the original.
    """
    if copy is None:
        return None
    if not isinstance(copy, Mapping):
        logger.warning(
            "ON_TOOL_CALL hook left a structured_content of type %s, not a mapping; "
            "the call's structured copy is dropped",
            type(copy).__name__,
        )
        return None
    return dict(copy)


def chained_call_event(hook_result: Any, event: ToolCallEvent) -> ToolCallEvent:
    """The call as ON_TOOL_CALL's SYNC chain left it: *event* when no hook replaced it."""
    chained = hook_result.event
    return chained if isinstance(chained, ToolCallEvent) else event


def withheld_call_event(event: ToolCallEvent, reason: str) -> ToolCallEvent:
    """What ON_TOOL_CALL's observers see of a call a SYNC hook withheld: the
    failure, with the reason the model reads, and no structured copy."""
    return replace(event, result=reason, is_error=True, structured_content=None)


def observed_call_event(hook_result: Any, event: ToolCallEvent, read: Any) -> ToolCallEvent:
    """What ON_TOOL_CALL's observers see of a served call, *read* being what the model reads.

    The event the SYNC chain left carrying that result, or the failure when a
    hook withheld it (RFC §9.3).
    """
    if not hook_result.allowed:
        return withheld_call_event(event, read)
    return replace(chained_call_event(hook_result, event), result=read)


# Callback type injected into AIChannel by the framework: the hooks' verdict,
# a bare result (str or content parts) to override, or None to keep the
# original.
ToolCallCallback = Callable[..., Awaitable[ToolCallVerdict | str | list[Any] | None]]
"""``(event, *, claim=None)``: *claim* claims the call's one report between
the chain and its observers (RFC §9.3)."""


# Callback type injected into AIChannel by the framework for a call that failed
# or was refused. Returns nothing: it reaches the ASYNC observers of
# ON_TOOL_CALL only, never a hook that could serve the call (see
# ``HookEngine.run_observers``).
ToolCallObserver = Callable[[ToolCallEvent], Awaitable[None]]


RESPONSE_SEGMENT_SEPARATOR = "\n\n"
"""What separates two segments of a turn in :attr:`AIResponseEvent.response_content`.

A tool call cuts the model's text: what it said before the call and what it
said after are two segments, persisted as two MESSAGE events. Joined with
nothing between them they read as one run-on sentence (``first.Working``);
this is the paragraph break that keeps them apart. It only ever sits between
two segments — never inside one, never at either end.
"""


def response_transcript(segments: Iterable[str]) -> tuple[list[str], str]:
    """The turn's text as ``ON_AI_RESPONSE`` reports it.

    Drops the empty stretches (a tool round in which the model said nothing)
    and returns the segments kept, and their join. The one place the contract
    lives: the AI channel's tool loop and the ACP channel both report through it.
    """
    kept = [segment for segment in segments if segment]
    return kept, RESPONSE_SEGMENT_SEPARATOR.join(kept)


ToolDeclarationOrigin = Literal["always", "pinned", "sticky", "revealed"]
"""Why a tool was in the toolset the provider received (:class:`DeclaredTool`).

``"always"`` is a tool Tool Search never hid: every tool of a turn that ran
without it, and, under it, a discovery or skill infrastructure tool, a tool
orchestration injected (a handoff, a delegation, a result tool; RFC §21.1)
and anything a hook added. The other three name which term of the Tool Search
keep-set admitted a catalogue tool: ``"pinned"`` by the channel's
``tool_search_pinned`` configuration, ``"sticky"`` because the room already
called or found it in an earlier turn, ``"revealed"`` because ``find_tools``
found it in this turn. A tool that qualifies on several counts carries the
earliest reason it was visible, in that order.
"""


@dataclass(frozen=True)
class DeclaredTool:
    """One tool as the provider received it, and why it was there.

    The record :attr:`AIResponseEvent.declared_tools` is made of. ``name``,
    ``description`` and ``parameters`` are the ``AITool`` fields RoomKit handed
    the provider: its declaration, not the provider's wire form of it. The
    schema is that object, not a copy, so a consumer reads it and never
    writes it.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    origin: ToolDeclarationOrigin = "always"

    @classmethod
    def from_tool(cls, tool: AITool, origin: ToolDeclarationOrigin = "always") -> DeclaredTool:
        """The record for *tool*, declared for the reason *origin* says."""
        return cls(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
            origin=origin,
        )


@dataclass(frozen=True)
class AIResponseEvent:
    """Emitted through ON_AI_RESPONSE hooks after AI generation completes.

    Provides response content, usage metrics, and timing for evaluation
    and scoring integrations.
    """

    channel_id: str
    """ID of the AI channel that generated the response."""

    response_content: str
    """Everything the model said in the turn, as one readable transcript.

    The turn's :attr:`segments` joined with :data:`RESPONSE_SEGMENT_SEPARATOR`
    — a blank line at every tool-call boundary, nothing inside a segment. A
    turn without a tool call is its single segment, verbatim.
    """

    room_id: str | None = None
    """Room where the response was generated."""

    tool_calls_count: int = 0
    """Number of tool calls executed during generation."""

    usage: dict[str, Any] = field(default_factory=dict)
    """Token usage from the provider (input_tokens, output_tokens)."""

    thinking: str = ""
    """Extended thinking/reasoning text (if supported by provider)."""

    round_count: int = 0
    """Number of tool execution rounds."""

    loop_end_reason: LoopEndReason | str | None = None
    """Which of the tool loop's rules ended the turn, or None if unreported.

    The loop names it on its :class:`LoopEndMarker`, and this event carries
    it: without it a hook could see *that* a turn ended and how much work it did, never
    whether it finished or was cut off. Counting tool calls does not answer it
    — :attr:`tool_calls_count` reports the calls the turn *ran*, so a healthy
    multi-round answer and one guillotined by the round cap both report a
    positive count. Read this instead: ``"completed"`` is a turn that ended on
    its own terms, and ``"max_rounds"``, ``"timeout"``, ``"budget_exceeded"``, ``"cancelled"``,
    ``"force_stopped"``, ``"truncated"``, ``"empty_response"``, ``"unfinished"``
    and ``"error"`` each name the rule that stopped it (``"unfinished"``: the
    channel's continuation policy still asked to go on once its tries had run
    out; ``"error"``: the provider interrupted the turn after a tool round and
    the turn was delivered once its loop ended, RFC §6.4). An ACP agent's
    turn carries its stop reason: ``"completed"`` for ``end_turn``, else the
    agent's own (``"max_tokens"``, ``"max_turn_requests"``, ``"refusal"``,
    ``"cancelled"``).

    A turn whose loop did not reach its end fires no event at all: one that
    raised (a streamed turn the provider interrupted included) or whose
    stream was closed first (a barge-in, a transport that stopped reading, a
    task cancelled from outside). Its ``llm.generate`` span carries what its
    rounds used.

    None means the path that fired the hook reported no reason, not that the
    turn completed.
    """

    latency_ms: int = 0
    """Total generation time in milliseconds."""

    streaming: bool = False
    """Whether the response was streamed."""

    timestamp: datetime = field(default_factory=_utcnow)
    """When the response was generated."""

    segments: list[str] = field(default_factory=list)
    """The turn's text, one entry per stretch between tool calls, in order.

    Empty stretches are dropped, so :attr:`response_content` is exactly these
    joined with :data:`RESPONSE_SEGMENT_SEPARATOR`, and ``segments[-1]`` is
    the text that followed the last tool call — the answer, for a consumer
    that wants it without the narration before it. Empty when the turn
    produced no text.
    """

    usage_metadata: dict[str, Any] = field(default_factory=dict)
    """Optional provenance and scope of the provider's usage observations.

    ACP includes the native session, prompt source and session usage report.
    These are observations, not a price or a claim that generation succeeded.
    Existing providers and consumers can leave this empty.
    """

    declared_tools: list[DeclaredTool] = field(default_factory=list)
    """The tools the provider received this turn, over every generation round.

    ``BEFORE_AI_GENERATION`` sees the turn's toolset once, as the turn starts:
    under Tool Search the whole catalogue, not what any round declares. A tool
    ``find_tools`` reveals only enters the declaration of the *next* round, so
    a host recording "what the model was offered" reads it here. This is the union of
    the toolsets handed to the provider on each round of the turn, in first
    declaration order, one entry per name: a tool declared on several rounds
    keeps its first entry, and since the reveal window slides from one
    ``find_tools`` to the next, the union is the only reading that keeps a
    tool revealed earlier in the turn. A turn without Tool Search reports its
    one declaration here too, every entry ``"always"``: one reading for the
    host, whatever the turn's mode.

    Empty when the turn declared no tool, and for a channel whose toolset is
    not RoomKit's to declare (an external ACP agent).
    """


# Callback type for AI response observation (fire-and-forget).
AfterResponseCallback = Callable[["AIResponseEvent"], Awaitable[None]]


@dataclass
class AIGenerationEvent:
    """Emitted through BEFORE_AI_GENERATION hooks before AI provider invocation.

    Provides the full AI context for inspection and modification.
    Hooks can mutate ``ai_context`` in-place (e.g. append messages,
    modify system_prompt, adjust tools) and return ``HookResult.allow()``,
    or return ``HookResult.block(reason)`` to prevent generation.
    """

    ai_context: AIContext
    """The context the turn starts from; what the hook leaves is the turn's.

    Its ``tools`` are the turn's toolset after the tool policy and skill
    gating, under Tool Search the whole catalogue: each round declares Tool
    Search's collapse of what the hook left (RFC §6.4).
    """

    channel_id: str
    """ID of the AI channel about to generate."""

    room_id: str | None = None
    """Room where generation is happening."""

    provider_name: str | None = None
    """Name of the AI provider that will be invoked."""

    timestamp: datetime = field(default_factory=_utcnow)
    """When the generation was initiated."""


@dataclass
class ToolRoundEvent:
    """Emitted through AFTER_TOOL_ROUND between two rounds of an AI channel's
    tool loop, once the channel ran a round's calls (RFC §6.4).

    The round has run: a hook reads it whole (its calls run concurrently, so a
    rule about them is a rule about the round) and acts on the rest of the
    turn through :meth:`withdraw` and :meth:`add_message`.
    """

    channel_id: str
    """ID of the AI channel whose loop ran the round."""

    room_id: str | None
    """Room the turn runs in."""

    round_index: int
    """Index of the round in the turn, from 0."""

    calls: list[AIToolCall]
    """The calls the channel ran, in the order the model made them."""

    results: list[AIToolResultPart]
    """The channel's result of each call, in the same order."""

    answered: list[AIToolResultPart] = field(default_factory=list)
    """The results of the round's calls the provider served itself."""

    tools: list[str] = field(default_factory=list)
    """Names of the turn's toolset the next round is built from, as
    ``BEFORE_AI_GENERATION`` sees it: what the tool policy and skill gating
    let the turn reach, Tool Search's whole catalogue included, less what was
    withdrawn. :meth:`withdraw` takes any name, listed here or not (an
    external handler's tool)."""

    withdrawn: set[str] = field(default_factory=set)
    """Names withdrawn for the rest of the turn (see :meth:`withdraw`)."""

    messages: list[str] = field(default_factory=list)
    """Texts the next round reads after this round's results (see :meth:`add_message`)."""

    def withdraw(self, *names: str) -> None:
        """Take *names* out of the rest of the turn, with every guarantee of a
        withdrawal by BEFORE_AI_GENERATION: never declared again, a call naming
        one refused (a tool the channel provides itself included)."""
        self.withdrawn.update(names)

    def add_message(self, text: str) -> None:
        """Have the next round read *text*, as a user message after the results."""
        self.messages.append(text)


ContinuationPolicy = Callable[[str], "str | None"]
"""An AI channel's continuation policy (RFC §6.4): given the text a round ended
on naturally, without a call, the instruction that makes the model go on, or
``None`` when the answer stands (a recognizer of an announced action, say)."""


# Callback type for BEFORE_AI_GENERATION hook (sync, can block/modify).
# Returns SyncPipelineResult (from roomkit.core.hooks) — typed as Any to
# avoid circular import from models into core.  Only the framework's
# _build_before_generation_hook closure creates instances of this type.
BeforeGenerationCallback = Callable[["AIGenerationEvent"], Awaitable[Any]]
