"""Hook engine for sync and async hook pipelines."""

from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import Callable, Coroutine, Iterable
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Protocol

from roomkit.core._changes import holds_change
from roomkit.core._instructions import kept_metadata
from roomkit.core._reentrance import (
    ReentryBudget,
    after_delivery,
    chain_floor,
    floors_lent_while_it_runs,
    orders_lent_while_it_runs,
    work_budget,
)
from roomkit.core.exceptions import ReentryLoopCapError
from roomkit.core.locks import lent_while_it_runs, unlocked_context
from roomkit.core.task_utils import (
    spared_by_close,
    start_named,
    under_no_held_run,
    wait_for_unspared,
)
from roomkit.core.visibility import answer_scope, narrower_scope
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import (
    ChannelType,
    EventDirection,
    EventType,
    HookExecution,
    HookTrigger,
    Visibility,
)
from roomkit.models.event import (
    MAX_COMPOSITE_DEPTH,
    DeleteContent,
    EditContent,
    EventContent,
    RoomEvent,
    holds_too_deep,
)
from roomkit.models.hook import HookResult, InjectedEvent
from roomkit.models.task import Observation, Task

logger = logging.getLogger("roomkit.hooks")

SyncHookFn = Callable[[RoomEvent, RoomContext], Coroutine[Any, Any, HookResult]]
AsyncHookFn = Callable[[RoomEvent, RoomContext], Coroutine[Any, Any, None]]

_ROUTER_MARK = "_roomkit_router"


def mark_router_hook(fn: SyncHookFn) -> SyncHookFn:
    """Mark *fn* as a conversation router's hook (RFC §19.4), which a room a
    discussion holds cannot have (RFC §19.7.5 rule 1)."""
    setattr(fn, _ROUTER_MARK, True)
    return fn


@dataclass
class HookRegistration:
    """A registered hook function.

    Attributes:
        trigger: When the hook fires (BEFORE_BROADCAST, AFTER_BROADCAST, etc.)
        execution: SYNC (can block/modify) or ASYNC (fire-and-forget)
        fn: The hook function
        priority: Lower numbers run first (default: 0)
        name: Unique within its scope, the kit's hooks or one room's (RFC
            §9.1); left empty, the engine names the hook after its function
        timeout: Max execution time in seconds (default: 30.0)
        channel_types: Only run for events from these channel types (None = all)
        channel_ids: Only run for events from these channel IDs (None = all)
        directions: Only run for events with these directions (None = all)
        event_types: Only run for events of these types (None = all)
        fail_closed: SYNC only — a timeout, an exception or an unusable result
            blocks instead of allowing (RFC §9.3). For a content check that
            must never let an unchecked payload through.
        needs_lock: BEFORE_BROADCAST SYNC only — ``False`` runs the hook off
            the room lock, before it is taken, ordered by the room's
            admission ticket (RFC §9.5.1). Only for a hook that reads the
            event, not the room's state: its context is the pre-lock one.
    """

    trigger: HookTrigger
    execution: HookExecution
    fn: SyncHookFn | AsyncHookFn
    priority: int = 0
    name: str = ""
    timeout: float = 30.0
    # Filters (None = match all)
    channel_types: set[ChannelType] | None = None
    channel_ids: set[str] | None = None
    directions: set[EventDirection] | None = None
    event_types: set[EventType] | None = None
    fail_closed: bool = False
    needs_lock: bool = True


TURN_TRIGGERS: frozenset[HookTrigger] = frozenset(
    {
        HookTrigger.BEFORE_AI_GENERATION,
        HookTrigger.ON_AI_THINKING,
        HookTrigger.ON_AI_RESPONSE,
        HookTrigger.BEFORE_TOOL_USE,
        HookTrigger.AFTER_TOOL_ROUND,
        HookTrigger.ON_TOOL_CALL,
        HookTrigger.ON_USER_INPUT_REQUIRED,
        HookTrigger.ON_SPEAK_DECISION,
        HookTrigger.ON_THOUGHT,
        HookTrigger.ON_PLAN_UPDATED,
    }
)
"""The triggers that fire on the turn executor hosting the channel, during a
turn or between turns (RFC §10.6.6); every other trigger fires on the room
owner."""

PER_TOOL_CALL_TRIGGERS: frozenset[HookTrigger] = frozenset(
    {
        HookTrigger.BEFORE_TOOL_USE,
        HookTrigger.ON_TOOL_CALL,
        HookTrigger.AFTER_TOOL_ROUND,
        HookTrigger.ON_USER_INPUT_REQUIRED,
    }
)
"""The SYNC triggers that fire per tool call or per round of a turn (a
realtime or a conference model's tool calls included, ``ON_TOOL_CALL``) or
per input request: a turn may make hundreds, so their observers do not run
as the cascade's work (RMK-978). They run as any task started from the
turn's work does: a post they make while the turn's delivery runs draws one
of that cascade's passes, a direct change (``update_event``) draws nothing.
What they send is at least as deep as the turn was when the trigger fired,
as every ASYNC hook's body is (:func:`_run_body`)."""

TYPED_TURN_TRIGGERS: frozenset[HookTrigger] = frozenset(
    {HookTrigger.BEFORE_TOOL_USE, HookTrigger.ON_TOOL_CALL, HookTrigger.ON_USER_INPUT_REQUIRED}
)
"""The turn triggers whose payload carries the agent's channel type (a tool
call, an input request): only there does a hook's ``channel_types`` filter
(RFC §9.1)."""


class TurnSide(Protocol):
    """Where a kit's engine sends a hook on a trigger a turn fires: the
    kit's own turn executor (RFC §10.6.6)."""

    def register(self, hook: HookRegistration) -> HookRegistration: ...

    def add_room_hook(self, room_id: str, hook: HookRegistration) -> HookRegistration: ...

    def remove_global_hook(self, name: str) -> bool: ...

    def remove_room_hook(self, room_id: str, name: str) -> bool: ...

    def global_hook_names(self) -> list[str]: ...

    def room_hook_names(self, room_id: str) -> list[str]: ...

    @property
    def engine(self) -> HookEngine:
        """The executor's engine, read for the hooks the kit gave it."""
        ...

    @property
    def fail_closed(self) -> frozenset[HookTrigger]: ...

    def set_fail_closed(self, triggers: frozenset[HookTrigger]) -> None: ...


def hook_decorator(
    register: Callable[[HookRegistration], object],
    trigger: HookTrigger,
    execution: HookExecution = HookExecution.SYNC,
    priority: int = 0,
    name: str = "",
    timeout: float = 30.0,
    channel_types: set[ChannelType] | None = None,
    channel_ids: set[str] | None = None,
    directions: set[EventDirection] | None = None,
    event_types: set[EventType] | None = None,
    fail_closed: bool = False,
    needs_lock: bool = True,
) -> Callable[..., Any]:
    """A decorator that registers the function it decorates through
    *register*, as a hook with these options (``RoomKit.hook``'s)."""

    def decorator(fn: SyncHookFn | AsyncHookFn) -> SyncHookFn | AsyncHookFn:
        register(
            HookRegistration(
                trigger=trigger,
                execution=execution,
                fn=fn,
                priority=priority,
                # Left empty, the engine names the hook after its function,
                # unique within its scope (RFC §9.1).
                name=name,
                timeout=timeout,
                channel_types=channel_types,
                channel_ids=channel_ids,
                directions=directions,
                event_types=event_types,
                fail_closed=fail_closed,
                needs_lock=needs_lock,
            )
        )
        return fn

    return decorator


@dataclass
class IdentityHookRegistration:
    """A registered identity hook function.

    Attributes:
        trigger: When the hook fires (ON_IDENTITY_AMBIGUOUS, ON_IDENTITY_UNKNOWN)
        fn: The hook function
        channel_types: Only run for events from these channel types (None = all)
        channel_ids: Only run for events from these channel IDs (None = all)
        directions: Only run for events with these directions (None = all)
    """

    trigger: HookTrigger
    fn: Any  # IdentityHookFn - using Any to avoid circular import
    channel_types: set[ChannelType] | None = None
    channel_ids: set[str] | None = None
    directions: set[EventDirection] | None = None


@dataclass
class SyncPipelineResult:
    """Result of running the sync hook pipeline."""

    allowed: bool = True
    event: Any = None
    reason: str | None = None
    blocked_by: str | None = None
    injected_events: list[InjectedEvent] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    hook_errors: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    failed_closed: bool = False
    """Whether a hook's failure (a raise, a timeout, an unusable result)
    blocked, as opposed to a hook's deliberate BLOCK (RFC §9.3)."""


_TURN_ROOM_HOOKS: contextvars.ContextVar[tuple[str, tuple[str, ...]] | None] = (
    contextvars.ContextVar("roomkit_turn_room_hooks", default=None)
)
"""The room of the turn the current code runs under and the names of the
room hooks its request carries (RFC §10.6.6)."""


def set_turn_room_hooks(room_id: str, names: Iterable[str]) -> None:
    """Scope the room hooks of the current context to a turn's request: its
    hooks run those names whatever the room's set becomes during it."""
    _TURN_ROOM_HOOKS.set((room_id, tuple(names)))


RESERVED_HOOK_NAMES = frozenset(
    {
        "source_read_only",
        "source_muted",
        "source_unbound",
        "event_chain_depth_limit",
        "reentry_loop_cap",
    }
)
"""The values the framework writes in ``blocked_by`` (RFC §9.1): a hook
named after one would read, when it blocks, as the framework's refusal."""

_KEPT_FIELDS = (
    "id",
    "room_id",
    "type",
    "source",
    "index",
    "chain_depth",
    "responds_to",
    "idempotency_key",
    "created_at",
    "status",
    "blocked_by",
    "consumption",
    "delivery_results",
)
"""What a BEFORE_BROADCAST replacement keeps of the event it replaces (RFC
§9.4): which event it is, never what it says or whom it asks."""


def kept_identity(
    original: RoomEvent, replacement: RoomEvent, bindings: Iterable[ChannelBinding] = ()
) -> RoomEvent:
    """*replacement* with the fields of *original* a hook may not change put
    back (RFC §9.4): its identity, and the checks already passed on it, the
    event an EDIT or a DELETE targets, its ``edit_source`` or ``delete_type``
    and its kind of content (:func:`_kept_content`), an INSTRUCTION's
    visibility, address and ``standalone`` word, and the instruction
    fingerprint the pipeline wrote on a response, or its absence. The
    visibility ``internal``, which only the framework writes (§7.3), is
    the original's whenever either holds it: a hook can neither give an
    event the framework's mark nor widen a framework record that has it.
    Where the event's answers may go is narrowed only
    (:func:`_hook_answer_scope`), a list read against the room's
    *bindings*. The hook's other changes stand."""
    kept: dict[str, Any] = {
        name: getattr(original, name)
        for name in _KEPT_FIELDS
        if name in RoomEvent.model_fields and getattr(replacement, name) != getattr(original, name)
    }
    content = _kept_content(original.content, replacement.content)
    if content != replacement.content:
        kept["content"] = content
    if original.type == EventType.INSTRUCTION:
        kept["visibility"] = original.visibility
        kept["addressed_to"] = original.addressed_to
    elif Visibility.INTERNAL in (original.visibility, replacement.visibility):
        kept["visibility"] = original.visibility
    metadata = kept_metadata(original, replacement.metadata)
    if metadata != replacement.metadata:
        kept["metadata"] = metadata
    scope = _hook_answer_scope(
        original.response_visibility, replacement.response_visibility, bindings
    )
    if scope != replacement.response_visibility:
        kept["response_visibility"] = scope
    return replacement.model_copy(update=kept) if kept else replacement


def _hook_answer_scope(
    original: str | None, written: str | None, bindings: Iterable[ChannelBinding]
) -> str | None:
    """Where an event's answers may go once a hook wrote *written* in place
    of *original* (RFC §5.12, §9.4): the hook may narrow it, never widen or
    clear it, so the result is *original* intersected with what it wrote,
    ``None`` on the hook's side keeping *original*. ``internal``, which only
    the framework writes (§7.3), reads as ``none`` from a hook."""
    if written is None:
        return original
    scope = answer_scope(written)
    if scope == Visibility.INTERNAL:
        scope = Visibility.NONE
    return narrower_scope(original, scope, bindings)


def _kept_content(original: EventContent, content: EventContent) -> EventContent:
    """What a replacement *content* keeps of the *original* it replaces
    (RFC §9.4, §10.3): the checks step 8a passed on a change, and no change
    the step never checked.

    An edit keeps its target and its ``edit_source``, and says the hook's
    new content; a delete keeps its target and its ``delete_type``, with
    the hook's reason. A replacement that would hide a change (an edit's
    new content holding one, a composite with one among its parts, a
    message made a change) leaves the original's content."""
    if isinstance(original, EditContent):
        if isinstance(content, EditContent):
            content = content.new_content
        if holds_change(content):
            return original
        # A hook that rewrote what the edit says (a redaction), as an edit or
        # as the text it says: the edit of the same event, saying that.
        return original.model_copy(update={"new_content": content})
    if isinstance(original, DeleteContent):
        if isinstance(content, DeleteContent):
            return original.model_copy(update={"reason": content.reason})
        # A delete says nothing a hook could rewrite: it stays the delete.
        return original
    return original if holds_change(content) else content


def _usable_payload(current: Any, payload: Any) -> bool:
    """Whether a MODIFY's *payload* can replace *current* (RFC §9.3): one is
    carried, of the type the trigger passed in."""
    return payload is not None and isinstance(payload, type(current))


def _holds_too_deep(hook_result: HookResult) -> bool:
    """Whether *hook_result* carries content past five composite levels, in
    its replacement or in an event it injects (RFC §5.3)."""
    payload = hook_result.event
    if isinstance(payload, RoomEvent) and holds_too_deep(payload.content):
        return True
    return any(holds_too_deep(injected.event.content) for injected in hook_result.injected_events)


def _committed_injections(
    injected: list[InjectedEvent], trigger: HookTrigger, hook: HookRegistration, room_id: str
) -> list[InjectedEvent]:
    """The events of *hook*'s result that *trigger* commits (RFC §9.4): all
    of them on BEFORE_BROADCAST, none on any other SYNC trigger, where they
    are dropped with a warning, never silently."""
    if trigger == HookTrigger.BEFORE_BROADCAST or not injected:
        return injected
    logger.warning(
        "Hook %s injected %d event(s) on %s: dropped, since only BEFORE_BROADCAST commits a "
        "hook's injected events (RFC §9.4); the rest of its result stands",
        hook.name,
        len(injected),
        trigger.value,
        extra={"room_id": room_id, "hook": hook.name, "trigger": trigger.value},
    )
    return []


def _named(hook: HookRegistration, scope: Iterable[HookRegistration | str]) -> HookRegistration:
    """*hook* under a name unique within its *scope*, the kit's hooks or one
    room's (RFC §9.1). An explicit name a hook of the scope holds, or one the
    framework writes in ``blocked_by``, is refused; a hook given none takes
    its function's name, made unique.

    Raises:
        ValueError: the explicit name is taken or reserved.
    """
    taken = {entry if isinstance(entry, str) else entry.name for entry in scope}
    if hook.name:
        if hook.name in RESERVED_HOOK_NAMES:
            raise ValueError(
                f"Hook name {hook.name!r} is a value the framework writes in "
                "blocked_by; a block of this hook would read as the framework's"
            )
        if hook.name in taken:
            raise ValueError(
                f"A hook named {hook.name!r} is already registered in this scope: "
                "a hook's name is unique within its scope (RFC §9.1)"
            )
        return hook
    # A leading "_" marks the framework's own hooks (an internal
    # ON_SESSION_STARTED hook is awaited): a default name never takes it.
    base = getattr(hook.fn, "__name__", "").lstrip("_") or "hook"
    name, rank = base, 2
    while name in taken or name in RESERVED_HOOK_NAMES:
        name, rank = f"{base}#{rank}", rank + 1
    return replace(hook, name=name)


async def _run_body(
    hook: HookRegistration,
    event: RoomEvent | Any,
    context: RoomContext,
    room_id: str,
    budget: ReentryBudget | None,
    floor: int,
) -> None:
    """*hook*'s body, run asynchronously (every hook of an ASYNC trigger,
    whatever its declared mode, RFC §9.2, and an ASYNC hook on a SYNC one),
    within its timeout. What it sends is at least *floor* deep, the depth
    its trigger fired at, for as long as it runs, the work that fired it
    over or not, and a task it starts that calls once it has returned is at
    the depth of its own event (:func:`floors_lent_while_it_runs`).
    On *budget*, the budget of the cascade that fired it once an event's
    delivery set has run, or of the work a direct change was made in, it is
    that work until it is over: what it sends into the room, and a direct
    change it makes, draws on the budget's hook calls and is at least one
    deeper than *event*, and a task it starts that calls afterwards has a
    budget of its own (RFC §8.3, §9.5.1, §10.1 step 16, §10.3)."""
    with floors_lent_while_it_runs(floor):
        if budget is None:
            await _hook_body(hook, event, context)
            return
        depth = event.chain_depth if isinstance(event, RoomEvent) else 0
        with after_delivery(room_id, budget, depth):
            await _hook_body(hook, event, context)


async def _hook_body(hook: HookRegistration, *args: Any) -> Any:
    """*hook*'s body on *args* within its timeout, the room locks its caller
    holds and the admission orders it stands in (an off-lock hook's) lent to
    it while it runs: a task it starts acts for it until it returns, and
    calls into the room as a new arrival afterwards (RFC §9.5.1, §13.5)."""
    with lent_while_it_runs(), orders_lent_while_it_runs():
        return await asyncio.wait_for(hook.fn(*args), timeout=hook.timeout)


class HookEngine:
    """Manages global and per-room hook registration and execution.

    An engine sits on one side of the turn backend (RFC §10.6.6): a kit's
    takes the room owner's triggers, a turn executor's (*turns*) the
    triggers a turn fires. A kit's engine hands a hook on a trigger a turn
    fires to its own executor (its ``turn_side``); otherwise each engine
    refuses the other side's at registration, where the hook would never
    fire.
    """

    def __init__(self, *, turns: bool = False) -> None:
        self._turns = turns
        self._fail_closed = frozenset(
            t for t in self.DEFAULT_FAIL_CLOSED_TRIGGERS if (t in TURN_TRIGGERS) == turns
        )
        self._global_hooks: list[HookRegistration] = []
        self._room_hooks: dict[str, list[HookRegistration]] = {}
        # A turn executor's room hooks, defined by name: a request names those
        # of its room (RFC §10.6.6), and they apply to the room from its turn.
        self._named_room_hooks: dict[str, list[HookRegistration]] = {}
        # The names each room uses on a turn executor between its turns (its
        # owner's, or those its last turn named), resolved at each read: a
        # name defined, removed or defined again after the room named it
        # reads as it stands.
        self._room_hook_names: dict[str, tuple[str, ...]] = {}
        # The kit's own executor's room hooks the kit gave it, by room: they
        # apply to the room's turns there, beside those a request names.
        self._owner_room_hooks: dict[str, list[HookRegistration]] = {}
        self._trigger_index: set[HookTrigger] = set()
        self.turn_side: TurnSide | None = None
        """Set by RoomKit on its engine: its own executor, where a hook on a
        trigger a turn fires registered here goes."""
        self._telemetry: Any = None  # Set by RoomKit after init
        # Set by RoomKit after init — RFC §8.2 mandates a ``hook_timeout``
        # framework event, which the engine alone knows how to raise.
        self._framework_emitter: Any = None
        # Set by RoomKit after init: whether a room (any room, for None) holds
        # a discussion (RFC §19.7.5 rule 1), which a router's hook cannot join.
        self._holds_discussion: Callable[[str | None], bool] | None = None
        # Set by RoomKit after init: the names a room enables on the turn
        # executors (RFC §10.6.6), which join the room's scope (§9.1).
        self._enabled_room_hooks: Callable[[str], list[str]] | None = None
        self._suppressed_triggers: set[str] = {
            "on_input_audio_level",
            "on_output_audio_level",
            "on_vad_audio_level",
        }
        # The ASYNC observers still running: the operation they observe went
        # on without them (RFC §9), and close() waits for them.
        self._observers: set[asyncio.Task[None]] = set()

    async def _emit_hook_timeout(
        self, room_id: str, hook: HookRegistration, trigger: HookTrigger
    ) -> None:
        """Raise the RFC §8.2 ``hook_timeout`` framework event.

        A timeout is its own observable condition, distinct from
        ``hook_error``: an operator reading the event stream can tell a hook
        that raised from one that never came back.
        """
        if self._framework_emitter is None:
            return
        await self._framework_emitter(
            "hook_timeout",
            room_id=room_id,
            data={
                "hook_name": hook.name,
                "trigger": str(trigger),
                "timeout": hook.timeout,
            },
        )

    def _check_lock_placement(self, hook: HookRegistration, room_id: str | None) -> None:
        """Enforce RFC §9.1's registration rules before *hook* is added.

        ``fail_closed`` only exists on a SYNC hook: an ASYNC one cannot block,
        and accepting the flag there would promise a guarantee nobody keeps.
        ``needs_lock=False`` only exists on a SYNC ``BEFORE_BROADCAST`` hook.
        And off-lock hooks run first, so a locked hook ordered before an
        off-lock one could not run in its declared place — a consent or
        budget gate meant to refuse a message before it is scanned would run
        after the scan. That is refused here, loudly, rather than reordered.
        """
        if hook.fail_closed and hook.execution != HookExecution.SYNC:
            raise ValueError(
                f"Hook {hook.name!r}: fail_closed=True needs a SYNC hook; an ASYNC hook "
                "cannot block"
            )
        is_check = (
            hook.trigger == HookTrigger.BEFORE_BROADCAST and hook.execution == HookExecution.SYNC
        )
        if not hook.needs_lock and not is_check:
            raise ValueError(
                f"Hook {hook.name!r}: needs_lock=False is only supported on a SYNC "
                f"BEFORE_BROADCAST hook, not {hook.execution} {hook.trigger}"
            )
        if not is_check:
            return
        # A global hook runs in every room, so it meets every registered hook;
        # a room hook meets the global ones and its own room's.
        peers = list(self._global_hooks)
        if room_id is None:
            for hooks in self._room_hooks.values():
                peers.extend(hooks)
        else:
            peers.extend(self._room_hooks.get(room_id, []))
        for peer in peers:
            if peer.trigger != hook.trigger or peer.execution != HookExecution.SYNC:
                continue
            locked, off = (peer, hook) if not hook.needs_lock else (hook, peer)
            if locked.needs_lock and not off.needs_lock and locked.priority < off.priority:
                raise ValueError(
                    f"Hook {locked.name!r} (priority {locked.priority}) needs the room "
                    f"lock but is ordered before off-lock hook {off.name!r} (priority "
                    f"{off.priority}). Off-lock hooks run first (RFC §9.5.1): give "
                    f"{off.name!r} a priority at or below {locked.priority}, or keep "
                    f"it needs_lock=True. Channel and event filters are not considered."
                )

    def has_off_lock_hooks(self, room_id: str, event: RoomEvent) -> bool:
        """Whether a ``needs_lock=False`` check applies to *event* (RFC §9.5.1)."""
        return any(
            not h.needs_lock
            for h in self._get_hooks(
                room_id, HookTrigger.BEFORE_BROADCAST, HookExecution.SYNC, event=event
            )
        )

    def room_hook_names(self, room_id: str) -> list[str]:
        """The names of the hooks registered for *room_id* alone, those a
        kit's engine gave its own executor included."""
        names = [h.name for h in self._room_hooks.get(room_id, [])]
        names += [n for n in self._room_hook_names.get(room_id, ()) if n in self._named_room_hooks]
        if self.turn_side is not None and not self._turns:
            names += self.turn_side.room_hook_names(room_id)
        return names

    def has_router_hook(self, room_id: str) -> bool:
        """Whether a conversation router's hook applies to *room_id*."""
        hooks = [*self._global_hooks, *self._room_hooks.get(room_id, [])]
        return any(getattr(h.fn, _ROUTER_MARK, False) for h in hooks)

    def has_sync_hooks(
        self,
        room_id: str,
        trigger: HookTrigger,
        event: RoomEvent,
        *,
        needs_lock: bool | None = None,
    ) -> bool:
        """Whether a SYNC hook of *trigger* applies to *event* in *room_id*;
        with *needs_lock*, one that needs the lock (``True``) or not
        (``False``), as :meth:`run_sync_hooks` selects them (RFC §9.5.1)."""
        if (engine := self._turn_engine(trigger)) is not None:
            return engine.has_sync_hooks(room_id, trigger, event, needs_lock=needs_lock)
        hooks = self._get_hooks(room_id, trigger, HookExecution.SYNC, event=event)
        return any(needs_lock is None or h.needs_lock == needs_lock for h in hooks)

    def has_hooks_for(self, room_id: str, trigger: HookTrigger, event: Any) -> bool:
        """Whether a hook of *trigger* applies to *event* in *room_id*, its
        filters read (a turn's event: its agent, RFC §9.1)."""
        if (engine := self._turn_engine(trigger)) is not None:
            return engine.has_hooks_for(room_id, trigger, event)
        return bool(self._get_hooks(room_id, trigger, None, event=event))

    def _turn_engine(self, trigger: HookTrigger) -> HookEngine | None:
        """The engine a kit's engine reads *trigger*'s hooks from: its own
        executor's, for a trigger a turn fires."""
        if self._turns or self.turn_side is None or trigger not in TURN_TRIGGERS:
            return None
        return self.turn_side.engine

    def _check_router_placement(self, hook: HookRegistration, room_id: str | None) -> None:
        """Refuse a router's hook where a discussion takes the turns (RFC
        §19.7.5 rule 1): one rule decides who speaks, never two."""
        holds = self._holds_discussion
        if holds is None or not getattr(hook.fn, _ROUTER_MARK, False) or not holds(room_id):
            return
        where = f"Room {room_id} holds" if room_id is not None else "A room holds"
        raise ValueError(f"{where} a discussion: a router cannot be installed there")

    def _check_side(self, hook: HookRegistration, *, room: bool) -> None:
        """Refuse a hook on a trigger that fires on the other side of the
        turn backend (RFC §10.6.6), naming the call that registers it there:
        a room's hook stays a room's, a global one stays global."""
        if (hook.trigger in TURN_TRIGGERS) == self._turns:
            return
        if self._turns:
            where = "the room owner"
            call = "kit.add_room_hook(room_id, ...)" if room else "kit.hook(...)"
        else:
            where = "the turn executor"
            call = (
                "kit.turn_executor.room_hook(name, ...) and kit.enable_room_hook(room_id, name)"
                if room
                else "kit.turn_executor.hook(...) or kit.turn_executor.hooks.register(...)"
            )
        raise ValueError(
            f"Hook {hook.name!r}: {hook.trigger} fires on {where} (RFC §10.6.6); "
            f"register it with {call}"
        )

    def _to_turn_side(self, hook: HookRegistration) -> TurnSide | None:
        """The executor a kit's engine hands *hook* to, when it fires there."""
        if self._turns or hook.trigger not in TURN_TRIGGERS:
            return None
        return self.turn_side

    def register(self, hook: HookRegistration) -> HookRegistration:
        """Register a global hook; the registration held, under the name it
        was given or made unique (RFC §9.1)."""
        # The kit's hooks are one scope, the ones its own executor holds
        # included (RFC §9.1).
        hook = _named(hook, self._global_names())
        if (turn_side := self._to_turn_side(hook)) is not None:
            return turn_side.register(hook)
        self._check_side(hook, room=False)
        self._check_router_placement(hook, None)
        self._check_lock_placement(hook, None)
        self._global_hooks.append(hook)
        self._trigger_index.add(hook.trigger)
        return hook

    def add_room_hook(self, room_id: str, hook: HookRegistration) -> HookRegistration:
        """Register a hook for a specific room.

        The engine holds no store and reads no room: this call is the
        synchronous, trusted level below the kit's
        :meth:`RoomKit.add_room_hook`, which reads the room with the
        caller's organization first (RFC §17.2).

        A turn executor's room hooks are defined by name instead
        (:meth:`define_room_hook`), the room owner naming a room's in its
        turns' requests (RFC §10.6.6): a hook held for a room on one executor
        would never fire for the room's turns on another.
        """
        hook = _named(hook, self._room_names(room_id))
        if (turn_side := self._to_turn_side(hook)) is not None:
            return turn_side.add_room_hook(room_id, hook)
        if self._turns:
            raise ValueError(
                f"Hook {hook.name!r}: a turn executor's room hook is defined by name, "
                "executor.room_hook(name, ...), and enabled for a room on its owner, "
                "kit.enable_room_hook(room_id, name) (RFC §10.6.6)"
            )
        self._check_side(hook, room=True)
        self._check_router_placement(hook, room_id)
        self._check_lock_placement(hook, room_id)
        self._room_hooks.setdefault(room_id, []).append(hook)
        self._trigger_index.add(hook.trigger)
        return hook

    def add_owner_room_hook(self, room_id: str, hook: HookRegistration) -> HookRegistration:
        """Register on the kit's own executor a room hook the kit gave it
        for *room_id* (RFC §10.6.6): it applies to the room's turns here,
        beside the room hooks a request names."""
        if not self._turns:
            raise ValueError("A room hook the kit gives its executor registers on an executor")
        hook = _named(hook, self._owner_room_hooks.get(room_id, []))
        self._check_side(hook, room=True)
        self._check_lock_placement(hook, room_id)
        self._owner_room_hooks.setdefault(room_id, []).append(hook)
        self._trigger_index.add(hook.trigger)
        return hook

    def _hooks_of_room(self, room_id: str) -> list[HookRegistration]:
        """*room_id*'s hooks: inside a turn of the room, those its request
        named (RFC §10.6.6), whatever the room's set became meanwhile; else
        the room's set; with, on the kit's own executor, those the kit gave
        it for the room."""
        given = self._owner_room_hooks.get(room_id, [])
        if not self._turns:
            return [*self._room_hooks.get(room_id, []), *given]
        turn = _TURN_ROOM_HOOKS.get()
        if turn is not None and turn[0] == room_id:
            names = turn[1]
        else:
            names = self._room_hook_names.get(room_id, ())
        return [*self._defined_under(names), *given]

    def _defined_under(self, names: Iterable[str]) -> list[HookRegistration]:
        """The room hooks defined under *names* now, in their order."""
        return [hook for name in names for hook in self._named_room_hooks.get(name, ())]

    def define_room_hook(self, hook: HookRegistration) -> None:
        """Define a turn executor's room hook under its name: it applies to a
        room whose turn's request names it (RFC §10.6.6). A name carries a
        hook per trigger: defining the name on a trigger again replaces it."""
        if not self._turns:
            raise ValueError("Room hooks are defined by name on a turn executor only")
        if not hook.name:
            raise ValueError("A room hook defined on a turn executor needs a name")
        if hook.name in RESERVED_HOOK_NAMES:
            raise ValueError(
                f"Hook name {hook.name!r} is a value the framework writes in blocked_by"
            )
        self._check_side(hook, room=True)
        self._check_lock_placement(hook, None)
        defined = self._named_room_hooks.setdefault(hook.name, [])
        defined[:] = [h for h in defined if h.trigger != hook.trigger]
        defined.append(hook)
        self._trigger_index.add(hook.trigger)

    def undefine_room_hook(self, name: str) -> bool:
        """Forget the room hooks defined under *name*, in every room."""
        if self._named_room_hooks.pop(name, None) is None:
            return False
        self._rebuild_trigger_index()
        return True

    def defines_room_hooks(self, names: Iterable[str]) -> bool:
        """Whether a room hook is defined under each of *names*."""
        return all(name in self._named_room_hooks for name in names)

    def defined_room_hooks(self) -> frozenset[str]:
        """The names room hooks are defined under."""
        return frozenset(self._named_room_hooks)

    def use_room_hooks(self, room_id: str, names: Iterable[str]) -> None:
        """Apply to *room_id* the room hooks defined under *names*, as its
        turn's request names them, in place of those it had: each read finds
        what is defined under them then."""
        names = tuple(names)
        if names:
            self._room_hook_names[room_id] = names
        else:
            self._room_hook_names.pop(room_id, None)

    def remove_global_hook(self, name: str) -> bool:
        """Remove a global hook by name, one a kit's engine gave its own
        executor included."""
        if self._removed(self._global_hooks, name):
            return True
        return self.turn_side is not None and self.turn_side.remove_global_hook(name)

    def replace_room_hook(self, room_id: str, hook: HookRegistration) -> HookRegistration:
        """Register *hook* for *room_id* in place of the room's hook of the
        same name: a strategy installed again in its room replaces what it
        set up there, a name being unique within a room's hooks (RFC §9.1).
        A registration refused leaves the hook it would replace in place."""
        hooks = self._room_hooks.get(room_id, [])
        old = next((h for h in hooks if hook.name and h.name == hook.name), None)
        position = hooks.index(old) if old is not None else -1
        if old is not None:
            hooks.remove(old)
        try:
            return self.add_room_hook(room_id, hook)
        except Exception:
            if old is not None:
                hooks.insert(position, old)
            raise

    def _global_names(self) -> list[str]:
        """The names of the kit's hooks, those its own executor holds too."""
        names = [hook.name for hook in self._global_hooks]
        if self.turn_side is not None:
            names.extend(self.turn_side.global_hook_names())
        return names

    def _room_names(self, room_id: str) -> list[str]:
        """The names of *room_id*'s hooks, those the kit's own executor holds
        for it and those it enables on the executors too (RFC §10.6.6)."""
        names = [hook.name for hook in self._room_hooks.get(room_id, [])]
        if self.turn_side is not None:
            names.extend(self.turn_side.room_hook_names(room_id))
        if self._enabled_room_hooks is not None:
            names.extend(self._enabled_room_hooks(room_id))
        return names

    def remove_room_hook(self, room_id: str, name: str) -> bool:
        """Remove a room hook by name, one a kit's engine gave its own
        executor included. The engine reads no room: the synchronous, trusted
        level below :meth:`RoomKit.remove_room_hook`, which reads the room
        with the caller's organization first (RFC §17.2)."""
        for hooks in (self._room_hooks.get(room_id, []), self._owner_room_hooks.get(room_id, [])):
            if self._removed(hooks, name):
                return True
        if self._unnamed(room_id, name):
            return True
        return self.turn_side is not None and self.turn_side.remove_room_hook(room_id, name)

    def _unnamed(self, room_id: str, name: str) -> bool:
        """Stop *room_id* using the room hooks defined under *name*; whether
        it used them."""
        names = self._room_hook_names.get(room_id, ())
        if name not in names or name not in self._named_room_hooks:
            return False
        self.use_room_hooks(room_id, [n for n in names if n != name])
        return True

    def holds(self, hook: HookRegistration) -> bool:
        """Whether the registration *hook* itself is registered here."""
        return any(any(h is hook for h in listed) for listed in self._registrations())

    def remove_hook(self, hook: HookRegistration, room_id: str | None = None) -> bool:
        """Remove the registration *hook* itself, once: of *room_id*, or
        global."""
        scope = (
            [self._global_hooks]
            if room_id is None
            else [self._room_hooks.get(room_id, []), self._owner_room_hooks.get(room_id, [])]
        )
        for listed in scope:
            for i, registered in enumerate(listed):
                if registered is hook:
                    listed.pop(i)
                    self._rebuild_trigger_index()
                    return True
        return False

    def _registrations(self) -> list[list[HookRegistration]]:
        return [self._global_hooks, *self._room_hooks.values(), *self._owner_room_hooks.values()]

    def _removed(self, hooks: list[HookRegistration], name: str) -> bool:
        for i, h in enumerate(hooks):
            if h.name == name:
                hooks.pop(i)
                self._rebuild_trigger_index()
                return True
        return False

    def has_hooks(self, trigger: HookTrigger | None = None) -> bool:
        """Whether a hook is registered — for ``trigger``, or for any trigger at all.

        O(1) either way: a set lookup, or the set's emptiness. The framework
        skips the work only a hook would consume when nothing is listening —
        the context built for ``BEFORE_DELIVER``, the room history loaded for
        the inbound pipeline — so it runs a handful of times per message,
        never per hook. Global and room hooks alike are in the index; identity
        hooks live in the framework's own registry, and a caller that needs
        both asks both.
        """
        if trigger is None:
            turn_side = None if self._turns else self.turn_side
            return bool(self._trigger_index) or (
                turn_side is not None and turn_side.engine.has_hooks()
            )
        if (engine := self._turn_engine(trigger)) is not None:
            return engine.has_hooks(trigger)
        return trigger in self._trigger_index

    def _rebuild_trigger_index(self) -> None:
        """Rebuild the trigger index from all registered hooks."""
        self._trigger_index = {h.trigger for h in self._global_hooks}
        for hooks in [
            *self._room_hooks.values(),
            *self._named_room_hooks.values(),
            *self._owner_room_hooks.values(),
        ]:
            self._trigger_index.update(h.trigger for h in hooks)

    def _hook_matches_event(self, hook: HookRegistration, event: RoomEvent) -> bool:
        """Check if a hook's filters match the given event."""
        if not isinstance(event, RoomEvent):
            # A turn's event (a tool call, a response, a thought): its
            # channel is the agent's; a type it does not carry filters nothing.
            channel_type = getattr(event, "channel_type", None)
            return (hook.channel_ids is None or event.channel_id in hook.channel_ids) and (
                hook.channel_types is None
                or channel_type is None
                or channel_type in hook.channel_types
            )
        source = event.source

        # All filters must pass (None means "match all")
        type_ok = hook.channel_types is None or source.channel_type in hook.channel_types
        id_ok = hook.channel_ids is None or source.channel_id in hook.channel_ids
        dir_ok = hook.directions is None or source.direction in hook.directions
        event_ok = hook.event_types is None or event.type in hook.event_types

        return type_ok and id_ok and dir_ok and event_ok

    def _get_hooks(
        self,
        room_id: str,
        trigger: HookTrigger,
        execution: HookExecution | None,
        event: RoomEvent | None = None,
    ) -> list[HookRegistration]:
        """Get merged global + room hooks filtered and sorted by priority.

        Args:
            room_id: The room ID to get hooks for
            trigger: The hook trigger to filter by
            execution: The execution mode to filter by, or ``None`` to
                match all execution modes.
            event: Optional event to filter hooks by channel_type/id/direction
        """
        all_hooks = [
            h
            for h in self._global_hooks
            if h.trigger == trigger and (execution is None or h.execution == execution)
        ]
        room_hooks = [
            h
            for h in self._hooks_of_room(room_id)
            if h.trigger == trigger and (execution is None or h.execution == execution)
        ]
        all_hooks.extend(room_hooks)

        # Apply event-based filters if event is provided
        if event is not None:
            all_hooks = [h for h in all_hooks if self._hook_matches_event(h, event)]

        all_hooks.sort(key=lambda h: h.priority)
        return all_hooks

    #: Triggers whose payload is content that a hook may be there to withhold —
    #: redacting a transcript, holding back speech — or an action it may be
    #: there to prevent: a tool call behind an approval hook. On those, a hook
    #: that raises blocks rather than letting the payload through: logging the
    #: error and carrying on would publish, or run, exactly what the hook
    #: existed to stop (RFC §9.3). Everywhere else a failing hook stays
    #: non-fatal, so a broken hook cannot take a room down.
    DEFAULT_FAIL_CLOSED_TRIGGERS: ClassVar[frozenset[HookTrigger]] = frozenset(
        {HookTrigger.BEFORE_TTS, HookTrigger.ON_TRANSCRIPTION, HookTrigger.BEFORE_TOOL_USE}
    )

    @property
    def FAIL_CLOSED_TRIGGERS(self) -> frozenset[HookTrigger]:  # noqa: N802 - kept the public name
        """This engine's fail-closed triggers: the defaults that fire on its
        side of the turn backend (RFC §10.6.6), or the set a host gave it;
        a kit's engine reads its own executor's with them."""
        if self.turn_side is not None and not self._turns:
            return self._fail_closed | self.turn_side.fail_closed
        return self._fail_closed

    @FAIL_CLOSED_TRIGGERS.setter
    def FAIL_CLOSED_TRIGGERS(self, triggers: frozenset[HookTrigger]) -> None:  # noqa: N802
        """Set the fail-closed triggers: a kit's engine gives its own
        executor those a turn fires; another engine refuses a trigger of the
        other side, since it never runs its hooks."""
        if self.turn_side is not None and not self._turns:
            self.turn_side.set_fail_closed(frozenset(t for t in triggers if t in TURN_TRIGGERS))
            self._fail_closed = frozenset(t for t in triggers if t not in TURN_TRIGGERS)
            return
        elsewhere = sorted(str(t) for t in triggers if (t in TURN_TRIGGERS) != self._turns)
        if elsewhere:
            where = "kit.hook_engine" if self._turns else "kit.turn_executor.hooks"
            raise ValueError(
                f"{', '.join(elsewhere)} fire(s) on the other side of the turn backend "
                f"(RFC §10.6.6): set them on {where}.FAIL_CLOSED_TRIGGERS"
            )
        self._fail_closed = frozenset(triggers)

    def _apply_rewrite(
        self,
        result: SyncPipelineResult,
        hook: HookRegistration,
        trigger: HookTrigger,
        room_id: str,
        original: Any,
        hook_result: HookResult,
        fold: Callable[[Any, dict[str, Any]], Any] | None,
        bindings: Iterable[ChannelBinding] = (),
    ) -> bool:
        """Carry *hook_result*'s rewrite into the event the next hook sees.

        A MODIFY replaces the payload; with a *fold*, the hook's metadata
        rewrite is written into it too, and the fold runs after every hook
        that rewrote the payload, either way. A MODIFY that carries no
        payload, or one not of the type the trigger passed in, is an unusable
        result and replaces nothing (RFC §9.3): it blocks when the hook fails
        closed, else the chain carries on from the previous outcome, under a
        *fold* with the hook's metadata rewrite still applied. A
        BEFORE_BROADCAST replacement keeps the identity of the event it
        replaces (§9.4), a list in it read against the room's *bindings*.
        Returns whether the pipeline was closed.
        """
        current = result.event if result.event is not None else original
        payload = hook_result.event
        if hook_result.action == "modify":
            if _usable_payload(current, payload):
                if trigger == HookTrigger.BEFORE_BROADCAST and isinstance(payload, RoomEvent):
                    payload = kept_identity(current, payload, bindings)
                result.event = payload
            else:
                # The consumer would silently ignore a payload it cannot use
                # and carry on with the original — which for a redaction hook
                # publishes the very content it meant to replace.
                expected = type(current).__name__
                got = "nothing" if payload is None else type(payload).__name__
                logger.error(
                    "Sync hook %s returned a %s where a %s was expected",
                    hook.name,
                    got,
                    expected,
                    extra={"room_id": room_id},
                )
                result.hook_errors.append(
                    {"hook": hook.name, "error": f"modify returned {got}, expected {expected}"}
                )
                if self._close(
                    result,
                    hook,
                    trigger,
                    "hook_invalid_result",
                    f"hook {hook.name} returned an unusable payload",
                ):
                    return True
        rewrote = hook_result.action == "modify" and payload is not None
        if fold is not None and (hook_result.metadata or rewrote):
            latest = result.event if result.event is not None else original
            result.event = fold(latest, hook_result.metadata or {})
        return False

    def _refuse_too_deep(
        self,
        result: SyncPipelineResult,
        hook: HookRegistration,
        trigger: HookTrigger,
        room_id: str,
        span_id: str | None,
    ) -> bool:
        """Take *hook*'s result holding a sixth composite level as the
        unusable result it is (RFC §5.3, §9.3): such content is refused where
        it is built, so nothing of the result is applied or collected, and
        its error is reported. Returns whether the pipeline was closed."""
        logger.error(
            "Sync hook %s returned content nested over %d composite levels",
            hook.name,
            MAX_COMPOSITE_DEPTH,
            extra={"room_id": room_id},
        )
        error = f"content nested over {MAX_COMPOSITE_DEPTH} composite levels"
        result.hook_errors.append({"hook": hook.name, "error": error})
        if span_id is not None and self._telemetry is not None:
            self._telemetry.end_span(span_id, status="error", error_message=error)
        return self._close(
            result, hook, trigger, "hook_invalid_result", f"hook {hook.name} returned {error}"
        )

    def _fails_closed(self, hook: HookRegistration, trigger: HookTrigger) -> bool:
        """Whether an unusable outcome of *hook* blocks (RFC §9.3)."""
        return hook.fail_closed or trigger in self.FAIL_CLOSED_TRIGGERS

    def fail_closed_hook(
        self, room_id: str, trigger: HookTrigger, event: Any = None
    ) -> str | None:
        """The name of the first SYNC hook on *trigger* that fails closed, or None.

        For a caller that cannot run the pipeline at all (the room's context
        would not build): with such a hook registered, the payload must be
        withheld as the hook's own failure would withhold it.
        """
        if (engine := self._turn_engine(trigger)) is not None:
            return engine.fail_closed_hook(room_id, trigger, event)
        for hook in self._get_hooks(room_id, trigger, HookExecution.SYNC, event=event):
            if self._fails_closed(hook, trigger):
                return hook.name
        return None

    def _close(
        self,
        result: SyncPipelineResult,
        hook: HookRegistration,
        trigger: HookTrigger,
        outcome: str,
        trigger_reason: str,
    ) -> bool:
        """Block *result* when *hook* fails closed; return whether it did.

        A hook that declared ``fail_closed`` names itself and the outcome
        (``hook_timeout:<name>``) so the sender can be told why the message
        did not go out. A fail-closed *trigger* reports a human-readable
        reason instead.
        """
        if not self._fails_closed(hook, trigger):
            return False
        result.allowed = False
        result.failed_closed = True
        if hook.fail_closed:
            result.reason = f"{outcome}:{hook.name}"
            result.blocked_by = hook.name
        else:
            result.reason = trigger_reason
        return True

    async def run_sync_hooks(
        self,
        room_id: str,
        trigger: HookTrigger,
        event: RoomEvent | Any,
        context: RoomContext,
        *,
        skip_event_filter: bool = False,
        needs_lock: bool | None = None,
        fold: Callable[[Any, dict[str, Any]], Any] | None = None,
        fire_observers: bool = True,
    ) -> SyncPipelineResult:
        """Run sync hooks sequentially. Stops on block, passes modified events.

        A kit's engine runs a trigger a turn fires on its own executor's.

        Args:
            room_id: The room ID to run hooks for.
            trigger: The hook trigger type.
            event: The event to pass to hooks. For voice hooks, this may be
                a VoiceSession or str instead of RoomEvent.
            context: The room context.
            skip_event_filter: If True, skip channel-based event filtering.
                Use this for voice hooks where event is not a RoomEvent.
            needs_lock: ``None`` runs every SYNC hook. ``False`` runs only the
                off-lock ones (RFC §9.5.1) and fires no ASYNC observer — the
                locked pass that follows fires them once, on the final event.
                ``True`` runs only the hooks that need the lock.
            fold: For a trigger whose hooks may also rewrite the payload
                through ``metadata`` (ON_TOOL_CALL's result override),
                ``fold(event, metadata)`` returns the event as a hook's
                rewrite left it (a MODIFY's payload or the event, with the
                metadata written in; ``metadata`` is empty after a MODIFY
                that set none), so the next hook, the ASYNC observers and the caller
                all see the chain's latest state. ``None`` leaves metadata
                beside the event.
            fire_observers: ``False`` leaves the ASYNC observers to the
                caller, for a firing whose observers must see something other
                than the event the chain left: a served tool call's result as
                the model reads it, or a report no SYNC hook can change; or
                must start later: a pass's ``BEFORE_BROADCAST`` observers,
                once its event is committed and laned (RFC §10.1).
        """
        if (engine := self._turn_engine(trigger)) is not None:
            return await engine.run_sync_hooks(
                room_id,
                trigger,
                event,
                context,
                skip_event_filter=skip_event_filter,
                needs_lock=needs_lock,
                fold=fold,
                fire_observers=fire_observers,
            )
        filter_event = None if skip_event_filter else event
        hooks = self._get_hooks(room_id, trigger, HookExecution.SYNC, event=filter_event)
        if needs_lock is not None:
            hooks = [h for h in hooks if h.needs_lock == needs_lock]
        result = SyncPipelineResult(event=event)

        for hook in hooks:
            span_id = None
            should_trace = (
                self._telemetry is not None and str(trigger) not in self._suppressed_triggers
            )
            if should_trace:
                from roomkit.telemetry.base import Attr, SpanKind
                from roomkit.telemetry.context import get_current_span

                span_id = self._telemetry.start_span(
                    SpanKind.HOOK_SYNC,
                    f"hook.sync.{hook.name or 'unnamed'}",
                    parent_id=get_current_span(),
                    room_id=room_id,
                    attributes={
                        Attr.HOOK_NAME: hook.name or "unnamed",
                        Attr.HOOK_TRIGGER: str(trigger),
                    },
                )
            try:
                # ``is not None`` rather than truthiness: redacting to an empty
                # string is a modification, and treating it as absent would hand
                # the next hook the original secret.
                current_event = result.event if result.event is not None else event
                hook_result: HookResult = await _hook_body(hook, current_event, context)
            except TimeoutError:
                logger.warning(
                    "Sync hook %s timed out after %.1fs",
                    hook.name,
                    hook.timeout,
                    extra={"room_id": room_id},
                )
                await self._emit_hook_timeout(room_id, hook, trigger)
                result.hook_errors.append(
                    {"hook": hook.name, "error": f"timeout ({hook.timeout}s)"}
                )
                if span_id is not None:
                    self._telemetry.end_span(span_id, status="error", error_message="timeout")
                if self._close(
                    result,
                    hook,
                    trigger,
                    "hook_timeout",
                    f"hook {hook.name} timed out after {hook.timeout}s",
                ):
                    return result
                continue
            except Exception as exc:
                logger.exception("Sync hook %s failed", hook.name, extra={"room_id": room_id})
                result.hook_errors.append({"hook": hook.name, "error": str(exc)})
                if span_id is not None:
                    self._telemetry.end_span(span_id, status="error", error_message=str(exc))
                if self._close(
                    result, hook, trigger, "hook_error", f"hook {hook.name} failed: {exc}"
                ):
                    return result
                continue

            if not isinstance(hook_result, HookResult):
                logger.error(
                    "Sync hook %s returned %s instead of HookResult — skipping",
                    hook.name,
                    type(hook_result).__name__,
                    extra={"room_id": room_id},
                )
                result.hook_errors.append(
                    {
                        "hook": hook.name,
                        "error": f"expected HookResult, got {type(hook_result).__name__}",
                    }
                )
                if span_id is not None:
                    self._telemetry.end_span(
                        span_id, status="error", error_message="invalid return type"
                    )
                if self._close(
                    result,
                    hook,
                    trigger,
                    "hook_invalid_result",
                    f"hook {hook.name} returned {type(hook_result).__name__} "
                    "instead of HookResult",
                ):
                    return result
                continue

            if _holds_too_deep(hook_result):
                if self._refuse_too_deep(result, hook, trigger, room_id, span_id):
                    return result
                continue

            if span_id is not None:
                self._telemetry.end_span(
                    span_id,
                    attributes={Attr.HOOK_RESULT: hook_result.action},
                )

            # Side effects are always collected (RFC §7.5 rule 3); an
            # unusable result's injected events are not (§9.3).
            current = result.event if result.event is not None else event
            if hook_result.action != "modify" or _usable_payload(current, hook_result.event):
                result.injected_events.extend(
                    _committed_injections(hook_result.injected_events, trigger, hook, room_id)
                )
            result.tasks.extend(hook_result.tasks)
            result.observations.extend(hook_result.observations)
            if hook_result.metadata:
                result.metadata.update(hook_result.metadata)

            if hook_result.action == "block":
                result.allowed = False
                result.reason = hook_result.reason
                result.blocked_by = hook.name
                return result

            if self._apply_rewrite(
                result, hook, trigger, room_id, event, hook_result, fold, context.bindings
            ):
                return result

        # Fire ASYNC observers for the same trigger (fire-and-forget).
        # This allows ASYNC hooks to observe events from triggers that
        # are only invoked via run_sync_hooks (e.g. ON_TRANSCRIPTION,
        # ON_VISION_RESULT, ON_TOOL_CALL).  Only ASYNC hooks are fired
        # — SYNC hooks already ran above.
        if needs_lock is False or not fire_observers:
            return result
        final_event = result.event if result.event is not None else event
        filter_ev = None if skip_event_filter else final_event
        async_hooks = self._get_hooks(
            room_id,
            trigger,
            HookExecution.ASYNC,
            event=filter_ev,
        )
        self._observe(async_hooks, room_id, trigger, final_event, context)
        return result

    async def run_async_hooks(
        self,
        room_id: str,
        trigger: HookTrigger,
        event: RoomEvent | Any,
        context: RoomContext,
        *,
        skip_event_filter: bool = False,
        name_prefix: str | None = None,
        exclude_name_prefix: str | None = None,
        budget: ReentryBudget | None = None,
    ) -> None:
        """Run async hooks concurrently. Errors are logged, never raised.

        A kit's engine runs a trigger a turn fires on its own executor's.

        Finds hooks regardless of their declared execution mode so that
        hooks registered with the default ``SYNC`` execution still fire
        for triggers that are only invoked asynchronously (e.g.
        ``AFTER_BROADCAST``, lifecycle hooks, voice hooks).

        Args:
            room_id: The room ID to run hooks for.
            trigger: The hook trigger type.
            event: The event to pass to hooks. For voice hooks, this may be
                a VoiceSession or str instead of RoomEvent.
            context: The room context.
            skip_event_filter: If True, skip channel-based event filtering.
                Use this for voice hooks where event is not a RoomEvent.
            name_prefix: Only run hooks whose name starts with this prefix.
            exclude_name_prefix: Skip hooks whose name starts with this prefix.
            budget: The reentry budget of the cascade that fires *trigger*
                once an event's delivery set has run (``AFTER_BROADCAST``, a
                change's mutation trigger), or of the work a direct change
                was made in: what each hook's body sends into the room draws
                on its hook calls while the body runs (RFC §9.5.1).
        """
        if (engine := self._turn_engine(trigger)) is not None:
            await engine.run_async_hooks(
                room_id,
                trigger,
                event,
                context,
                skip_event_filter=skip_event_filter,
                name_prefix=name_prefix,
                exclude_name_prefix=exclude_name_prefix,
                budget=budget,
            )
            return
        filter_event = None if skip_event_filter else event
        hooks = self._get_hooks(room_id, trigger, None, event=filter_event)
        if name_prefix is not None:
            hooks = [h for h in hooks if h.name.startswith(name_prefix)]
        if exclude_name_prefix is not None:
            hooks = [h for h in hooks if not h.name.startswith(exclude_name_prefix)]
        if not hooks:
            return

        await self._run_async_hooks_list(
            hooks, room_id, trigger, event, context, budget=budget, floor=chain_floor()
        )

    async def run_observers(
        self,
        room_id: str,
        trigger: HookTrigger,
        event: RoomEvent | Any,
        context: RoomContext,
        *,
        skip_event_filter: bool = False,
    ) -> None:
        """Run only the ASYNC-registered hooks for *trigger*, fire-and-forget.

        The counterpart of :meth:`run_async_hooks`, which deliberately ignores
        the declared execution mode. Here the mode is the whole point: it is
        what separates a hook that *observes* a call from one that *serves* it.

        A refused tool call must still be observable — an audit trail that
        cannot see a denial cannot tell a denied agent from an idle one — but
        it must not reach a hook that would serve it, or the denial would
        merely hide the side effect instead of preventing it. Only a SYNC hook
        can serve a call (it is the one that returns a result), so dispatching
        the ASYNC hooks alone makes the distinction structural rather than a
        rule each hook author has to remember. A kit's engine runs a trigger
        a turn fires on its own executor's.
        """
        if (engine := self._turn_engine(trigger)) is not None:
            await engine.run_observers(
                room_id, trigger, event, context, skip_event_filter=skip_event_filter
            )
            return
        hooks = self._get_hooks(
            room_id,
            trigger,
            HookExecution.ASYNC,
            event=None if skip_event_filter else event,
        )
        self._observe(hooks, room_id, trigger, event, context)

    def _start_async_hooks(
        self,
        room_id: str,
        trigger: HookTrigger,
        event: RoomEvent | Any,
        context: RoomContext,
        *,
        budget: ReentryBudget | None = None,
        aftermath: bool = False,
    ) -> None:
        """Start the hooks :meth:`run_async_hooks` runs for *trigger* and
        return at once: an ASYNC trigger's hooks run after the operation
        that fired it and never hold it up (RFC §9.3). Held as observers
        are (:meth:`_hold`, *aftermath* included), each body on *budget*
        when given."""
        self._hold(
            self.run_async_hooks(room_id, trigger, event, context, budget=budget),
            name=str(trigger),
            aftermath=aftermath,
        )

    def _has_async_hooks(self, room_id: str, trigger: HookTrigger, event: RoomEvent) -> bool:
        """Whether :meth:`run_async_hooks` finds hooks for *trigger* on *event*."""
        engine = self._turn_engine(trigger) or self
        return bool(engine._get_hooks(room_id, trigger, None, event=event))

    def _hold(
        self, work: Coroutine[Any, Any, None], *, name: str, aftermath: bool = False
    ) -> None:
        """Run *work*, hooks that observe an operation already over, as a
        task of its own, and return at once.

        Started in a context that holds no room lock: an observer of a
        locked section's chain calls into the room as a new arrival, which
        waits for the section, never as the lock's holder beside it (RFC
        §9.5.1). Named in :func:`held_runs`, so a ``close()`` it runs spares
        it rather than wait for itself, and under none of the runs that
        fired it: they do not wait for it, so such a ``close()`` waits for
        them, and cuts them, as for any other work. With *aftermath*, the
        hooks of a delivery's aftermath (an ``AFTER_BROADCAST`` hook, a
        change's mutation trigger), it runs under the runs that fired it,
        the room's lane among them: it is their work, so its ``close()``
        spares them as its own, and the lane drops the delivery it kept.
        The close gives it the grace, then cuts it (RFC §12.10.4).
        """
        context = unlocked_context()
        task = start_named(
            work,
            name=f"roomkit-observers-{name}",
            context=context if aftermath else under_no_held_run(context),
        )
        self._observers.add(task)
        task.add_done_callback(self._observers.discard)

    def _observe(
        self,
        hooks: list[HookRegistration],
        room_id: str,
        trigger: HookTrigger,
        event: RoomEvent | Any,
        context: RoomContext,
    ) -> None:
        """Start *hooks* as observers of *event* and return at once (:meth:`_hold`).

        An ASYNC hook runs after the operation that fired it and cannot hold
        it up (RFC §9): a slow observer of BEFORE_TTS delays no synthesis,
        one of ON_TOOL_CALL no tool loop. A hook that must have run before the
        operation goes on is a SYNC one.

        Fired from the room's work (a pass's section, an event's
        admission work, a delivery set's work), each body is that cascade's
        work as an ``AFTER_BROADCAST`` body is (:func:`_run_body`): what it
        sends into the room draws one of the cascade's hook calls, and past
        the cap is refused as theirs is, so an observer that posts on every
        event it observes stops (RMK-978). The observers of a trigger that
        fires per tool call (:data:`PER_TOOL_CALL_TRIGGERS`) are not the
        cascade's work: they run as any task the turn's work starts does.
        Each body sends at least as deep as the work that fired it was when
        the trigger fired, a turn, a tool call or a model's answer, even once
        that work is over, read here as the trigger fires: a hook that tells
        an agent again from its body ends at the depth limit (RMK-1004).
        """
        if not hooks:
            return
        per_tool_call = trigger in PER_TOOL_CALL_TRIGGERS
        budget = None if per_tool_call else work_budget(room_id)
        work = self._run_async_hooks_list(
            hooks, room_id, trigger, event, context, budget=budget, floor=chain_floor()
        )
        self._hold(work, name=str(trigger))

    async def drain_observers(self) -> None:
        """Wait for the observers still running, this engine's and its turn side's.

        Each hook is bounded by its own timeout; an observer that fires
        another trigger is waited for too. The observers the caller runs
        under are spared: waiting for them would wait for itself.
        """
        # Read again at each wake: an observer started meanwhile is waited
        # for, one whose caller joined the close is spared.
        await wait_for_unspared(self.running_observers)

    def running_observers(self) -> list[asyncio.Task[None]]:
        """The observers still running, this engine's and its turn side's,
        but the ones the caller runs under (:meth:`drain_observers`)."""
        engines = [self]
        if self.turn_side is not None and self.turn_side.engine is not self:
            engines.append(self.turn_side.engine)
        spared = spared_by_close()
        return [
            task
            for engine in engines
            for task in engine._observers
            if not task.done() and task not in spared
        ]

    async def _run_async_hooks_list(
        self,
        hooks: list[HookRegistration],
        room_id: str,
        trigger: HookTrigger,
        event: RoomEvent | Any,
        context: RoomContext,
        *,
        budget: ReentryBudget | None = None,
        floor: int,
    ) -> None:
        """Run a list of hooks concurrently. Errors are logged, never raised.

        Each body runs on *budget* when given, and at least *floor* deep,
        the depth *trigger* fired at (:func:`_run_body`)."""
        if not hooks:
            return

        async def _run_one(hook: HookRegistration) -> None:
            span_id = None
            should_trace = (
                self._telemetry is not None and str(trigger) not in self._suppressed_triggers
            )
            if should_trace:
                from roomkit.telemetry.base import Attr, SpanKind
                from roomkit.telemetry.context import get_current_span

                span_id = self._telemetry.start_span(
                    SpanKind.HOOK_ASYNC,
                    f"hook.async.{hook.name or 'unnamed'}",
                    parent_id=get_current_span(),
                    room_id=room_id,
                    attributes={
                        Attr.HOOK_NAME: hook.name or "unnamed",
                        Attr.HOOK_TRIGGER: str(trigger),
                    },
                )
            try:
                await _run_body(hook, event, context, room_id, budget, floor)
                if span_id is not None:
                    self._telemetry.end_span(span_id)
            except TimeoutError:
                logger.warning(
                    "Async hook %s timed out after %.1fs",
                    hook.name,
                    hook.timeout,
                    extra={"room_id": room_id},
                )
                await self._emit_hook_timeout(room_id, hook, trigger)
                if span_id is not None:
                    self._telemetry.end_span(span_id, status="error", error_message="timeout")
            except ReentryLoopCapError as capped:
                # The hook's own loop, stopped at its cascade's cap: not a
                # failure of the framework, nor of the room.
                logger.warning(
                    "Async hook %s stopped at the reentry cap: %s",
                    hook.name,
                    capped,
                    extra={"room_id": room_id},
                )
                if span_id is not None:
                    self._telemetry.end_span(span_id, status="error", error_message="capped")
            except Exception:
                logger.exception(
                    "Async hook %s failed",
                    hook.name,
                    extra={"room_id": room_id},
                )
                if span_id is not None:
                    self._telemetry.end_span(span_id, status="error", error_message="failed")

        await asyncio.gather(*[_run_one(hook) for hook in hooks], return_exceptions=True)
