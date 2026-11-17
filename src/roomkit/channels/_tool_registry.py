"""What a channel serves and what is installed on it, per room (RFC §19.7, §21.1).

One channel object serves every room it is attached to. The tools it serves
itself and the ones orchestration sets up on it are kept here, one entry per
tool: the definition the model reads, the function that serves a call, who set
it up, and the traits the channel's rules read (exempt from the policy, never
deferred by Tool Search, left out of the tool memory, pure within a turn).
An entry is the channel's for every room, or one room's: a strategy sets up a
room's tools, with that room's configuration, beside the other rooms' and
without touching them. A name is served by one tool in a room, so a second one
is refused when it is given.

The host's own tools stay the host's: its definitions and the handler it gave
serve every name no entry serves. The registry reads their names only, to
refuse an entry that would shadow one.

The tools orchestration sets up are the room owner's: the owner reads them
here for each turn it submits (:meth:`ChannelRegistry.owner_entries`) and
serves their calls (RFC §10.6.5). Inside a turn the turn backend runs, the
channel declares and calls the ones the request lists instead
(:class:`TurnOwnerTools`), so a channel an executor built elsewhere, whose
registry holds none of them, runs the same turn.

One more thing orchestration installs lives here rather than on the channel
object: the source a realtime channel reads a new session's configuration from
(a pipeline's active agent). Nothing is ever written onto the shared channel
for one room: a write made for one room would be read by every other room the
channel serves.
"""

from __future__ import annotations

import contextvars
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from roomkit.channels._skill_constants import (
    TOOL_ACTIVATE_SKILL,
    TOOL_READ_REFERENCE,
    TOOL_RUN_SCRIPT,
)
from roomkit.channels._tool_eviction import REREAD_TOOL
from roomkit.channels._tool_search_constants import (
    TOOL_CALL_TOOL,
    TOOL_FIND_TOOLS,
    TOOL_LIST_TOOLS,
)
from roomkit.core.exceptions import ToolNameCollisionError
from roomkit.providers.ai.base import AITool

if TYPE_CHECKING:
    from roomkit.tools.policy import ToolPolicy
    from roomkit.tools.timeout import ToolTimeouts

ToolServe = Callable[[dict[str, Any]], Any]
"""Serves one call from its arguments; may return an awaitable. The room is the
call's (``current_tool_room_id``, RFC §21.4)."""


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """What a realtime session of a room starts with: a pipeline's active agent."""

    system_prompt: str | None
    voice: str | None
    tools: list[dict[str, Any]] | None
    tool_policy: ToolPolicy | None = None
    """The active agent's tool policy: the session admits a tool only when
    the channel's policy and this one both do (RFC §19.5)."""


SessionSource = Callable[[str], Awaitable[SessionConfig | None]]
"""A realtime session's configuration for a room id, or ``None`` for the channel's."""


class ToolSource(StrEnum):
    """Who set a tool up on the channel."""

    CHANNEL = "channel"
    """One of the channel's own features: skills, Tool Search, the re-read, the planner."""

    ORCHESTRATION = "orchestration"
    """A strategy, a handoff, a delegation or a delegation's result capture."""


@dataclass(frozen=True, slots=True)
class ToolTraits:
    """What the channel's rules read of a tool."""

    exempt: bool = False
    """Only reads or unlocks: escapes the tool policy and skill gating (RFC §21.1)."""

    deferrable: bool = True
    """Tool Search may hide it behind its discovery tools."""

    always_declared: bool = False
    """Declared at every round and in every session of its scope, outside the
    catalogue whose size decides Tool Search, reported ``always`` (RFC §21.1)."""

    in_tool_memory: bool = True
    """Work the agent did: replayed in the room's later turns and re-revealed
    as a tool in use."""

    pure: bool = False
    """Reads what cannot change within a turn: an identical repeat says nothing new."""

    waits: bool = False
    """Waits on another agent or a person by design, under a bound of its own:
    the channel's default call bound does not apply to it (RFC §21.6)."""

    ends_turn: bool = False
    """Its served call ends the turn: the model is asked nothing after the
    round that made it, a delegation's result tool (RFC §23.3)."""


CHANNEL_TOOL_TRAITS: dict[str, ToolTraits] = {
    TOOL_ACTIVATE_SKILL: ToolTraits(exempt=True, deferrable=False, in_tool_memory=False),
    TOOL_READ_REFERENCE: ToolTraits(exempt=True, deferrable=False, in_tool_memory=False),
    TOOL_RUN_SCRIPT: ToolTraits(deferrable=False, in_tool_memory=False),
    TOOL_FIND_TOOLS: ToolTraits(exempt=True, deferrable=False, in_tool_memory=False, pure=True),
    TOOL_LIST_TOOLS: ToolTraits(exempt=True, deferrable=False, in_tool_memory=False, pure=True),
    # Only carries a call to the tool it names: the policy judges that tool,
    # at the gate, once the call is unwrapped.
    TOOL_CALL_TOOL: ToolTraits(exempt=True, deferrable=False, in_tool_memory=False),
    REREAD_TOOL: ToolTraits(exempt=True, deferrable=False, in_tool_memory=False),
    "plan_tasks": ToolTraits(deferrable=False),
}
"""The traits of each tool a channel serves itself, by its name."""

ORCHESTRATION_TRAITS = ToolTraits(deferrable=False, always_declared=True)
"""An orchestration tool: the agent is told to call it, so it is never hidden."""


@dataclass(frozen=True, slots=True)
class ToolEntry:
    """One tool a channel serves: its definition, its server, its origin, its traits."""

    definition: AITool
    serve: ToolServe | None
    """``None`` for a tool the channel runs on a path of its own (a realtime
    channel's skills and Tool Search)."""

    source: ToolSource
    traits: ToolTraits = field(default_factory=ToolTraits)
    declared_as: tuple[AITool, ...] = ()
    """The other declarations its server answers under its name (a
    pipeline agent's handoff, its targets its own): a session may declare
    them, never another tool under the name."""

    @property
    def name(self) -> str:
        return self.definition.name

    def declares(self, description: str, parameters: dict[str, Any]) -> bool:
        """Whether a declaration of *description* and *parameters* is one of
        this entry's own."""
        return any(
            (definition.description, definition.parameters) == (description, parameters)
            for definition in (self.definition, *self.declared_as)
        )


def schema_tool(schema: dict[str, Any]) -> AITool:
    """The definition a tool schema given as a dict describes."""
    return AITool(
        name=schema["name"],
        description=schema.get("description", ""),
        parameters=schema.get("parameters", {}),
        tags=list(schema.get("tags") or []),
    )


def tool_dict(definition: AITool) -> dict[str, Any]:
    """The declaration a realtime provider takes for *definition*."""
    return {
        "name": definition.name,
        "description": definition.description,
        "parameters": definition.parameters,
    }


def channel_tool(definition: AITool, serve: ToolServe | None) -> ToolEntry:
    """A tool the channel serves itself, with the traits its name carries."""
    traits = CHANNEL_TOOL_TRAITS.get(definition.name, ToolTraits())
    return ToolEntry(definition, serve, ToolSource.CHANNEL, traits)


def orchestration_tool(
    definition: AITool,
    serve: ToolServe,
    *,
    declared_as: Iterable[AITool] = (),
    **traits: bool,
) -> ToolEntry:
    """A tool orchestration sets up: declared always, never deferred."""
    return ToolEntry(
        definition,
        serve,
        ToolSource.ORCHESTRATION,
        replace(ORCHESTRATION_TRAITS, **traits),
        tuple(declared_as),
    )


@dataclass(frozen=True, slots=True)
class TurnOwnerTools:
    """The tools the room owner serves one turn of *channel_id* in *room_id*,
    as its request lists them, each served through the turn's link."""

    channel_id: str
    room_id: str
    entries: Mapping[str, ToolEntry]


_TURN_OWNER_TOOLS: contextvars.ContextVar[TurnOwnerTools | None] = contextvars.ContextVar(
    "roomkit_turn_owner_tools", default=None
)


def set_turn_owner_tools(tools: TurnOwnerTools | None) -> None:
    """Set the owner's tools of the turn the current context runs: an
    executor sets them for every turn it runs, so a turn submitted from inside
    another never reads that one's."""
    _TURN_OWNER_TOOLS.set(tools)


@dataclass(slots=True)
class _Held:
    entry: ToolEntry
    owner: object | None


@dataclass(slots=True)
class _Slot[T]:
    value: T
    owner: object | None


class ChannelRegistry:
    """The tools a channel serves, for all its rooms or one, and what
    orchestration installs on it per room."""

    def __init__(self, channel_id: str, host_names: Callable[[], Iterable[str]]) -> None:
        self._channel_id = channel_id
        # The names the host's own tools carry, read at each registration.
        self._host_names = host_names
        self._channel: dict[str, _Held] = {}
        self._rooms: dict[str, dict[str, _Held]] = {}
        self._session_source: _Slot[SessionSource] | None = None

    # -- Tools -------------------------------------------------------------

    def register(
        self, entry: ToolEntry, *, room_id: str | None = None, owner: object | None = None
    ) -> None:
        """Serve *entry* in *room_id*, or in every room when it is ``None``.

        Refuses a name something else already serves where the entry would be
        declared (:class:`ToolNameCollisionError`); the same *owner* registering the
        same name in the same scope again replaces its own entry.
        """
        self.check(entry, room_id=room_id, owner=owner)
        scope = self._channel if room_id is None else self._rooms.setdefault(room_id, {})
        scope[entry.name] = _Held(entry, owner)

    def register_all(
        self, entries: list[ToolEntry], *, room_id: str | None = None, owner: object
    ) -> None:
        """Serve every one of *entries*, or none when one of them is refused."""
        for entry in entries:
            self.check(entry, room_id=room_id, owner=owner)
        for entry in entries:
            self.register(entry, room_id=room_id, owner=owner)

    def check(
        self, entry: ToolEntry, *, room_id: str | None = None, owner: object | None = None
    ) -> None:
        """Refuse *entry* where something else already serves its name
        (:class:`ToolNameCollisionError`); *owner*'s own entry is replaceable."""
        held = (self._channel if room_id is None else self._rooms.get(room_id, {})).get(entry.name)
        if held is not None and owner is not None and held.owner is owner:
            return
        clash = self._clash(entry.name, room_id, host=entry.source is ToolSource.ORCHESTRATION)
        if clash is not None:
            raise ToolNameCollisionError(
                f"Tool {entry.name!r} is already {clash} on channel {self._channel_id!r}: "
                "rename one of them (RFC §21.1)"
            )

    def _clash(self, name: str, room_id: str | None, *, host: bool) -> str | None:
        """What already serves *name* where an entry for *room_id* would be declared.

        The host's tools count for an orchestration entry; a host tool under
        one of the channel's own names is refused where the host gives it.
        """
        if host and name in set(self._host_names()):
            return "one of the channel's own tools"
        if name in self._channel:
            return "served by the channel"
        if room_id is None:
            rooms = list(self._rooms.items())
        else:
            rooms = [(room_id, self._rooms.get(room_id, {}))]
        for rid, scope in rooms:
            if name in scope:
                return f"served in room {rid!r}"
        return None

    def own_names(self) -> set[str]:
        """The names the channel's own tools carry: the host's, and the ones
        the channel serves itself (never orchestration's)."""
        names = set(self._host_names())
        for scope in (self._channel, *self._rooms.values()):
            names.update(
                name
                for name, held in scope.items()
                if held.entry.source is not ToolSource.ORCHESTRATION
            )
        return names

    def refuse_host_names(self, names: Iterable[str | None]) -> None:
        """Refuse host tools given under a name orchestration serves in any room."""
        scopes = [self._channel, *self._rooms.values()]
        for name in names:
            if name is not None and any(
                name in scope and scope[name].entry.source is ToolSource.ORCHESTRATION
                for scope in scopes
            ):
                raise ToolNameCollisionError(
                    f"Tool {name!r} is already served by orchestration on channel "
                    f"{self._channel_id!r}: rename it (RFC §21.1)"
                )

    def unregister(self, name: str, *, room_id: str | None = None, owner: object) -> None:
        """Stop serving the entry *owner* registered under *name* in that scope."""
        scope = self._channel if room_id is None else self._rooms.get(room_id)
        if scope is None:
            return
        held = scope.get(name)
        if held is not None and held.owner is owner:
            del scope[name]
        if room_id is not None and not scope:
            self._rooms.pop(room_id, None)

    def room_tool_names(self, room_id: str) -> frozenset[str]:
        """The names of the tools served in *room_id* alone."""
        return frozenset(self._rooms.get(room_id, {}))

    def release(self, room_id: str, names: Iterable[str]) -> None:
        """Stop serving *names* in *room_id*: what a strategy's install set up
        there, which its uninstall takes back (RFC §19.7)."""
        scope = self._rooms.get(room_id)
        if scope is not None:
            for name in names:
                scope.pop(name, None)
            if not scope:
                self._rooms.pop(room_id, None)

    def lookup(self, name: str, room_id: str | None) -> ToolEntry | None:
        """The entry serving *name* in *room_id* (the channel's, outside any
        room); in a turn the backend runs, the owner's tools are the turn's."""
        turn = self._turn_owner_tools(room_id)
        if turn is None:
            return self._held_entry(name, room_id)
        entry = self._held_entry(name, room_id)
        if entry is not None and entry.source is not ToolSource.ORCHESTRATION:
            return entry
        return turn.entries.get(name)

    def entries(self, room_id: str | None, *, source: ToolSource | None = None) -> list[ToolEntry]:
        """The entries declared in *room_id*: the channel's, then the room's,
        each in the order they were registered; in a turn the backend runs,
        the owner's tools are the turn's, after the channel's own."""
        found = self._held_entries(room_id)
        turn = self._turn_owner_tools(room_id)
        if turn is not None:
            found = [e for e in found if e.source is not ToolSource.ORCHESTRATION]
            found.extend(turn.entries.values())
        return [e for e in found if source is None or e.source is source]

    def owner_entries(self, room_id: str) -> list[ToolEntry]:
        """The tools orchestration set up for *room_id*, which the room owner
        lists in each turn's request and serves (RFC §10.6.5)."""
        return [e for e in self._held_entries(room_id) if e.source is ToolSource.ORCHESTRATION]

    def owner_lookup(self, name: str, room_id: str) -> ToolEntry | None:
        """The tool orchestration set up under *name* for *room_id*, which
        serves a turn's call on the owner."""
        entry = self._held_entry(name, room_id)
        return entry if entry is not None and entry.source is ToolSource.ORCHESTRATION else None

    def _held_entry(self, name: str, room_id: str | None) -> ToolEntry | None:
        held = self._rooms.get(room_id, {}).get(name) if room_id is not None else None
        if held is None:
            held = self._channel.get(name)
        return held.entry if held is not None else None

    def _held_entries(self, room_id: str | None) -> list[ToolEntry]:
        held = [*self._channel.values()]
        if room_id is not None:
            held.extend(self._rooms.get(room_id, {}).values())
        return [h.entry for h in held]

    def _turn_owner_tools(self, room_id: str | None) -> TurnOwnerTools | None:
        """The owner's tools of the turn running here, when it is this
        channel's turn in *room_id*."""
        turn = _TURN_OWNER_TOOLS.get()
        if turn is None or turn.channel_id != self._channel_id or turn.room_id != room_id:
            return None
        return turn

    def names(
        self, room_id: str | None, where: Callable[[ToolTraits], bool] | None = None
    ) -> set[str]:
        """The names of the entries declared in *room_id* whose traits *where* admits."""
        return {e.name for e in self.entries(room_id) if where is None or where(e.traits)}

    def traits(self, name: str, room_id: str | None = None) -> ToolTraits | None:
        """The traits of the tool serving *name* in *room_id*, if an entry serves it."""
        entry = self.lookup(name, room_id)
        return entry.traits if entry is not None else None

    def bound(
        self, name: str, room_id: str | None, timeouts: ToolTimeouts, *, own: bool = False
    ) -> float | None:
        """The bound of a call to *name* in *room_id* (RFC §21.6): *timeouts*',
        unless the tool keeps a bound of its own, as its traits or *own* say."""
        traits = self.traits(name, room_id)
        return timeouts.for_call(name, waits=own or (traits is not None and traits.waits))

    # -- What orchestration installs per room ------------------------------

    def set_session_source(self, source: SessionSource, *, owner: object) -> None:
        """Read a new realtime session's configuration from *source*."""
        held = self._session_source
        if held is not None and held.owner is not owner:
            raise ValueError(
                f"Channel {self._channel_id!r} already reads its sessions' configuration "
                "from another source"
            )
        self._session_source = _Slot(source, owner)

    @property
    def session_source(self) -> SessionSource | None:
        """Where a new session's configuration comes from, when orchestration set one."""
        return self._session_source.value if self._session_source is not None else None
