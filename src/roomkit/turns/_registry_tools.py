"""The tools orchestration sets up for a kit's channels, served by the kit
(RFC §10.6.2, §10.6.5).

The room owner keeps them per channel and room, beside the channel's
declaration, never on a live channel object: a channel it holds by its
declaration alone has none. It lists them in each turn it submits and runs a
call's handler here (under the call's context, which the owner's calls
install).
"""

from __future__ import annotations

import inspect
from functools import partial
from typing import TYPE_CHECKING, Any

from roomkit.channels._tool_registry import ChannelRegistry
from roomkit.core.exceptions import UnservedToolCallError
from roomkit.turns.base import OwnerTools
from roomkit.turns.wire import OwnerTool

if TYPE_CHECKING:
    from collections.abc import Callable

    from roomkit.channels.base import Channel
    from roomkit.turns.wire import ToolCall, TurnRequest


class OwnerToolStore:
    """The tools orchestration set up for each channel the owner holds, by
    channel id: in every room it serves, or in one.

    Each channel's are kept in a registry of their own, which refuses a name
    something else serves where the tool would be declared: another of the
    owner's tools, or one of the channel's own tools when the owner holds the
    channel live (*channel_of*). For a channel held by its declaration alone,
    the executor refuses that clash at its turn (RFC §10.6.5).
    """

    def __init__(self, channel_of: Callable[[str], Channel | None]) -> None:
        self._channel_of = channel_of
        self._registries: dict[str, ChannelRegistry] = {}

    def of(self, channel_id: str) -> ChannelRegistry:
        """The registry of *channel_id*'s tools, made on first use."""
        registry = self._registries.get(channel_id)
        if registry is None:
            registry = ChannelRegistry(channel_id, partial(self._own_names, channel_id))
            self._registries[channel_id] = registry
        return registry

    def get(self, channel_id: str) -> ChannelRegistry | None:
        """The registry of *channel_id*'s tools, if any was set up."""
        return self._registries.get(channel_id)

    def registries(self) -> dict[str, ChannelRegistry]:
        """Each channel's registry, by channel id."""
        return dict(self._registries)

    def forget(self, channel_id: str) -> None:
        """Drop every tool set up for *channel_id*: the channel left the
        owner, and one registered again under its id gets its own setup."""
        self._registries.pop(channel_id, None)

    def _own_names(self, channel_id: str) -> set[str]:
        registry = getattr(self._channel_of(channel_id), "_registry", None)
        return registry.own_names() if isinstance(registry, ChannelRegistry) else set()


class RegistryOwnerTools(OwnerTools):
    """The owner's tools of the channels it holds, read in its store."""

    def __init__(self, store: OwnerToolStore) -> None:
        self._store = store

    def declared(self, channel_id: str, room_id: str) -> list[OwnerTool]:
        registry = self._registry(channel_id)
        if registry is None:
            return []
        return [
            OwnerTool(
                tool=entry.definition,
                waits_by_design=entry.traits.waits,
                ends_turn=entry.traits.ends_turn,
            )
            for entry in registry.owner_entries(room_id)
        ]

    async def serve(self, request: TurnRequest, call: ToolCall) -> Any:
        registry = self._registry(request.channel_id)
        entry = registry.owner_lookup(call.name, request.room_id) if registry else None
        if entry is None or entry.serve is None:
            raise UnservedToolCallError(f"No tool {call.name!r} is served by the room owner")
        answer = entry.serve(call.arguments)
        if inspect.isawaitable(answer):
            answer = await answer
        return answer

    def _registry(self, channel_id: str) -> ChannelRegistry | None:
        return self._store.get(channel_id)
