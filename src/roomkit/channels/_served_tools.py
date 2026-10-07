"""What a host tool becomes under a name the channel serves itself (RFC §21.1).

A channel serves some tools itself (skill activation, Tool Search, the eviction
re-read, sandbox commands). A tool of the host under one of those names would
be declared with the host's schema and served by the channel, so it is refused
when given at construction and not declared when it arrives later. And no name
is declared twice: a provider rejects a duplicate name; nor one no vendor
accepts, refused at definition as ``AITool`` refuses it (RFC §6.7).

Shared by every channel kind whatever the shape of its tool definitions (an
``AITool``, a realtime tool dict).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Container, Iterable, Sized
from typing import TYPE_CHECKING

from roomkit.channels._tool_search_constants import TOOL_SEARCH_INFRA_TOOL_NAMES
from roomkit.providers.ai.base import some_vendor_accepts_tool_name

if TYPE_CHECKING:
    from roomkit.voice.realtime.provider import RealtimeVoiceProvider
    from roomkit.voice.realtime.reasoning import ReasoningBackend

logger = logging.getLogger("roomkit.channels.tools")


def refuse_host_tools(
    names: Iterable[str | None], served: Container[str], channel_id: str
) -> None:
    """Refuse the host tools a door is given, as every door does (RFC §6.7,
    §21.1): a name no vendor accepts, one given twice, or one the channel or
    the person's tools serve (*served*)."""
    names = list(names)
    refuse_unnamable(names, channel_id)
    refuse_given_twice(names, channel_id)
    refuse_served_names(names, served, channel_id)


def refuse_backend_names(
    names: Iterable[str | None], backend: ReasoningBackend | None, channel_id: str
) -> None:
    """Refuse a session tool declared under a name the channel's reasoning
    backend serves itself: its agent would answer the call instead of the
    tool's handler, outside the channel's gate (RFC §12.4.1)."""
    if backend is None:
        return
    taken = backend.served_names()
    for name in names:
        if name in taken:
            raise ValueError(
                f"Tool {name!r} on channel {channel_id!r} is served by its reasoning "
                f"backend itself (RFC §12.4.1); give the tool another name"
            )


def refuse_served_names(
    names: Iterable[str | None], served: Container[str], channel_id: str
) -> None:
    """Refuse a host tool given at construction under a name the channel serves."""
    for name in names:
        if name is None or name not in served:
            continue
        hint = " or pass tool_search=False" if name in TOOL_SEARCH_INFRA_TOOL_NAMES else ""
        raise ValueError(
            f"Tool {name!r} is a tool channel {channel_id!r} serves itself: "
            f"rename it{hint} (RFC §21.1)"
        )


def refuse_unnamable(names: Iterable[str | None], channel_id: str) -> None:
    """Refuse a host tool given under a name no vendor accepts, as ``AITool``
    refuses it at definition (RFC §6.7); a tool without a name (a provider's
    native tool) has none to check."""
    for name in names:
        if name is not None and not some_vendor_accepts_tool_name(name):
            raise ValueError(
                f"Tool {name!r} given to channel {channel_id!r} is accepted by no provider: "
                "use letters, digits, '_', '.', ':' or '-' (RFC §6.7)"
            )


def refuse_given_twice(names: Iterable[str | None], channel_id: str) -> None:
    """Refuse a host tool given at construction under a name another already
    carries: declared once and served by the other, the model would call one
    tool's schema on the other's server (RFC §21.1)."""
    seen: set[str] = set()
    for name in names:
        if name is None:
            continue
        if name in seen:
            raise ValueError(
                f"Tool {name!r} is given twice to channel {channel_id!r}: two tools "
                "cannot serve one name, rename one of them (RFC §21.1)"
            )
        seen.add(name)


def warn_tools_uncallable(
    given: Sized | None, what: str, provider: RealtimeVoiceProvider, channel_id: str
) -> None:
    """Log that the tools or skills (*what*) given to a channel are declared
    to no session: its provider's model calls no tool (RFC §12.4)."""
    if given and not provider.supports_tools:
        logger.warning(
            "Channel %s: %s cannot call tools; the %d %s given are declared to no session",
            channel_id,
            provider.name,
            len(given),
            what,
        )


class CollisionLog:
    """The collisions a channel reported, each once: a wiring diagnostic, not a turn event."""

    def __init__(self, channel_id: str) -> None:
        self._channel_id = channel_id
        self._reported: set[tuple[str, str]] = set()

    def served(self, name: str) -> None:
        self._once(
            name, "served", "Channel %s does not declare the host's %r: it serves it itself"
        )

    def orchestrated(self, name: str) -> None:
        self._once(
            name,
            "orchestrated",
            "Channel %s does not declare another tool under %r: orchestration serves that name",
        )

    def duplicate(self, name: str) -> None:
        self._once(
            name, "duplicate", "Channel %s declares %r once: it was given twice, the later is kept"
        )

    def _once(self, name: str, kind: str, message: str) -> None:
        if (name, kind) in self._reported:
            return
        self._reported.add((name, kind))
        logger.warning(message, self._channel_id, name)


def declared_once[T](
    tools: Iterable[T],
    name_of: Callable[[T], str | None],
    served: Container[str],
    log: CollisionLog,
) -> list[T]:
    """*tools* with none under a served name, and each name once, the later kept.

    For the tools that arrive with a turn or a session, which come too late to
    be refused (RFC §21.1): the later is kept, as the last word of whoever gave
    them. A tool without a name is kept as it is: nothing can collide with it.
    """
    kept: dict[object, T] = {}
    for index, tool in enumerate(tools):
        name = name_of(tool)
        if name is None:
            kept[("unnamed", index)] = tool
            continue
        if name in served:
            log.served(name)
            continue
        if name in kept:
            log.duplicate(name)
            del kept[name]
        kept[name] = tool
    return list(kept.values())


def dict_tool_name(tool: object) -> str | None:
    """The name of a tool given as a dict (a realtime declaration), if it has one."""
    name = tool.get("name") if isinstance(tool, dict) else None
    return name if isinstance(name, str) else None
