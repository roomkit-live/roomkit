"""Shared event-visibility resolution.

A single source of truth for how an event's ``visibility`` scope maps to a
target channel binding, used by both the broadcast router and the streaming
target selector so the two can never drift — and, since RFC §7.5 rule 8, by
the reconstruction of a channel's history, which must answer the same question
one turn later that the broadcast answered at delivery.
"""

from __future__ import annotations

from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, Visibility
from roomkit.models.event import RoomEvent


def visibility_allows(visibility: str, target_binding: ChannelBinding) -> bool:
    """Return whether an event with ``visibility`` reaches ``target_binding``.

    ``visibility`` is either a well-known scope keyword (see
    :class:`~roomkit.models.enums.Visibility`) or a channel-id spec — a single
    channel id, or a comma-separated list of ids.
    """
    if visibility == Visibility.ALL:
        return True
    if visibility == Visibility.NONE:
        return False
    if visibility == Visibility.INTERNAL:
        # Framework-internal events (delegation, system, handoff) are never
        # delivered to channels — they live only in stored room history.
        return False
    if visibility == Visibility.TRANSPORT:
        return target_binding.category == ChannelCategory.TRANSPORT
    if visibility == Visibility.INTELLIGENCE:
        return target_binding.category == ChannelCategory.INTELLIGENCE
    if "," in visibility:
        allowed = {cid.strip() for cid in visibility.split(",") if cid.strip()}
        return target_binding.channel_id in allowed
    return target_binding.channel_id == visibility


def effective_visibility(event: RoomEvent, source_binding: ChannelBinding | None) -> str:
    """The scope that decides who may see ``event``.

    An event carries its own scope when the sender set one; otherwise the scope
    comes from the binding it was sent through. The broadcast router stamps the
    binding's scope onto its copy of the event before delivering
    (:meth:`EventRouter.plan`), but that stamp happens *after* the commit — the
    stored event keeps ``"all"``. So anything resolving visibility from storage
    (history rebuilt for a channel) MUST go back to the source binding, and only
    the event's own non-default scope overrides it.

    ``source_binding`` is ``None`` when the sending channel has since been
    detached: the event's own scope is then the whole answer (RFC §7.5 rule 8).
    Treating an unresolvable source as hidden would empty a room's context the
    moment a transport leaves, while every framework-internal event states its
    scope on the event itself and stays correctly hidden either way.
    """
    if event.visibility and event.visibility != Visibility.ALL:
        return event.visibility
    if source_binding is not None:
        return source_binding.visibility
    return Visibility.ALL


def visible_events(context: RoomContext, channel_id: str) -> list[RoomEvent]:
    """``context.recent_events`` as ``channel_id`` is allowed to know it.

    RFC §7.5 rule 8: an event visibility withheld from a channel at delivery
    MUST NOT come back to it through history the framework rebuilds on its
    behalf. Withholding an event at broadcast and returning it a turn later as
    context enforces nothing.

    A channel always keeps its **own** events. Rule 5 skips them at delivery
    because the channel produced them, not because it may not know them —
    dropping them here would erase an assistant's own turns from its own prompt
    whenever it is bound with a narrow visibility (the §7.4 assistant pattern).

    A channel with no binding in this room sees only what it produced itself.

    Answers the visibility half of §7.5 rule 1 only. Access is enforced where
    events are delivered (:meth:`EventRouter._filter_targets` drops WRITE_ONLY
    and NONE), which is why a channel that may not read never reaches a context
    rebuild in the first place; a caller reconstructing history by itself owes
    that check. READ_ONLY and muted channels do read, and keep their history.
    """
    bindings = {b.channel_id: b for b in context.bindings}
    reader = bindings.get(channel_id)
    if reader is None:
        return [e for e in context.recent_events if e.source.channel_id == channel_id]
    return [
        e
        for e in context.recent_events
        if e.source.channel_id == channel_id
        or visibility_allows(effective_visibility(e, bindings.get(e.source.channel_id)), reader)
    ]
