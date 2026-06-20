"""Shared event-visibility resolution.

A single source of truth for how an event's ``visibility`` scope maps to a
target channel binding, used by both the broadcast router and the streaming
target selector so the two can never drift.
"""

from __future__ import annotations

from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelCategory, Visibility


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
