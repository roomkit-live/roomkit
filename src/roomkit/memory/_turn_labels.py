"""The labels a memory's turns are read with, for the lines a summarizer reads
and for what a budget counts (RFC §6.4, §20.3)."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Mapping, Sequence

from roomkit.channels._speaker import (
    labelled_lines,
    labels_of,
    people_taking_part,
    several_speakers,
    turn_speakers,
)
from roomkit.memory.token_estimator import estimate_tokens, extract_event_text
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType, EventType
from roomkit.models.event import RoomEvent


def conversation_labels(
    context: RoomContext, turns: Sequence[RoomEvent], channel_id: str | None
) -> dict[str, str | None]:
    """Each of *turns*' label, or none when they hold one speaker and fewer
    than two people take part: the conversation's threshold, over every turn
    (a turn with no text, *channel_id*'s agent's own and the application's
    instruction aside), RFC §6.4."""
    speakers = turn_speakers(turns, context)
    counted = (speakers.get(e.id) for e in turns if _counts(e, channel_id))
    people = people_taking_part(context, channel_id)
    return labels_of(speakers) if several_speakers(counted, people) else {}


def labels_sent(
    context: RoomContext,
    events: Sequence[RoomEvent],
    current: RoomEvent,
    channel_id: str | None,
) -> dict[str, str]:
    """The label each participant's turn among *events* and *current* is
    sent with (RFC §6.4): none for *channel_id*'s agent's own turns, and none
    at all in a conversation that is not labelled (a one-to-one one)."""
    turns = list({e.id: e for e in (*events, current)}.values())
    labels = conversation_labels(context, turns, channel_id)
    return {
        e.id: label for e in turns if (label := labels.get(e.id)) and not own_turn(e, channel_id)
    }


def label_cost(event: RoomEvent, labels: Mapping[str, str] | None) -> int:
    """What *event*'s label adds to it where the conversation labels it
    (*labels*, :func:`labels_sent`): the label and the JSON string each of
    its lines is sent as (RFC §6.4, §20.3); nothing for a turn sent as typed."""
    label = labels.get(event.id) if labels else None
    text = extract_event_text(event) if label else ""
    if not label or not text:
        return 0
    key = (hashlib.blake2b(text.encode(), digest_size=16).digest(), label)
    # Taken out and put back, never moved in place: a read on another thread
    # may drop the key in between.
    cost = _LABEL_COSTS.pop(key, None)
    if cost is None:
        cost = max(0, estimate_tokens(labelled_lines(text, label)) - estimate_tokens(text))
    _LABEL_COSTS[key] = cost
    while len(_LABEL_COSTS) > _LABEL_COSTS_KEPT:
        _LABEL_COSTS.popitem(last=False)
    return cost


_LABEL_COSTS: OrderedDict[tuple[bytes, str], int] = OrderedDict()
"""What a label adds to a text, by the text's digest and the label: a
memory reads the same turns at every turn of a room. Keyed by digest, so no
text, a deleted turn's included, is kept."""

_LABEL_COSTS_KEPT = 1024


def own_turn(event: RoomEvent, channel_id: str | None) -> bool:
    """Whether *event* is *channel_id*'s agent's own turn."""
    if event.source.channel_type != ChannelType.AI:
        return False
    return channel_id is None or event.source.channel_id == channel_id


def _counts(event: RoomEvent, channel_id: str | None) -> bool:
    """Whether *event* counts toward the threshold, as the conversation's
    turns do."""
    if own_turn(event, channel_id) or event.type == EventType.INSTRUCTION:
        return False
    return bool(extract_event_text(event).strip())
