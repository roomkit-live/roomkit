"""An answer of the agent cut off by a barge-in, as the AI channel uses it (RFC §6.4).

A voice channel records the cut (RFC §12.3.13 step 2): an internal event holding
the text handed to speech and how long it played, naming the answer it cut by
its channel and the event that answer responds to. The AI channel marks that
answer as interrupted in the context its next turn reads, and hands a speak
policy the cut when the agent has not answered since.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from roomkit.models.enums import ChannelType, Visibility
from roomkit.models.event import TextContent
from roomkit.speaking.base import CutReply

if TYPE_CHECKING:
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent

CUT_MARK = "[You were interrupted while saying this: the person may not have heard the end of it.]"
"""What the context adds to an answer that was cut off."""


def cut_records(context: RoomContext, channel_id: str) -> dict[str | None, RoomEvent]:
    """The latest cut record of each of *channel_id*'s answers in *context*, by
    the event the answer responds to.

    Only a record a voice channel of the room wrote counts: internal, with no
    participant, from a channel bound to the room as voice. Metadata is anyone's
    to write, and a record taken at its word would put a sender's text in the
    agent's mouth; a malformed one is skipped rather than fail every turn of the
    room it stays in.
    """
    voices = {b.channel_id for b in context.bindings if b.channel_type == ChannelType.VOICE}
    records: dict[str | None, RoomEvent] = {}
    for event in context.recent_events:
        if _is_cut_record(event, channel_id, voices):
            records[event.metadata.get("answer_responds_to")] = event
    return records


def _is_cut_record(event: RoomEvent, channel_id: str, voices: set[str]) -> bool:
    metadata = event.metadata
    return (
        metadata.get("interrupted") is True
        and metadata.get("answer_channel_id") == channel_id
        and isinstance(metadata.get("answer_responds_to"), str | None)
        and event.visibility == Visibility.INTERNAL
        and event.source.participant_id is None
        and event.source.channel_type == ChannelType.VOICE
        and event.source.channel_id in voices
    )


def cut_answer_ids(
    events: Iterable[RoomEvent], records: dict[str | None, RoomEvent], channel_id: str
) -> set[str]:
    """The ids of the events that end each cut answer among *events*: the last
    event of *channel_id*'s answer to each event *records* names."""
    last: dict[str | None, str] = {}
    for event in events:
        if event.source.channel_id == channel_id and event.responds_to in records:
            last[event.responds_to] = event.id
    return set(last.values())


def cut_reply(
    recent: Iterable[RoomEvent], records: dict[str | None, RoomEvent], channel_id: str
) -> CutReply | None:
    """The agent's latest answer in *recent*, when it was cut off."""
    latest = None
    for event in recent:
        if event.source.channel_id == channel_id:
            latest = event
    if latest is None:
        return None
    record = records.get(latest.responds_to)
    if record is None or record.created_at < latest.created_at:
        return None
    text = record.content.body if isinstance(record.content, TextContent) else ""
    return CutReply(text=text, played_ms=_played_ms(record), at=record.created_at)


def _played_ms(record: RoomEvent) -> int:
    """The record's ``played_ms``, 0 when it is not a non-negative number."""
    value = record.metadata.get("played_ms")
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        return 0
    return int(value)
