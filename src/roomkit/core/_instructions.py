"""What the pipeline owns of an instruction (RFC §5.2, §9.4, §10.1.1).

Only the application issues an instruction, addressed to an agent the kit
registers. The pipeline alone writes which instruction produced a response:
``metadata.instruction``, a fingerprint of the instruction's text, never the
text. The key is reserved: the pipeline writes it on a response to an
instruction, whatever the channel wrote there, and removes it from every
other event, whoever wrote it (a parser, the caller's metadata, a channel, a
hook), so no event claims an instruction that did not produce it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from roomkit._media import text_form
from roomkit.models.delivery import STANDALONE
from roomkit.models.enums import ChannelCategory, EventType
from roomkit.models.event import RoomEvent
from roomkit.models.response_metadata import TURNS_KEY

if TYPE_CHECKING:
    from roomkit.channels.base import Channel

FINGERPRINT = "instruction"
"""The metadata key under which a response records the instruction that
produced it (RFC §10.1.1 step 6)."""


def names_an_agent(addressed_to: Iterable[str] | None, channels: Mapping[str, Channel]) -> bool:
    """Whether *addressed_to* names a channel the kit registers as an
    intelligence channel, as an instruction's address must (RFC §10.1.1
    step 1). Its other names ask nobody."""
    return any(
        (channel := channels.get(channel_id)) is not None
        and channel.category == ChannelCategory.INTELLIGENCE
        for channel_id in addressed_to or ()
    )


def fingerprint(instruction: RoomEvent) -> dict[str, Any]:
    """What a response records of *instruction*: the lowercase hex SHA-256 of
    the UTF-8 text its content reads as (RFC §6.6) and that text's length in
    Unicode code points."""
    text = text_form(instruction.content)
    return {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "length": len(text)}


def without_fingerprint(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """*metadata* without the reserved key, whoever wrote it."""
    return {key: value for key, value in metadata.items() if key != FINGERPRINT}


def response_metadata(metadata: Mapping[str, Any], trigger: RoomEvent) -> dict[str, Any]:
    """*metadata* of a response to *trigger* as the pipeline commits it: the
    trigger's fingerprint when it is an instruction, in place of whatever the
    channel wrote under the key, and no such key otherwise; never a value a
    turn wrote under ``turns``, the framework's key (RFC §6.7)."""
    kept = without_fingerprint(metadata)
    kept.pop(TURNS_KEY, None)
    if trigger.type == EventType.INSTRUCTION:
        kept[FINGERPRINT] = fingerprint(trigger)
    return kept


def kept_metadata(original: RoomEvent, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """*metadata*, a hook's replacement of *original*'s, with what the
    pipeline wrote put back (RFC §9.4): the fingerprint or its absence, and
    an instruction's ``standalone`` word or its absence. The hook's other
    keys stand."""
    owned = (FINGERPRINT, STANDALONE) if original.type == EventType.INSTRUCTION else (FINGERPRINT,)
    kept = {key: value for key, value in metadata.items() if key not in owned}
    kept.update({key: original.metadata[key] for key in owned if key in original.metadata})
    return kept
