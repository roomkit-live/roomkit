"""What a room keeps of the attributes a conference's provider attached.

A ``Participant``'s ``metadata`` is the integrator's own map — a tier, a case
number, whatever the deployment puts there — and a conference is the one place
where strangers write into it. A dial-in's attributes are chosen by whoever
connects: merging them flat lets a participant overwrite a field the integrator
relies on, and lets it write as much as it likes.

So provider attributes live under one key of their own, in the two bags RFC
§12.10.2 separates them into — what the SFU asserts, and what it does not — and
what is written there is bounded. The provenance survives into the store on
purpose: what an identity was founded on has to remain answerable after the
fact, and a flat bag cannot answer it.

The bound faces both ways, which is why the mint's check lives here too: a
credential the channel issues may carry attributes of the integrator's own, and
emitting what this module would refuse to store would promise a round trip it
does not make.

See RFC sections 12.10.2 and 12.10.3.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roomkit.conference.models import ConferenceParticipant

logger = logging.getLogger("roomkit.channels.conference")


CONFERENCE_METADATA_KEY = "conference"
"""The single ``Participant.metadata`` key a conference writes under.

Everything the provider said about a participant is nested here::

    participant.metadata
    {
        "tier": "gold",                                    # the integrator's own
        "conference": {
            "asserted":   {"sip.phoneNumber": "+15551234"},  # the SFU vouches
            "unasserted": {"nickname": "bob"},               # the client said so
        },
    }
"""

ASSERTED_KEY = "asserted"
"""Sub-bag of attributes the SFU itself asserts."""

UNASSERTED_KEY = "unasserted"
"""Sub-bag of attributes the SFU did not vouch for."""

MAX_ATTRIBUTES = 32
"""How many attributes each bag keeps."""

MAX_KEY_CHARS = 128
"""How long an attribute name may be."""

MAX_VALUE_CHARS = 1024
"""How large a single attribute value may be, serialized."""


def require_mintable_attributes(attributes: Mapping[str, str]) -> None:
    """Refuse attributes a credential must not carry into a room.

    The same three limits :func:`_bounded` applies to what a room persists,
    applied to what a mint emits — a library that hands out a credential
    carrying what it would drop on the way back in is promising a round trip it
    does not make.

    It refuses where the inbound bound drops, and the difference is who is on
    the other end. An attribute that arrives over-long was written by a stranger
    the framework cannot address, so dropping it is the only move left. One that
    is *minted* over-long was written by the integrator, right here, in a call
    that can raise — and an attribute silently missing from a token is a bug
    they would otherwise meet in the browser, a deployment later.

    Values must be strings: they ride a provider's own attribute map, and every
    SFU that carries one carries strings. Anything richer is the integrator's to
    serialize, so that what comes back is what went out.

    Raises:
        ValueError: naming the attribute and the limit it failed.
    """
    if len(attributes) > MAX_ATTRIBUTES:
        raise ValueError(
            f"A conference credential carries at most {MAX_ATTRIBUTES} attributes, and this "
            f"mint asks for {len(attributes)}. The bound is the one the room applies to the "
            "provider attributes it persists: minting more would emit what a re-join could "
            "not store."
        )
    for key, value in attributes.items():
        if not isinstance(key, str) or len(key) > MAX_KEY_CHARS:
            raise ValueError(
                f"Conference attribute name {key!r} is not a string of at most "
                f"{MAX_KEY_CHARS} characters, which is what the room will persist when the "
                "SFU reports it back."
            )
        if not isinstance(value, str):
            raise ValueError(
                f"Conference attribute {key!r} is {type(value).__name__}, and an SFU's "
                "attribute map carries strings. Serialize it yourself, so that what comes "
                "back is what went out."
            )
        if not _fits(value):
            raise ValueError(
                f"Conference attribute {key!r} is longer than the {MAX_VALUE_CHARS} "
                "characters the room will persist when the SFU reports it back."
            )


def split_provenance(participant: ConferenceParticipant) -> tuple[dict[str, Any], dict[str, Any]]:
    """The provider's attributes as two disjoint bags: asserted, then the rest.

    ``asserted_metadata`` is a *subset* of ``metadata``, so the second bag is
    what is left once the first is taken out. A backend that asserts nothing —
    or that says it cannot tell (``None``) — puts everything in the second,
    which is what stops an unvouched attribute being read as an address.
    """
    asserted = dict(participant.asserted_metadata or {})
    unasserted = {key: value for key, value in participant.metadata.items() if key not in asserted}
    return asserted, unasserted


def provider_record(
    participant: ConferenceParticipant, previous: object = None
) -> dict[str, dict[str, Any]]:
    """What goes under :data:`CONFERENCE_METADATA_KEY` for this participant.

    *previous* is what the record already held there, if this is a participant
    the room has met before. Attributes accumulate across connections — a
    re-join reports what the SFU knows now, not the whole history — but
    provenance follows the latest observation: an attribute the participant now
    supplies itself stops being asserted, and one the SFU now vouches for stops
    being merely claimed. The two bags stay disjoint either way.

    Each bag is bounded on its own, and the bound keeps what was seen first.
    Both matter: a participant that floods its own attributes on re-join can
    neither evict what the SFU asserted about it — a different bag — nor what
    it was seen carrying earlier.
    """
    asserted, unasserted = split_provenance(participant)
    prior_asserted = _bag(previous, ASSERTED_KEY)
    prior_unasserted = _bag(previous, UNASSERTED_KEY)

    merged_asserted = {**prior_asserted, **asserted}
    for key in unasserted:
        merged_asserted.pop(key, None)
    merged_unasserted = {**prior_unasserted, **unasserted}
    for key in merged_asserted:
        merged_unasserted.pop(key, None)

    participant_id = participant.participant_id
    return {
        ASSERTED_KEY: _bounded(merged_asserted, participant_id=participant_id, bag=ASSERTED_KEY),
        UNASSERTED_KEY: _bounded(
            merged_unasserted, participant_id=participant_id, bag=UNASSERTED_KEY
        ),
    }


def _bag(record: object, key: str) -> dict[str, Any]:
    """One sub-bag of a record written earlier, or nothing if it is not there.

    Read defensively rather than trusted: what comes back is whatever the store
    holds, and a room whose participants were written by an older version — or
    by an integrator's own code — must not turn a re-join into a crash.
    """
    if not isinstance(record, Mapping):
        return {}
    bag = record.get(key)
    if not isinstance(bag, Mapping):
        return {}
    return {name: value for name, value in bag.items() if isinstance(name, str)}


def _bounded(attributes: Mapping[str, Any], *, participant_id: str, bag: str) -> dict[str, Any]:
    """*attributes*, cut down to what a room agrees to persist.

    Three limits, in the order a value fails them: the name must be a string
    that fits, the value must survive being serialized and fit, and the bag
    holds :data:`MAX_ATTRIBUTES` of them. Serializing is the test because a
    Postgres store will do it too — an attribute that cannot be written is
    better dropped at the boundary it arrived at than at the one it would
    otherwise fail.
    """
    kept: dict[str, Any] = {}
    dropped = 0
    for key, value in attributes.items():
        if len(kept) >= MAX_ATTRIBUTES:
            dropped += 1
            continue
        if not isinstance(key, str) or len(key) > MAX_KEY_CHARS or not _fits(value):
            dropped += 1
            continue
        kept[key] = value
    if dropped:
        logger.debug(
            "Dropped %d %s conference attribute(s) for participant %s: over the "
            "%d-attribute, %d-character-key or %d-character-value bound",
            dropped,
            bag,
            participant_id,
            MAX_ATTRIBUTES,
            MAX_KEY_CHARS,
            MAX_VALUE_CHARS,
        )
    return kept


def _fits(value: Any) -> bool:
    """Whether a value is storable and small enough to store."""
    try:
        return len(json.dumps(value)) <= MAX_VALUE_CHARS
    except (TypeError, ValueError):
        return False
