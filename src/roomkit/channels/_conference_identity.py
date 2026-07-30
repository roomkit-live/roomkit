"""Who a conference participant is, when the framework did not name them.

A participant the framework admitted arrives already named: the id it was
minted under comes back from the backend and the Room participant behind it is
the one the integrator created. A participant it did not admit — a PSTN
dial-in, an admission arranged out of band — arrives under the backend's own
opaque identity, and the only thing connecting it to a person is the address
its provider attached: a caller number, above all.

Resolving that address is what this does, and it does it when the participant
*arrives* rather than when it first speaks. Someone can sit in a meeting for an
hour without publishing a word, and leaving them unidentified until they do
leaves every hook, every roster read and every disclosure obligation looking at
an unknown.

The awkward part, stated plainly: ``IdentityResolver`` resolves from an inbound
message, and an arrival is not one. So a message is built to carry the address
to the resolver, and nothing else is done with it — it is never processed, and
never reaches a room. What an arrival deliberately does *not* do is the rest of
Section 11: no ``ON_IDENTITY_*`` hooks, no challenge, no rejection. Those act on
a message, by holding it, answering it or refusing it, and here there is none to
hold. The participant's first utterance goes through the inbound pipeline like
any other, where all of that applies.

See RFC section 12.10.2.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from roomkit.channels._conference_metadata import split_provenance
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType, EventType, IdentificationStatus
from roomkit.models.event import SystemContent

if TYPE_CHECKING:
    from roomkit.conference.models import ConferenceParticipant
    from roomkit.core.framework import RoomKit
    from roomkit.models.identity import IdentityResult

logger = logging.getLogger("roomkit.channels.conference")


CONFERENCE_ADDRESS_KEYS: tuple[str, ...] = (
    "sip.phoneNumber",
    "phone_number",
    "phoneNumber",
    "caller_id",
    "callerId",
    "from_number",
)
"""Participant-attribute keys read as a caller's address, most specific first.

Providers name the same fact differently — ``sip.phoneNumber`` is LiveKit's —
and the list is ordered so that a provider's own key wins over a generic one it
also happens to carry.

What is *not* here matters as much as what is. The number a caller dialled
(``sip.trunkPhoneNumber`` and its equivalents) is the conference's number, not
the caller's: reading it would identify every dial-in as the same person, which
is worse than identifying none of them. And ``from`` is a SIP header name, not
a provider's key — generic enough that a value found under it says nothing
about where it came from. An integrator whose provider really does report the
caller there passes it in ``address_keys``.
"""

CONFERENCE_UNASSERTED_METADATA_KEY = "conference_unasserted"
"""Where an arrival's unvouched attributes travel on the message to a resolver.

Nested rather than flat, so that a resolver reading ``metadata["phone_number"]``
reads something the SFU asserted. What the participant said about itself is
still there — a resolver that wants it can have it — but it takes a deliberate
look to find, which is the difference between using it and being caught by it.
"""


class ConferenceIdentity:
    """Runs identity resolution for an arriving conference participant.

    Inert until a framework arrives: a channel is constructed before it is
    registered, and the resolver belongs to the framework.

    Answers only the question "who is this" — writing the answer onto the Room
    participant belongs to :class:`ConferenceRoster`, which owns that record.
    """

    def __init__(
        self,
        channel_id: str,
        address_keys: Sequence[str] | None = None,
        *,
        trust_unasserted: bool = False,
    ) -> None:
        self._channel_id = channel_id
        self._address_keys = (
            tuple(address_keys) if address_keys is not None else CONFERENCE_ADDRESS_KEYS
        )
        self._trust_unasserted = trust_unasserted
        self._framework: RoomKit | None = None

    def set_framework(self, framework: RoomKit) -> None:
        """Wire the framework whose resolver this consults, once it is known."""
        self._framework = framework

    @property
    def active(self) -> bool:
        """Whether resolution is going to happen for arrivals on this channel.

        Two configured facts, and the framework answers both at once: that a
        resolver exists, and that ``identity_channel_types`` did not exclude
        conferences from it. The second matters as much as the first — an
        integrator who restricted resolution to SMS did so to stop addresses
        leaving for the resolver, and a dial-in's caller number is exactly such
        an address.

        A per-channel fact, and cheap to read, so a caller can answer "is any of
        this going to happen" before doing anything that costs — reading the
        store, most of all, which every arrival would otherwise pay for on a
        deployment that configured no resolution.
        """
        framework = self._framework
        return framework is not None and framework.identity_enabled_for(ChannelType.CONFERENCE)

    def address_of(self, participant: ConferenceParticipant) -> str | None:
        """The first address the *SFU asserts*, in priority order.

        Which key carries the address is the second question; the first is who
        put the value there (RFC §12.10.2). An attribute a participant's own
        client supplied is a claim about itself: a caller writing its own
        ``phone_number`` and resolved on it reaches whatever Identity that
        number belongs to — someone else's — and the Participant then carries
        the victim's ``identity_id`` on the record every later attribution
        reads. So only ``asserted_metadata`` is read, and a backend that says
        it cannot tell the two apart yields no address at all.

        Provenance outranks specificity: an asserted address on a generic key
        beats an unasserted one on the provider's own key, because an attacker
        chooses the key and never the provenance. Only where the integrator
        said to trust unvouched attributes does the wider bag get read, and
        only after the asserted one had nothing.

        Only strings count. A provider that puts a number in as an integer has
        dropped the leading ``+`` on the way, and stringifying it would hand the
        resolver an address that silently fails to match the one the same person
        texts from.
        """
        asserted = participant.asserted_metadata
        if asserted:
            address = _first_address(asserted, self._address_keys)
            if address is not None:
                return address
        if self._trust_unasserted:
            return _first_address(participant.metadata, self._address_keys)
        return None

    async def resolve(
        self, room_id: str, participant: ConferenceParticipant
    ) -> IdentityResult | None:
        """Identify an arriving participant from the address its provider attached.

        Returns ``None`` — leaving the participant UNKNOWN — when resolution is
        not enabled for this channel, when no address was found, or when
        resolution did not complete. The middle case is the one the RFC is
        explicit about: with no address, the channel does *not* fall back to
        resolving on the opaque backend identity, which no resolver could match
        anyway and which would turn "we don't know" into a lookup that looks
        like it happened.

        The first case is re-checked here rather than left to :attr:`active`,
        which the arrival path happens to consult first: a configuration that
        says no must say no to every caller, not only to the one that asked
        politely beforehand.
        """
        framework = self._framework
        if framework is None or not self.active:
            return None
        resolver = framework.identity_resolver
        if resolver is None:
            return None
        address = self.address_of(participant)
        if address is None:
            return None

        context = await framework._build_context(room_id)
        message = self._arrival_message(room_id, participant, address)
        try:
            return await asyncio.wait_for(
                resolver.resolve(message, context),
                timeout=framework.identity_timeout,
            )
        except TimeoutError:
            logger.warning(
                "Identity resolution timed out after %.1fs for conference participant %s",
                framework.identity_timeout,
                participant.participant_id,
                extra={"room_id": room_id, "channel_id": self._channel_id},
            )
            await framework._emit_framework_event(
                "identity_timeout",
                room_id=room_id,
                channel_id=self._channel_id,
                data={"timeout": framework.identity_timeout},
            )
            return None
        except Exception:
            # A resolver that raises must not keep someone out of a meeting, and
            # letting it through would do exactly that: the arrival is a backend
            # callback, whose exceptions ConferenceBackend._emit logs and drops,
            # so the roster write further down this path would never run and the
            # participant would exist in no room at all. Unidentified is where it
            # started; invisible is worse than where it started.
            logger.exception(
                "Identity resolution failed for conference participant %s",
                participant.participant_id,
                extra={"room_id": room_id, "channel_id": self._channel_id},
            )
            return None

    def _arrival_message(
        self, room_id: str, participant: ConferenceParticipant, address: str
    ) -> InboundMessage:
        """The arrival, in the shape a resolver reads.

        ``sender_id`` is the address rather than the backend's identity, which
        is the whole point: resolvers match on the address a person is reachable
        at, so a caller dialling into a conference reaches the same Identity it
        would have reached by texting the room.

        The opaque identity is not thrown away — it travels as ``external_id``,
        with the provider's attributes on ``metadata``, so a resolver that wants
        more than the address has all of it. Typed as the participant-joined
        event it describes rather than as a message, because it is not one and
        nothing will ever deliver it.

        What is flat there is what the channel itself would act on: the
        attributes the SFU asserted, or every attribute where the integrator
        said to trust them. The rest travels nested under
        :data:`CONFERENCE_UNASSERTED_METADATA_KEY`, present but never mistakable
        for a fact — a resolver matching on ``metadata["phone_number"]`` is
        matching on something the provider vouched for.
        """
        return InboundMessage(
            channel_id=self._channel_id,
            sender_id=address,
            event_type=EventType.PARTICIPANT_JOINED,
            external_id=participant.participant_id,
            content=SystemContent(
                body=f"Participant {participant.participant_id} joined the conference",
                code="conference_participant_joined",
                data={"room_id": room_id, "participant_id": participant.participant_id},
            ),
            metadata=self._resolver_metadata(participant),
        )

    def _resolver_metadata(self, participant: ConferenceParticipant) -> dict[str, Any]:
        """The provider's attributes, arranged so provenance survives the trip."""
        asserted, unasserted = split_provenance(participant)
        trusted = dict(participant.metadata) if self._trust_unasserted else asserted
        metadata: dict[str, Any] = dict(trusted)
        if unasserted:
            metadata[CONFERENCE_UNASSERTED_METADATA_KEY] = dict(unasserted)
        return metadata


def _first_address(metadata: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    """The first key in *keys* carrying a non-empty string in *metadata*."""
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def participant_update(identity: IdentityResult | None) -> dict[str, Any]:
    """What an identity result changes on a Room participant, if anything.

    Deliberately the same mapping the inbound pipeline applies, so a dial-in and
    a text message from the same person produce the same record:

    - IDENTIFIED links the participant to the Identity and takes its name.
    - AMBIGUOUS and PENDING both become a pending identification carrying the
      candidate ids, which is what ``_create_pending_participant`` does.
    - UNKNOWN, REJECTED and CHALLENGE_SENT change nothing. The last two are
      answers to a message — there is none here, so there is nothing to refuse
      and no challenge anyone could have replied to.
    """
    if identity is None:
        return {}
    if identity.status is IdentificationStatus.IDENTIFIED and identity.identity is not None:
        update: dict[str, Any] = {
            "identification": IdentificationStatus.IDENTIFIED,
            "identity_id": identity.identity.id,
        }
        if identity.identity.display_name:
            update["display_name"] = identity.identity.display_name
        return update
    if identity.status in (IdentificationStatus.AMBIGUOUS, IdentificationStatus.PENDING):
        return {
            "identification": IdentificationStatus.PENDING,
            "candidates": [candidate.id for candidate in identity.candidates] or None,
        }
    return {}
