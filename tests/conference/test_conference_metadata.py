"""What a room keeps of a conference's provider attributes (RFC §12.10.2).

A conference is the one place where strangers write into a `Participant`'s
metadata: a dial-in chooses its own attributes, and on many SFUs so does any
client that joins. Two things therefore have to hold on the record — the
integrator's own fields cannot be overwritten by them, and what the SFU vouched
for stays distinguishable from what it did not, long after the arrival that
settled an identity.
"""

from __future__ import annotations

import pytest

from roomkit import CONFERENCE_METADATA_KEY, MockConferenceBackend, RoomKit
from roomkit.channels._conference_metadata import (
    ASSERTED_KEY,
    MAX_ATTRIBUTES,
    MAX_KEY_CHARS,
    MAX_VALUE_CHARS,
    UNASSERTED_KEY,
    provider_record,
    require_mintable_attributes,
)
from roomkit.channels.conference import ConferenceChannel
from roomkit.conference.models import ConferenceParticipant
from roomkit.models.participant import Participant

ROOM = "room-1"
DIAL_IN = "sip_15551234"
NUMBER = "+15551234"


async def _kit_with_channel() -> tuple[RoomKit, MockConferenceBackend]:
    backend = MockConferenceBackend()
    kit = RoomKit()
    kit.register_channel(ConferenceChannel("conf", backend=backend))
    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")
    return kit, backend


async def _record(kit: RoomKit, participant_id: str = DIAL_IN) -> Participant:
    record = await kit.store.get_participant(ROOM, participant_id)
    assert record is not None
    return record


def _participant(
    metadata: dict[str, object] | None = None,
    asserted: dict[str, object] | None = None,
) -> ConferenceParticipant:
    """A participant whose two bags are stated exactly, mock conventions aside."""
    return ConferenceParticipant(
        participant_id=DIAL_IN,
        metadata=dict(metadata or {}),
        asserted_metadata=None if asserted is None else dict(asserted),
    )


class TestWhatTheRecordKeeps:
    async def test_provider_attributes_never_overwrite_the_integrators(self) -> None:
        """The failure this exists to prevent: a participant that names its
        attribute ``tier`` and takes over a field the deployment relies on.
        """
        kit, backend = await _kit_with_channel()
        await kit.store.add_participant(
            Participant(id=DIAL_IN, room_id=ROOM, channel_id="conf", metadata={"tier": "gold"})
        )

        await backend.simulate_participant_joined(
            ROOM, DIAL_IN, client_metadata={"tier": "platinum"}
        )

        record = await _record(kit)
        assert record.metadata["tier"] == "gold"
        assert record.metadata[CONFERENCE_METADATA_KEY][UNASSERTED_KEY] == {"tier": "platinum"}

    async def test_an_arrival_writes_its_attributes_under_one_key(self) -> None:
        kit, backend = await _kit_with_channel()

        await backend.simulate_participant_joined(
            ROOM,
            DIAL_IN,
            metadata={"sip.phoneNumber": NUMBER},
            client_metadata={"nickname": "bob"},
        )

        record = await _record(kit)
        assert set(record.metadata) == {CONFERENCE_METADATA_KEY}
        assert record.metadata[CONFERENCE_METADATA_KEY] == {
            ASSERTED_KEY: {"sip.phoneNumber": NUMBER},
            UNASSERTED_KEY: {"nickname": "bob"},
        }

    async def test_a_participant_the_provider_said_nothing_about_gets_no_key(self) -> None:
        """Two empty bags on every framework-named participant would say the SFU
        reported something where it reported nothing.
        """
        kit, backend = await _kit_with_channel()

        await backend.simulate_participant_joined(ROOM, "p-alice")

        record = await _record(kit, "p-alice")
        assert record.metadata == {}

    async def test_a_backend_that_cannot_tell_vouches_for_nothing(self) -> None:
        kit, backend = await _kit_with_channel()

        await backend.simulate_participant_joined(
            ROOM, DIAL_IN, metadata={"sip.phoneNumber": NUMBER}, asserts_provenance=False
        )

        assert (await _record(kit)).metadata[CONFERENCE_METADATA_KEY] == {
            ASSERTED_KEY: {},
            UNASSERTED_KEY: {"sip.phoneNumber": NUMBER},
        }

    async def test_a_rejoin_refreshes_only_the_conference_key(self) -> None:
        kit, backend = await _kit_with_channel()
        await kit.store.add_participant(
            Participant(id=DIAL_IN, room_id=ROOM, channel_id="conf", metadata={"case": "42"})
        )

        await backend.simulate_participant_joined(ROOM, DIAL_IN, metadata={"sip.callID": "abc"})
        await backend.simulate_participant_left(ROOM, DIAL_IN)
        await backend.simulate_participant_joined(ROOM, DIAL_IN, metadata={"sip.callID": "def"})

        record = await _record(kit)
        assert record.metadata["case"] == "42"
        assert record.metadata[CONFERENCE_METADATA_KEY][ASSERTED_KEY]["sip.callID"] == "def"


class TestHowProvenanceIsKept:
    def test_the_two_bags_are_disjoint(self) -> None:
        record = provider_record(
            _participant(metadata={"a": "1", "b": "2"}, asserted={"a": "1"}),
        )

        assert record == {ASSERTED_KEY: {"a": "1"}, UNASSERTED_KEY: {"b": "2"}}

    def test_an_attribute_the_participant_now_claims_stops_being_asserted(self) -> None:
        """Provenance follows the latest observation, or a value the SFU once
        vouched for would keep vouching for whatever replaced it.
        """
        previous = provider_record(_participant(metadata={"a": "1"}, asserted={"a": "1"}))

        record = provider_record(_participant(metadata={"a": "2"}, asserted={}), previous)

        assert record == {ASSERTED_KEY: {}, UNASSERTED_KEY: {"a": "2"}}

    def test_an_attribute_the_sfu_now_vouches_for_stops_being_a_claim(self) -> None:
        previous = provider_record(_participant(metadata={"a": "1"}, asserted={}))

        record = provider_record(_participant(metadata={"a": "1"}, asserted={"a": "1"}), previous)

        assert record == {ASSERTED_KEY: {"a": "1"}, UNASSERTED_KEY: {}}

    def test_attributes_accumulate_across_connections(self) -> None:
        previous = provider_record(_participant(metadata={"a": "1"}, asserted={"a": "1"}))

        record = provider_record(_participant(metadata={"b": "2"}, asserted={"b": "2"}), previous)

        assert record[ASSERTED_KEY] == {"a": "1", "b": "2"}

    def test_a_record_written_by_something_else_is_not_trusted_to_be_shaped(self) -> None:
        """Whatever the store holds comes back, including from an older version
        or an integrator's own code. A re-join must not become a crash.
        """
        record = provider_record(_participant(metadata={"a": "1"}), "not a mapping")

        assert record == {ASSERTED_KEY: {}, UNASSERTED_KEY: {"a": "1"}}


class TestWhatIsBounded:
    def test_a_bag_stops_at_the_attribute_bound(self) -> None:
        flood = {f"k{index}": "v" for index in range(MAX_ATTRIBUTES + 10)}

        record = provider_record(_participant(metadata=flood))

        assert len(record[UNASSERTED_KEY]) == MAX_ATTRIBUTES

    def test_the_bound_keeps_what_was_seen_first(self) -> None:
        """So that flooding on re-join cannot evict what a participant was
        already carrying.
        """
        first = provider_record(_participant(metadata={"early": "kept"}))
        flood = {f"k{index}": "v" for index in range(MAX_ATTRIBUTES + 10)}

        record = provider_record(_participant(metadata=flood), first)

        assert record[UNASSERTED_KEY]["early"] == "kept"
        assert len(record[UNASSERTED_KEY]) == MAX_ATTRIBUTES

    def test_a_flood_of_claims_cannot_evict_what_the_sfu_asserted(self) -> None:
        """Each bag is bounded on its own, which is what keeps the address the
        identity was founded on out of reach of the participant carrying it.
        """
        flood = {f"k{index}": "v" for index in range(MAX_ATTRIBUTES + 10)}

        record = provider_record(
            _participant(
                metadata={"sip.phoneNumber": NUMBER, **flood}, asserted={"sip.phoneNumber": NUMBER}
            )
        )

        assert record[ASSERTED_KEY] == {"sip.phoneNumber": NUMBER}
        assert len(record[UNASSERTED_KEY]) == MAX_ATTRIBUTES

    def test_an_oversized_value_is_dropped_and_the_rest_kept(self) -> None:
        record = provider_record(
            _participant(metadata={"big": "x" * (MAX_VALUE_CHARS + 1), "small": "ok"})
        )

        assert record[UNASSERTED_KEY] == {"small": "ok"}

    def test_an_oversized_key_is_dropped(self) -> None:
        record = provider_record(_participant(metadata={"k" * (MAX_KEY_CHARS + 1): "v"}))

        assert record[UNASSERTED_KEY] == {}

    def test_a_value_no_store_could_write_is_dropped(self) -> None:
        record = provider_record(_participant(metadata={"obj": object(), "kept": 1}))

        assert record[UNASSERTED_KEY] == {"kept": 1}

    def test_a_structured_value_that_fits_survives(self) -> None:
        record = provider_record(_participant(metadata={"headers": {"X-Trace": "abc"}}))

        assert record[UNASSERTED_KEY] == {"headers": {"X-Trace": "abc"}}


class TestWhatMayBeMinted:
    """The same bound, facing out (RFC §12.10.3).

    A credential may carry attributes of the integrator's own, and emitting
    what this module would refuse to store promises a round trip that does not
    happen. It refuses where the inbound bound drops, because the party on this
    side is the integrator — the one that can be told.
    """

    def test_what_the_room_would_keep_is_mintable(self) -> None:
        require_mintable_attributes({"app.user": "user-42", "app.tier": "gold"})

    def test_nothing_is_a_valid_something(self) -> None:
        require_mintable_attributes({})

    def test_more_attributes_than_a_bag_holds_is_refused(self) -> None:
        flood = {f"k{index}": "v" for index in range(MAX_ATTRIBUTES + 1)}

        with pytest.raises(ValueError, match=f"at most {MAX_ATTRIBUTES} attributes"):
            require_mintable_attributes(flood)

    def test_a_key_the_room_would_drop_is_refused(self) -> None:
        with pytest.raises(ValueError, match=f"{MAX_KEY_CHARS} characters"):
            require_mintable_attributes({"k" * (MAX_KEY_CHARS + 1): "v"})

    def test_a_value_the_room_would_drop_is_refused(self) -> None:
        with pytest.raises(ValueError, match=f"{MAX_VALUE_CHARS}"):
            require_mintable_attributes({"big": "x" * (MAX_VALUE_CHARS + 1)})

    def test_a_value_that_is_not_a_string_is_refused(self) -> None:
        """An SFU's attribute map carries strings, so serializing is the
        integrator's — and then what comes back is what went out.
        """
        with pytest.raises(ValueError, match="carries strings"):
            require_mintable_attributes({"count": 3})  # type: ignore[dict-item]
