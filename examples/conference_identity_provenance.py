"""Who a conference believes when it identifies a participant.

A participant the framework did not admit — a PSTN dial-in, an admission
arranged out of band — arrives under the SFU's own opaque identity, and the only
thing connecting it to a person is an address among the attributes its provider
attached (RFC §12.10.2). But one attribute map carries two very different
things: what the SFU established (the number the SIP trunk reported) and what a
client supplied when it joined. Only the first can found an identity.

This runs four arrivals against MockConferenceBackend and shows what each one
leaves on the room's roster:

1. a dial-in whose number the SFU asserts    → identified;
2. a participant that writes the same number → left unknown;
3. a backend that cannot tell the two apart  → left unknown;
4. the same backend on a deployment that has its own reason to trust it, and
   said so → identified.

Then it shows where the provider's attributes end up: under one key of the
Participant's metadata, provenance kept, never merged over what the integrator
put there — not even when a participant re-joins naming the same field.

Run with:
    uv run python examples/conference_identity_provenance.py
"""

from __future__ import annotations

import asyncio

from roomkit import (
    CONFERENCE_METADATA_KEY,
    MockConferenceBackend,
    Participant,
    RoomKit,
)
from roomkit.channels.conference import ConferenceChannel
from roomkit.identity.mock import MockIdentityResolver
from roomkit.models.identity import Identity

ROOM = "board-meeting"
DIAL_IN = "sip_15551234"
NUMBER = "+15551234"
ALICE = Identity(id="user-42", display_name="Alice", channel_addresses={"sms": [NUMBER]})


async def conference(
    trusts_unasserted: bool = False,
) -> tuple[RoomKit, MockConferenceBackend]:
    backend = MockConferenceBackend()
    kit = RoomKit(identity_resolver=MockIdentityResolver({NUMBER: ALICE}))
    kit.register_channel(
        ConferenceChannel(
            "conf",
            backend=backend,
            identity_trusts_unasserted_metadata=trusts_unasserted,
        )
    )
    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")
    return kit, backend


async def record_of(kit: RoomKit) -> Participant:
    record = await kit.store.get_participant(ROOM, DIAL_IN)
    assert record is not None
    return record


async def arrive(trusts_unasserted: bool = False, **joined: object) -> Participant:
    """Let one participant into a fresh conference, and hand back its record."""
    kit, backend = await conference(trusts_unasserted)
    await backend.simulate_participant_joined(ROOM, DIAL_IN, **joined)  # type: ignore[arg-type]
    record = await record_of(kit)
    await kit.close_room(ROOM)
    return record


def report(title: str, record: Participant) -> None:
    print(f"\n{title}")
    print(f"  identified as:       {record.identity_id or 'nobody — left unknown'}")
    print(f"  provider attributes: {record.metadata[CONFERENCE_METADATA_KEY]}")


async def who_gets_believed() -> None:
    # 1. The trunk reported the caller number, so the SFU asserts it.
    report(
        "A dial-in the SFU vouches for",
        await arrive(metadata={"sip.phoneNumber": NUMBER}),
    )

    # 2. The same number, written by the participant itself. Resolving on it
    #    would hand this caller Alice's Identity — and put Alice's identity_id
    #    on the record every later attribution reads.
    report(
        "A participant claiming to be that number",
        await arrive(client_metadata={"sip.phoneNumber": NUMBER}),
    )

    # 3. A backend that does not distinguish says so by leaving
    #    asserted_metadata null, and nothing is founded on what it surfaces.
    report(
        "A backend that cannot tell the two apart",
        await arrive(metadata={"sip.phoneNumber": NUMBER}, asserts_provenance=False),
    )

    # 4. ...unless the integrator knows something the framework cannot: a closed
    #    client fleet, or provenance established elsewhere. The policy is
    #    theirs; what the framework owes is that the safe reading is the default.
    report(
        "The same backend, on a deployment that trusts it",
        await arrive(
            trusts_unasserted=True,
            metadata={"sip.phoneNumber": NUMBER},
            asserts_provenance=False,
        ),
    )


async def what_the_record_keeps() -> None:
    """A conference is where strangers get to propose keys for your metadata."""
    kit, backend = await conference()
    await backend.simulate_participant_joined(ROOM, DIAL_IN, metadata={"sip.phoneNumber": NUMBER})

    # The integrator's own field, written after the caller was identified.
    identified = await record_of(kit)
    await kit.store.update_participant(
        identified.model_copy(update={"metadata": {**identified.metadata, "tier": "gold"}})
    )

    # The caller drops off and dials back in, this time naming that field.
    await backend.simulate_participant_left(ROOM, DIAL_IN)
    await backend.simulate_participant_joined(
        ROOM, DIAL_IN, metadata={"sip.phoneNumber": NUMBER}, client_metadata={"tier": "platinum"}
    )

    record = await record_of(kit)
    print("\nWhat a re-join may and may not touch")
    print(f"  integrator's own field: tier={record.metadata['tier']}")
    print(f"  what the caller said:   {record.metadata[CONFERENCE_METADATA_KEY]['unasserted']}")
    await kit.close_room(ROOM)


async def main() -> None:
    await who_gets_believed()
    await what_the_record_keeps()


if __name__ == "__main__":
    asyncio.run(main())
