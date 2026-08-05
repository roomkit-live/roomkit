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

And last, the same boundary in the other direction: what a mint may send along
with an identity, so that an integrator's own clients can tell who a tile
belongs to without the identity becoming a format to parse.

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
BROWSER = "p-alice"
NUMBER = "+15551234"
ALICE = Identity(id="user-42", display_name="Alice", channel_addresses={"sms": [NUMBER]})


async def conference(
    trusts_unasserted: bool = False,
) -> tuple[RoomKit, ConferenceChannel, MockConferenceBackend]:
    backend = MockConferenceBackend()
    channel = ConferenceChannel(
        "conf",
        backend=backend,
        identity_trusts_unasserted_metadata=trusts_unasserted,
    )
    kit = RoomKit(identity_resolver=MockIdentityResolver({NUMBER: ALICE}))
    kit.register_channel(channel)
    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")
    return kit, channel, backend


async def record_of(kit: RoomKit, participant_id: str = DIAL_IN) -> Participant:
    record = await kit.store.get_participant(ROOM, participant_id)
    assert record is not None
    return record


async def arrive(trusts_unasserted: bool = False, **joined: object) -> Participant:
    """Let one participant into a fresh conference, and hand back its record."""
    kit, _, backend = await conference(trusts_unasserted)
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
    kit, _, backend = await conference()
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


async def what_the_mint_may_send_along() -> None:
    """The other direction: what the room gets to write into the conference.

    A participant id is a *channel* identity, opaque on purpose — the person
    behind it is carried by `identity_id`, which never leaves the room. So an
    integrator's own client has nothing to put on a tile, unless the mint says
    so: `attributes` is the field that travels beside the identity, per mint,
    and only when it is asked for (RFC §12.10.3).
    """
    kit, channel, backend = await conference()
    await kit.ensure_participant(ROOM, "conf", BROWSER, display_name="Alice")
    await channel.mint_access(ROOM, BROWSER, attributes={"app.user": ALICE.id})

    # The credential admits; the arrival is what surfaces what it carried.
    await backend.simulate_participant_joined(ROOM, BROWSER)
    record = await record_of(kit, BROWSER)

    founded = record.identity_id or "nothing — a token vouches for nobody"
    print("\nWhat a mint may send along")
    print(f"  the SFU's clients read: {record.metadata[CONFERENCE_METADATA_KEY]['unasserted']}")
    print(f"  what it founded:        {founded}")

    # And the room refuses to emit what it would refuse to store: a credential
    # carrying what cannot survive the round trip is a promise it cannot keep.
    try:
        await channel.mint_access(ROOM, BROWSER, attributes={"note": "x" * 2_000})
    except ValueError as exc:
        print(f"  over the bound:         {exc}")
    await kit.close_room(ROOM)


async def main() -> None:
    await who_gets_believed()
    await what_the_record_keeps()
    await what_the_mint_may_send_along()


if __name__ == "__main__":
    asyncio.run(main())
