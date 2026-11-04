"""An address is unique within an organization, not globally (RFC §17.2).

`identity_addresses` was keyed on `(channel_type, address)`, so a phone number
registered by one tenant resolved to that tenant's identity for every other
tenant — and a second tenant could not register the same number at all.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest

from roomkit import AddressLinkedError, IdentityNotFoundError
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelType
from roomkit.models.identity import Identity
from roomkit.models.participant import Participant
from roomkit.models.room import Room
from roomkit.store.base import ConversationStore
from roomkit.store.memory import InMemoryStore
from roomkit.store.sqlite import _SCHEMA_VERSION, SQLiteStore


@pytest.fixture(params=["memory", "sqlite", "postgres"])
async def contract_store(request: pytest.FixtureRequest) -> AsyncIterator[ConversationStore]:
    store: ConversationStore
    if request.param == "memory":
        store = InMemoryStore()
    elif request.param == "sqlite":
        store = SQLiteStore(":memory:")
    else:
        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            pytest.skip("POSTGRES_DSN not set")
        from roomkit.store.postgres import PostgresStore

        store = PostgresStore(dsn=dsn)
        await store.init(min_size=1, max_size=2)
    try:
        yield store
    finally:
        await store.close()


async def _two_tenants_one_number(store) -> tuple[Identity, Identity]:  # noqa: ANN001
    alice = Identity(id="acme-alice", organization_id="acme", display_name="Alice")
    bob = Identity(id="globex-bob", organization_id="globex", display_name="Bob")
    await store.create_identity(alice)
    await store.create_identity(bob)
    await store.link_address(alice.id, "sms", "+15551234567", organization_id="acme")
    await store.link_address(bob.id, "sms", "+15551234567", organization_id="globex")
    return alice, bob


class TestInMemory:
    async def test_same_address_resolves_per_organization(self) -> None:
        store = InMemoryStore()
        await _two_tenants_one_number(store)

        acme = await store.resolve_identity("sms", "+15551234567", organization_id="acme")
        globex = await store.resolve_identity("sms", "+15551234567", organization_id="globex")

        assert acme is not None and acme.id == "acme-alice"
        assert globex is not None and globex.id == "globex-bob"

    async def test_unscoped_lookup_does_not_reach_a_tenant(self) -> None:
        """The leak this key exists to prevent: an unscoped caller must not be
        handed some tenant's identity."""
        store = InMemoryStore()
        await _two_tenants_one_number(store)

        assert await store.resolve_identity("sms", "+15551234567") is None

    async def test_unscoped_registrations_still_work(self) -> None:
        store = InMemoryStore()
        solo = Identity(id="solo", display_name="Solo")
        await store.create_identity(solo)
        await store.link_address(solo.id, "sms", "+15559999999")

        assert (await store.resolve_identity("sms", "+15559999999")).id == "solo"
        assert await store.resolve_identity("sms", "+15559999999", organization_id="acme") is None


class TestSQLite:
    async def test_same_address_resolves_per_organization(self, tmp_path) -> None:
        store = SQLiteStore(tmp_path / "scoping.db")
        try:
            await _two_tenants_one_number(store)

            acme = await store.resolve_identity("sms", "+15551234567", organization_id="acme")
            globex = await store.resolve_identity("sms", "+15551234567", organization_id="globex")

            assert acme is not None and acme.id == "acme-alice"
            assert globex is not None and globex.id == "globex-bob"
            assert await store.resolve_identity("sms", "+15551234567") is None
        finally:
            await store.close()


class TestCreatingOverAHeldId:
    """An identity created under an id the store holds replaces the stored one
    within its organization (RFC §5.6, §17.2), and never across one."""

    async def test_an_id_another_organization_holds_is_refused(
        self, contract_store: ConversationStore
    ) -> None:
        address = f"+1555{uuid4().int % 10**7:07d}"
        held = Identity(
            id=f"id-{uuid4().hex}",
            organization_id="acme",
            display_name="Alice",
            channel_addresses={"sms": [address]},
        )
        await contract_store.create_identity(held)

        with pytest.raises(IdentityNotFoundError):
            await contract_store.create_identity(
                Identity(id=held.id, organization_id="globex", display_name="Intruder")
            )

        assert await contract_store.get_identity(held.id) == held
        resolved = await contract_store.resolve_identity("sms", address, "acme")
        assert resolved == held

    async def test_a_replaced_identity_stops_resolving_at_what_it_no_longer_holds(
        self, contract_store: ConversationStore
    ) -> None:
        old, new = (f"+1555{uuid4().int % 10**7:07d}" for _ in range(2))
        first = Identity(
            id=f"id-{uuid4().hex}", organization_id="acme", channel_addresses={"sms": [old]}
        )
        await contract_store.create_identity(first)
        replaced = first.model_copy(update={"channel_addresses": {"sms": [new]}})

        await contract_store.create_identity(replaced)

        assert await contract_store.resolve_identity("sms", old, "acme") is None
        assert await contract_store.resolve_identity("sms", new, "acme") == replaced

    @pytest.mark.parametrize(("stored", "given"), [(None, "acme"), ("acme", None)])
    async def test_no_organization_counts_as_another(
        self, contract_store: ConversationStore, stored: str | None, given: str | None
    ) -> None:
        held = Identity(id=f"id-{uuid4().hex}", organization_id=stored, display_name="Alice")
        await contract_store.create_identity(held)

        with pytest.raises(IdentityNotFoundError):
            await contract_store.create_identity(Identity(id=held.id, organization_id=given))

        assert await contract_store.get_identity(held.id) == held

    async def test_a_replacement_refused_for_an_address_keeps_the_old_ones(
        self, contract_store: ConversationStore
    ) -> None:
        kept, taken = (f"+1555{uuid4().int % 10**7:07d}" for _ in range(2))
        first = Identity(
            id=f"id-{uuid4().hex}", organization_id="acme", channel_addresses={"sms": [kept]}
        )
        await contract_store.create_identity(first)
        await contract_store.create_identity(
            Identity(
                id=f"id-{uuid4().hex}", organization_id="acme", channel_addresses={"sms": [taken]}
            )
        )

        with pytest.raises(AddressLinkedError):
            await contract_store.create_identity(
                first.model_copy(update={"channel_addresses": {"sms": [taken]}})
            )

        assert await contract_store.get_identity(first.id) == first
        assert await contract_store.resolve_identity("sms", kept, "acme") == first


class TestOneIdentityPerAddress:
    """An address belongs to one identity per organization and channel type
    (RFC §5.6, §14.1): a resolution never chooses, and an address moves only
    by an explicit unlink, never by a silent re-home of someone's rooms."""

    @staticmethod
    async def _created(store: ConversationStore, organization_id: str | None) -> Identity:
        identity = Identity(id=f"id-{uuid4().hex}", organization_id=organization_id)
        return await store.create_identity(identity)

    @pytest.mark.parametrize("organization_id", ["acme", None])
    async def test_an_address_another_identity_holds_is_refused(
        self, contract_store: ConversationStore, organization_id: str | None
    ) -> None:
        address = f"+1555{uuid4().int % 10**7:07d}"
        holder = await self._created(contract_store, organization_id)
        other = await self._created(contract_store, organization_id)
        await contract_store.link_address(holder.id, "sms", address, organization_id)

        with pytest.raises(AddressLinkedError):
            await contract_store.link_address(other.id, "sms", address, organization_id)

        resolved = await contract_store.resolve_identity("sms", address, organization_id)
        assert resolved is not None and resolved.id == holder.id
        unchanged = await contract_store.get_identity(other.id)
        assert unchanged is not None and unchanged.channel_addresses == {}

    async def test_an_identity_created_with_an_address_another_holds_is_refused(
        self, contract_store: ConversationStore
    ) -> None:
        address = f"+1555{uuid4().int % 10**7:07d}"
        holder = await self._created(contract_store, "acme")
        await contract_store.link_address(holder.id, "sms", address, "acme")
        newcomer = Identity(
            id=f"id-{uuid4().hex}", organization_id="acme", channel_addresses={"sms": [address]}
        )

        with pytest.raises(AddressLinkedError):
            await contract_store.create_identity(newcomer)

        assert await contract_store.get_identity(newcomer.id) is None
        resolved = await contract_store.resolve_identity("sms", address, "acme")
        assert resolved is not None and resolved.id == holder.id

    async def test_linking_an_address_the_identity_holds_changes_nothing(
        self, contract_store: ConversationStore
    ) -> None:
        address = f"+1555{uuid4().int % 10**7:07d}"
        identity = await self._created(contract_store, "acme")
        await contract_store.link_address(identity.id, "sms", address, "acme")

        await contract_store.link_address(identity.id, "sms", address, "acme")

        stored = await contract_store.get_identity(identity.id)
        assert stored is not None and stored.channel_addresses == {"sms": [address]}

    @pytest.mark.parametrize(
        ("owner", "linked_for"),
        [("acme", "globex"), ("acme", None), (None, "acme")],
        ids=["another-organization", "scoped-identity-unscoped-link", "unscoped-identity"],
    )
    async def test_another_organizations_identity_is_refused(
        self, contract_store: ConversationStore, owner: str | None, linked_for: str | None
    ) -> None:
        address = f"+1555{uuid4().int % 10**7:07d}"
        identity = await self._created(contract_store, owner)

        with pytest.raises(IdentityNotFoundError):
            await contract_store.link_address(identity.id, "sms", address, linked_for)

        assert await contract_store.resolve_identity("sms", address, linked_for) is None
        stored = await contract_store.get_identity(identity.id)
        assert stored is not None and stored.channel_addresses == {}

    async def test_an_unknown_identity_is_refused(self, contract_store: ConversationStore) -> None:
        address = f"+1555{uuid4().int % 10**7:07d}"

        with pytest.raises(IdentityNotFoundError):
            await contract_store.link_address(f"id-{uuid4().hex}", "sms", address, "acme")

        assert await contract_store.resolve_identity("sms", address, "acme") is None


class TestUnlinkingAnAddress:
    """``unlink_address`` frees an address, so moving one is an unlink then a
    link (RFC §5.6, §14.1); a stored conversation is never re-homed."""

    @staticmethod
    async def _created(store: ConversationStore, organization_id: str | None) -> Identity:
        identity = Identity(id=f"id-{uuid4().hex}", organization_id=organization_id)
        return await store.create_identity(identity)

    @pytest.mark.parametrize("organization_id", ["acme", None])
    async def test_an_unlinked_address_moves_to_another_identity(
        self, contract_store: ConversationStore, organization_id: str | None
    ) -> None:
        address = f"+1555{uuid4().int % 10**7:07d}"
        first = await self._created(contract_store, organization_id)
        second = await self._created(contract_store, organization_id)
        await contract_store.link_address(first.id, "sms", address, organization_id)

        await contract_store.unlink_address(first.id, "sms", address, organization_id)

        assert await contract_store.resolve_identity("sms", address, organization_id) is None
        freed = await contract_store.get_identity(first.id)
        assert freed is not None and address not in freed.channel_addresses.get("sms", [])
        await contract_store.link_address(second.id, "sms", address, organization_id)
        resolved = await contract_store.resolve_identity("sms", address, organization_id)
        assert resolved is not None and resolved.id == second.id

    async def test_an_address_the_identity_does_not_hold_changes_nothing(
        self, contract_store: ConversationStore
    ) -> None:
        address = f"+1555{uuid4().int % 10**7:07d}"
        holder = await self._created(contract_store, "acme")
        other = await self._created(contract_store, "acme")
        await contract_store.link_address(holder.id, "sms", address, "acme")

        await contract_store.unlink_address(other.id, "sms", address, "acme")
        await contract_store.unlink_address(holder.id, "email", address, "acme")

        resolved = await contract_store.resolve_identity("sms", address, "acme")
        assert resolved is not None and resolved.id == holder.id
        kept = await contract_store.get_identity(holder.id)
        assert kept is not None and kept.channel_addresses == {"sms": [address]}

    @pytest.mark.parametrize(
        ("owner", "unlinked_for"),
        [("acme", "globex"), ("acme", None), (None, "acme"), ("acme", "unknown")],
        ids=["another-organization", "scoped-identity", "unscoped-identity", "unknown"],
    )
    async def test_an_unknown_or_another_organizations_identity_is_refused(
        self, contract_store: ConversationStore, owner: str | None, unlinked_for: str | None
    ) -> None:
        address = f"+1555{uuid4().int % 10**7:07d}"
        identity = await self._created(contract_store, owner)
        await contract_store.link_address(identity.id, "sms", address, owner)
        target = f"id-{uuid4().hex}" if unlinked_for == "unknown" else identity.id
        scope = owner if unlinked_for == "unknown" else unlinked_for

        with pytest.raises(IdentityNotFoundError):
            await contract_store.unlink_address(target, "sms", address, scope)

        resolved = await contract_store.resolve_identity("sms", address, owner)
        assert resolved is not None and resolved.id == identity.id

    async def test_the_conversations_that_named_the_address_stay_as_they_are(
        self, contract_store: ConversationStore
    ) -> None:
        address = f"+1555{uuid4().int % 10**7:07d}"
        identity = await self._created(contract_store, "acme")
        await contract_store.link_address(identity.id, "sms", address, "acme")
        room = await contract_store.create_room(Room(id=uuid4().hex, organization_id="acme"))
        binding = await contract_store.add_binding(
            ChannelBinding(
                room_id=room.id,
                channel_id="sms",
                channel_type=ChannelType.SMS,
                participant_id=address,
            )
        )
        member = await contract_store.add_participant(
            Participant(id=address, room_id=room.id, channel_id="sms", identity_id=identity.id)
        )
        try:
            await contract_store.unlink_address(identity.id, "sms", address, "acme")

            assert await contract_store.get_binding(room.id, "sms") == binding
            assert await contract_store.get_participant(room.id, address) == member
        finally:
            await contract_store.delete_room(room.id)


async def test_two_concurrent_links_of_one_address_leave_one_winner_on_postgres() -> None:
    """Two links racing on separate connections: the address table's key and
    the link's single statement leave one identity holding the address."""
    dsn = os.environ.get("POSTGRES_DSN")
    if not dsn:
        pytest.skip("POSTGRES_DSN not set")
    from roomkit.store.postgres import PostgresStore

    store = PostgresStore(dsn=dsn)
    await store.init(min_size=2, max_size=4)
    try:
        address = f"+1555{uuid4().int % 10**7:07d}"
        rivals = [
            await store.create_identity(Identity(id=f"id-{uuid4().hex}", organization_id="acme"))
            for _ in range(2)
        ]

        results = await asyncio.gather(
            *(store.link_address(r.id, "sms", address, "acme") for r in rivals),
            return_exceptions=True,
        )

        refused = [r for r in results if isinstance(r, AddressLinkedError)]
        assert len(refused) == 1 and results.count(None) == 1, results
        winner = rivals[results.index(None)]
        resolved = await store.resolve_identity("sms", address, "acme")
        assert resolved is not None and resolved.id == winner.id
        loser = await store.get_identity(rivals[results.index(refused[0])].id)
        assert loser is not None and loser.channel_addresses == {}
    finally:
        await store.close()


class TestAddressesDeclaredOnTheIdentity:
    """``create_identity`` carries its own addresses, and they are the tenant's.

    Every other test here reaches the address table through ``link_address``,
    which takes an ``organization_id`` and cannot forget it. An identity
    created with ``channel_addresses`` already populated writes that table too,
    from a field rather than an argument — the one path where the organization
    can be dropped silently.
    """

    @staticmethod
    def _acme() -> Identity:
        return Identity(
            id="acme-carol",
            organization_id="acme",
            display_name="Carol",
            channel_addresses={"sms": ["+15550001111"]},
        )

    async def test_memory_files_them_under_the_owning_tenant(self) -> None:
        store = InMemoryStore()
        await store.create_identity(self._acme())

        own = await store.resolve_identity("sms", "+15550001111", organization_id="acme")
        unscoped = await store.resolve_identity("sms", "+15550001111", organization_id="")

        assert own is not None and own.id == "acme-carol"
        assert unscoped is None

    async def test_sqlite_files_them_under_the_owning_tenant(self, tmp_path) -> None:  # noqa: ANN001
        store = SQLiteStore(tmp_path / "declared.db")
        try:
            await store.create_identity(self._acme())

            own = await store.resolve_identity("sms", "+15550001111", organization_id="acme")
            unscoped = await store.resolve_identity("sms", "+15550001111", organization_id="")

            # Dropping the organization here filed the address under the empty
            # one: the owning tenant missed its own identity, and an unscoped
            # caller reached it.
            assert own is not None and own.id == "acme-carol"
            assert unscoped is None
        finally:
            await store.close()

    async def test_sqlite_keeps_an_unscoped_identity_unscoped(self, tmp_path) -> None:  # noqa: ANN001
        """No organization on the identity is a value, not a missing one."""
        store = SQLiteStore(tmp_path / "unscoped.db")
        try:
            await store.create_identity(
                Identity(id="nobody", channel_addresses={"sms": ["+15559998888"]})
            )

            assert await store.resolve_identity("sms", "+15559998888") is not None
            assert await store.resolve_identity("sms", "+15559998888", "acme") is None
        finally:
            await store.close()


class TestMigration:
    async def test_v2_file_migrates_and_keeps_its_addresses(self, tmp_path) -> None:
        """A v2 file's rows carry the unscoped tenant, which preserves exactly
        what they resolved to before."""
        path = str(tmp_path / "v2.db")
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            CREATE TABLE identities(id TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE identity_addresses(
                channel_type TEXT NOT NULL,
                address TEXT NOT NULL,
                identity_id TEXT NOT NULL,
                PRIMARY KEY(channel_type, address)
            );
            PRAGMA user_version=2;
            """
        )
        v2_identity = Identity(id="pre-scoping", display_name="Registered before scoping")
        conn.execute(
            "INSERT INTO identities(id, data) VALUES(?, ?)",
            (v2_identity.id, v2_identity.model_dump_json()),
        )
        conn.execute(
            "INSERT INTO identity_addresses(channel_type, address, identity_id) VALUES(?, ?, ?)",
            ("sms", "+15550000000", v2_identity.id),
        )
        conn.commit()
        conn.close()

        store = SQLiteStore(path)
        try:
            resolved = await store.resolve_identity("sms", "+15550000000")
            assert resolved is not None and resolved.id == "pre-scoping"
        finally:
            await store.close()

        conn = sqlite3.connect(path)
        try:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == _SCHEMA_VERSION
            columns = [r[1] for r in conn.execute("PRAGMA table_info(identity_addresses)")]
            assert "organization_id" in columns
            # The rebuilt key admits the same address for a second tenant.
            conn.execute(
                "INSERT INTO identity_addresses(channel_type, address, identity_id,"
                " organization_id) VALUES(?, ?, ?, ?)",
                ("sms", "+15550000000", "pre-scoping", "acme"),
            )
            conn.commit()
        finally:
            conn.close()

    async def test_migration_is_not_rerun_on_reopen(self, tmp_path) -> None:
        path = tmp_path / "stable.db"
        store = SQLiteStore(path)
        try:
            identity = Identity(id="i1", organization_id="acme")
            await store.create_identity(identity)
            await store.link_address(identity.id, "sms", "+1555", organization_id="acme")
        finally:
            await store.close()

        reopened = SQLiteStore(path)
        try:
            resolved = await reopened.resolve_identity("sms", "+1555", organization_id="acme")
            assert resolved is not None and resolved.id == "i1"
        finally:
            await reopened.close()


class TestABCContract:
    def test_signatures_carry_the_optional_scope(self) -> None:
        """A custom store implementing the pre-scoping signature keeps working
        for its own callers, but the ABC now offers the parameter."""
        import inspect

        from roomkit.store.base import ConversationStore

        for name in ("resolve_identity", "link_address"):
            params = inspect.signature(getattr(ConversationStore, name)).parameters
            assert "organization_id" in params
            assert params["organization_id"].default is None

        with pytest.raises(TypeError):
            ConversationStore()  # still abstract
