"""An address is unique within an organization, not globally (RFC §17.2).

`identity_addresses` was keyed on `(channel_type, address)`, so a phone number
registered by one tenant resolved to that tenant's identity for every other
tenant — and a second tenant could not register the same number at all.
"""

from __future__ import annotations

import sqlite3

import pytest

from roomkit.models.identity import Identity
from roomkit.store.memory import InMemoryStore
from roomkit.store.sqlite import _SCHEMA_VERSION, SQLiteStore


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

    async def test_relinking_within_one_organization_replaces(self, tmp_path) -> None:
        """Scoping widens the key; it does not turn it into a multimap."""
        store = SQLiteStore(tmp_path / "relink.db")
        try:
            first = Identity(id="first", organization_id="acme")
            second = Identity(id="second", organization_id="acme")
            await store.create_identity(first)
            await store.create_identity(second)
            await store.link_address(first.id, "sms", "+1555", organization_id="acme")
            await store.link_address(second.id, "sms", "+1555", organization_id="acme")

            resolved = await store.resolve_identity("sms", "+1555", organization_id="acme")
            assert resolved is not None and resolved.id == "second"
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
        legacy = Identity(id="legacy", display_name="Legacy")
        conn.execute(
            "INSERT INTO identities(id, data) VALUES(?, ?)", (legacy.id, legacy.model_dump_json())
        )
        conn.execute(
            "INSERT INTO identity_addresses(channel_type, address, identity_id) VALUES(?, ?, ?)",
            ("sms", "+15550000000", legacy.id),
        )
        conn.commit()
        conn.close()

        store = SQLiteStore(path)
        try:
            resolved = await store.resolve_identity("sms", "+15550000000")
            assert resolved is not None and resolved.id == "legacy"
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
                ("sms", "+15550000000", "legacy", "acme"),
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
