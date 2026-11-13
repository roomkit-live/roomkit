"""Identity verification: requests, completion, revocation and state (RFC §11.7).

Run on the in-memory, SQLite and (with POSTGRES_DSN) Postgres stores: the
framework keeps the state and the lifecycle, the method stays the caller's.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from roomkit import (
    FrameworkEvent,
    KitClosedError,
    RoomKit,
    RoomNotFoundError,
    VerificationPolicy,
    VerificationRefusedError,
    VerificationRequestStatus,
    VerificationStatus,
)
from roomkit.core.locks import InMemoryLockManager, RoomLockManager
from roomkit.core.mixins import verification as verification_mixin
from roomkit.models.verification import Verification
from roomkit.store.base import ConversationStore
from roomkit.store.memory import InMemoryStore
from roomkit.store.postgres_lock import PostgresAdvisoryLockManager
from roomkit.store.sqlite import SQLiteStore
from tests.test_locks import SharedLockManager, a_lock_manager

# The production work factor, ~55 ms a hash: what a leaked store's codes cost.
_PRODUCTION_SCRYPT_N = verification_mixin._SCRYPT_N


@pytest.fixture(autouse=True)
def _cheap_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    """scrypt at a test's cost; the slow-hash test puts the production factor back."""
    monkeypatch.setattr(verification_mixin, "_SCRYPT_N", 2**4)


@pytest.fixture(params=["memory", "sqlite", "postgres"])
async def store(request: pytest.FixtureRequest) -> AsyncIterator[ConversationStore]:
    backend: ConversationStore
    if request.param == "memory":
        backend = InMemoryStore()
    elif request.param == "sqlite":
        backend = SQLiteStore(":memory:")
    else:
        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            pytest.skip("POSTGRES_DSN not set")
        from roomkit.store.postgres import PostgresStore

        backend = PostgresStore(dsn=dsn)
        await backend.init(min_size=1, max_size=2)
    try:
        yield backend
    finally:
        await backend.close()


def _kit(store: ConversationStore, **policy: float) -> tuple[RoomKit, list[FrameworkEvent]]:
    kit = RoomKit(store=store, verification_policy=VerificationPolicy(**policy))
    events: list[FrameworkEvent] = []
    for event_type in ("verification_requested", "verification_completed", "verification_ended"):

        @kit.on(event_type)
        async def record(event: FrameworkEvent) -> None:
            events.append(event)

    return kit, events


async def _ended(events: list[FrameworkEvent], reason: str) -> None:
    """Until a verification_ended of *reason* was announced: the expiry timer ran."""
    async with asyncio.timeout(5):
        while not any(
            e.type == "verification_ended" and e.data.get("reason") == reason for e in events
        ):
            await asyncio.sleep(0.005)


def _identity() -> str:
    return f"id-{uuid4().hex[:12]}"


async def _room(kit: RoomKit, organization_id: str | None = None) -> str:
    """A fresh room: a request is opened in one, scoped to its organization."""
    room = await kit.create_room(room_id=f"r-{uuid4().hex[:12]}", organization_id=organization_id)
    return room.id


async def test_a_completed_request_verifies_the_identity(store: ConversationStore) -> None:
    kit, events = _kit(store)
    room = await _room(kit)
    alice = _identity()

    request, secret = await kit.request_verification(alice, room_id=room, channel_id="sms")
    pending = await kit.verification_of(alice)
    verification = await kit.complete_verification(request.id, secret)

    assert pending.status == VerificationStatus.PENDING
    assert verification.status == VerificationStatus.VERIFIED
    assert verification.verified_via == "sms" and verification.method == "link"
    assert verification.expires_at is not None
    assert 899 < (verification.expires_at - datetime.now(UTC)).total_seconds() <= 900
    assert (await kit.verification_of(alice)).status == VerificationStatus.VERIFIED
    assert [e.type for e in events] == ["verification_requested", "verification_completed"]
    await kit.close()


@pytest.mark.parametrize(
    "locks", [InMemoryLockManager, SharedLockManager, PostgresAdvisoryLockManager]
)
async def test_a_room_named_like_a_verification_key_holds_no_verification(
    locks: type[RoomLockManager],
) -> None:
    """An identity's verification lock is in a key space of its own (RFC
    §13.5): a section of the room ``verification::alice`` leaves alice's
    transitions free, with the in-memory manager, one shared across
    processes, or PostgreSQL's advisory locks."""
    kit = RoomKit(lock_manager=await a_lock_manager(locks))
    room = await _room(kit)
    alice = _identity()
    held, done = asyncio.Event(), asyncio.Event()

    async def section() -> None:
        async with kit.lock_manager.locked(f"verification::{alice}"):
            held.set()
            await done.wait()

    holder = asyncio.create_task(section())
    await held.wait()
    try:
        request, _ = await asyncio.wait_for(
            kit.request_verification(alice, room_id=room, channel_id="sms"), 1.0
        )
    finally:
        done.set()
        await holder
    assert request.identity_id == alice
    await kit.close()


async def test_a_manager_built_on_the_in_memory_one_takes_every_kind_of_lock() -> None:
    """A subclass that takes its locks its own way (a distributed lock over
    the in-memory reentrance) is asked for an identity's lock too."""

    class Recording(InMemoryLockManager):
        def __init__(self) -> None:
            super().__init__()
            self.keys: list[str] = []

        @asynccontextmanager
        async def locked(self, room_id: str) -> AsyncGenerator[None, None]:
            self.keys.append(room_id)
            async with super().locked(room_id):
                yield

    locks = Recording()
    kit = RoomKit(lock_manager=locks)
    await kit.request_verification("alice", room_id=await _room(kit), channel_id="sms")
    assert "\x00verification::alice" in locks.keys
    await kit.close()


async def test_two_kits_sharing_a_lock_manager_share_an_identity_s_lock() -> None:
    """Kits sharing a store and an in-memory lock manager, as processes
    sharing a distributed one, serialise an identity's transitions."""
    store, locks = InMemoryStore(), InMemoryLockManager()
    first = RoomKit(store=store, lock_manager=locks)
    second = RoomKit(store=store, lock_manager=locks)
    room = await _room(first)
    held, done = asyncio.Event(), asyncio.Event()

    async def holder() -> None:
        async with first._verification_locks.locked("verification::alice"):
            held.set()
            await done.wait()

    holding = asyncio.create_task(holder())
    await held.wait()
    asking = asyncio.create_task(
        second.request_verification("alice", room_id=room, channel_id="sms")
    )
    await asyncio.sleep(0.02)
    assert not asking.done()
    done.set()
    await holding
    await asyncio.wait_for(asking, 1.0)
    await first.close()
    await second.close()


async def test_the_close_cuts_a_wait_for_an_identity_s_lock_under_its_namesake_room() -> None:
    """A call waiting for alice's lock under the lock of the room
    ``verification::alice`` waits for alice's lock all the same: the
    kit's close cuts it at once (RFC §12.10.4)."""
    kit = RoomKit()
    held, done = asyncio.Event(), asyncio.Event()

    async def holder() -> None:
        async with kit._verification_locks.locked("verification::alice"):
            held.set()
            await done.wait()

    async def revoke() -> None:
        async with kit.lock_manager.locked("verification::alice"):
            await kit.revoke_verification("alice")

    holding = asyncio.create_task(holder())
    await held.wait()
    call = asyncio.create_task(revoke())
    async with asyncio.timeout(1):
        while not kit._door_runs._waiting:
            await asyncio.sleep(0)
    closing = asyncio.create_task(kit.close())
    with pytest.raises(KitClosedError):
        await asyncio.wait_for(call, 0.5)
    done.set()
    await holding
    await closing


async def test_the_secret_is_returned_once_and_never_stored(store: ConversationStore) -> None:
    kit, _ = _kit(store)
    room = await _room(kit)

    request, secret = await kit.request_verification(_identity(), room_id=room, channel_id="sms")
    stored = await store.get_verification_request(request.id)

    assert len(secret) >= 32  # a link secret carries at least 128 bits
    assert stored is not None
    assert secret not in stored.model_dump_json()
    assert stored.secret_hash != secret
    await kit.close()


async def test_a_wrong_secret_counts_and_the_last_attempt_fails_the_request(
    store: ConversationStore,
) -> None:
    kit, events = _kit(store, max_attempts=2)
    room = await _room(kit)
    alice = _identity()
    request, secret = await kit.request_verification(alice, room_id=room, channel_id="sms")

    with pytest.raises(VerificationRefusedError) as first:
        await kit.complete_verification(request.id, "wrong")
    with pytest.raises(VerificationRefusedError) as second:
        await kit.complete_verification(request.id, "wrong again")
    with pytest.raises(VerificationRefusedError) as after:
        await kit.complete_verification(request.id, secret)

    assert (first.value.reason, second.value.reason, after.value.reason) == (
        "mismatch",
        "failed",
        "not_open",
    )
    stored = await store.get_verification_request(request.id)
    assert stored is not None and stored.status == VerificationRequestStatus.FAILED
    assert (await kit.verification_of(alice)).status == VerificationStatus.UNVERIFIED
    assert events[-1].type == "verification_ended" and events[-1].data["reason"] == "failed"
    await kit.close()


async def test_an_expired_request_cannot_be_completed(store: ConversationStore) -> None:
    kit, events = _kit(store, request_ttl=0.05)
    room = await _room(kit)
    alice = _identity()
    request, secret = await kit.request_verification(alice, room_id=room, channel_id="sms")
    await _ended(events, "expired")

    with pytest.raises(VerificationRefusedError) as refused:
        await kit.complete_verification(request.id, secret)

    assert refused.value.reason == "expired"
    assert (await kit.verification_of(alice)).status == VerificationStatus.UNVERIFIED
    assert events[-1].data["reason"] == "expired"
    await kit.close()


async def test_a_new_request_cancels_the_open_one(store: ConversationStore) -> None:
    kit, events = _kit(store)
    room = await _room(kit)
    alice = _identity()
    first, first_secret = await kit.request_verification(alice, room_id=room, channel_id="sms")
    second, second_secret = await kit.request_verification(alice, room_id=room, channel_id="sms")

    with pytest.raises(VerificationRefusedError) as refused:
        await kit.complete_verification(first.id, first_secret)
    verification = await kit.complete_verification(second.id, second_secret)

    assert refused.value.reason == "not_open"
    assert verification.status == VerificationStatus.VERIFIED
    cancelled = [e for e in events if e.data.get("reason") == "cancelled"]
    assert [e.data["request_id"] for e in cancelled] == [first.id]
    await kit.close()


async def test_revoking_ends_the_verification_and_its_open_request(
    store: ConversationStore,
) -> None:
    kit, events = _kit(store)
    room = await _room(kit)
    alice = _identity()
    request, secret = await kit.request_verification(alice, room_id=room, channel_id="sms")
    await kit.complete_verification(request.id, secret)
    later, later_secret = await kit.request_verification(alice, room_id=room, channel_id="sms")

    revoked = await kit.revoke_verification(alice, reason="advisor")
    with pytest.raises(VerificationRefusedError) as refused:
        await kit.complete_verification(later.id, later_secret)

    assert revoked is not None and revoked.status == VerificationStatus.REVOKED
    assert revoked.ended_reason == "advisor"
    assert refused.value.reason == "not_open"
    assert (await kit.verification_of(alice)).status == VerificationStatus.REVOKED
    assert await kit.revoke_verification(alice) is None  # nothing left in force
    assert any(e.data.get("reason") == "advisor" for e in events)
    await kit.close()


async def test_revoking_a_pending_identity_kills_its_link(store: ConversationStore) -> None:
    """Revoked before verifying (a suspected fraud): the link already sent dies."""
    kit, _ = _kit(store)
    room = await _room(kit)
    bob = _identity()
    request, secret = await kit.request_verification(bob, room_id=room, channel_id="sms")

    assert await kit.revoke_verification(bob, reason="fraud suspected") is None
    with pytest.raises(VerificationRefusedError) as refused:
        await kit.complete_verification(request.id, secret)

    assert refused.value.reason == "not_open"
    assert (await kit.verification_of(bob)).status == VerificationStatus.UNVERIFIED
    await kit.close()


async def test_a_failed_request_ends_the_verification_in_force(
    store: ConversationStore,
) -> None:
    """A verified person asked again (a suspicion) and wrong secrets: fail closed."""
    kit, _ = _kit(store, max_attempts=2)
    room = await _room(kit)
    alice = _identity()
    first, secret = await kit.request_verification(alice, room_id=room, channel_id="sms")
    await kit.complete_verification(first.id, secret)
    again, _ = await kit.request_verification(alice, room_id=room, channel_id="sms", method="code")

    for _ in range(2):
        with pytest.raises(VerificationRefusedError):
            await kit.complete_verification(again.id, "000000")

    state = await kit.verification_of(alice)
    assert state.status == VerificationStatus.UNVERIFIED and state.ended_reason == "failed"
    await kit.close()


async def test_an_unanswered_request_leaves_the_verification_in_force(
    store: ConversationStore,
) -> None:
    kit, events = _kit(store, request_ttl=0.05)
    room = await _room(kit)
    alice = _identity()
    first, secret = await kit.request_verification(alice, room_id=room, channel_id="sms")
    await kit.complete_verification(first.id, secret)
    again, again_secret = await kit.request_verification(alice, room_id=room, channel_id="sms")
    await _ended(events, "expired")

    with pytest.raises(VerificationRefusedError):
        await kit.complete_verification(again.id, again_secret)

    assert (await kit.verification_of(alice)).status == VerificationStatus.VERIFIED
    await kit.close()


def test_the_secret_hash_keeps_its_production_work_factor() -> None:
    """The factor a leaked store's codes are hashed with: lowering it makes
    each one cheaper to break."""
    assert _PRODUCTION_SCRYPT_N == 2**15


async def test_a_secret_is_stored_through_a_slow_hash(
    store: ConversationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 6-digit code read from a leaked store must not fall to a fast hash."""
    import hashlib

    monkeypatch.setattr(verification_mixin, "_SCRYPT_N", _PRODUCTION_SCRYPT_N)
    kit, _ = _kit(store)
    room = await _room(kit)
    request, code = await kit.request_verification(
        _identity(), room_id=room, channel_id="sms", method="code"
    )

    assert request.secret_hash.startswith("scrypt$")
    assert hashlib.sha256(f"{request.id}:{code}".encode()).hexdigest() not in request.secret_hash
    await kit.close()


async def test_a_verification_reads_expired_once_its_time_passed(
    store: ConversationStore,
) -> None:
    kit, events = _kit(store, ttl=0.05)
    room = await _room(kit)
    alice = _identity()
    request, secret = await kit.request_verification(alice, room_id=room, channel_id="sms")
    await kit.complete_verification(request.id, secret)
    await _ended(events, "expired")

    state = await kit.verification_of(alice)

    assert state.status == VerificationStatus.EXPIRED and state.ended_reason == "expired"
    await kit.close()


async def test_asking_a_verified_identity_again_keeps_it_verified_meanwhile(
    store: ConversationStore,
) -> None:
    kit, _ = _kit(store)
    room = await _room(kit)
    alice = _identity()
    request, secret = await kit.request_verification(alice, room_id=room, channel_id="sms")
    await kit.complete_verification(request.id, secret)

    await kit.request_verification(alice, room_id=room, channel_id="whatsapp")

    assert (await kit.verification_of(alice)).status == VerificationStatus.VERIFIED
    await kit.close()


async def test_a_code_request_sends_digits_and_lives_shorter(store: ConversationStore) -> None:
    kit, _ = _kit(store)
    room = await _room(kit)

    request, code = await kit.request_verification(
        _identity(), room_id=room, channel_id="sms", method="code"
    )

    assert code.isdigit() and len(code) == 6
    assert (request.expires_at - request.created_at).total_seconds() == 300
    assert (await kit.complete_verification(request.id, code)).method == "code"
    await kit.close()


async def test_organizations_are_kept_apart(store: ConversationStore) -> None:
    kit, _ = _kit(store)
    room = await _room(kit, "org-a")
    alice = _identity()
    request, secret = await kit.request_verification(
        alice, room_id=room, channel_id="sms", organization_id="org-a"
    )

    with pytest.raises(VerificationRefusedError) as refused:
        await kit.complete_verification(request.id, secret, organization_id="org-b")
    await kit.complete_verification(request.id, secret, organization_id="org-a")

    assert refused.value.reason == "not_found"
    assert (await kit.verification_of(alice, organization_id="org-a")).status == (
        VerificationStatus.VERIFIED
    )
    assert (await kit.verification_of(alice)).status == VerificationStatus.UNVERIFIED
    await kit.close()


async def test_a_request_is_scoped_to_its_rooms_organization(
    store: ConversationStore,
) -> None:
    """The scope that labels a room's messages is the room's organization: a
    request opened there takes it, and a caller of another is refused."""
    kit, _ = _kit(store)
    alice = _identity()
    room = await _room(kit, "org-a")

    with pytest.raises(RoomNotFoundError):
        await kit.request_verification(
            alice, room_id=room, channel_id="sms", organization_id="org-b"
        )
    request, secret = await kit.request_verification(alice, room_id=room, channel_id="sms")
    await kit.complete_verification(request.id, secret)

    assert request.organization_id == "org-a"
    assert (await kit.verification_of(alice, organization_id="org-a")).status == (
        VerificationStatus.VERIFIED
    )
    assert (await kit.verification_of(alice)).status == VerificationStatus.UNVERIFIED
    await kit.close()


async def test_the_store_lists_the_verifications_in_force(store: ConversationStore) -> None:
    kit, _ = _kit(store)
    room = await _room(kit)
    alice, bob = _identity(), _identity()
    for identity in (alice, bob):
        request, secret = await kit.request_verification(identity, room_id=room, channel_id="sms")
        await kit.complete_verification(request.id, secret)
    await kit.revoke_verification(bob)

    in_force = {v.identity_id for v in await store.list_verifications_in_force()}

    assert alice in in_force and bob not in in_force
    await kit.close()


async def test_a_store_without_verification_refuses_by_name() -> None:
    store = InMemoryStore()

    with pytest.raises(NotImplementedError, match="does not store identity verifications"):
        await ConversationStore.put_verification(store, Verification(identity_id="x"))


async def test_a_v4_sqlite_file_gains_the_verification_tables(tmp_path) -> None:
    import sqlite3

    from roomkit.models.identity import Identity
    from roomkit.store.sqlite import _SCHEMA_VERSION

    path = tmp_path / "v4.db"
    store = SQLiteStore(path)
    await store.create_identity(Identity(id="id-alice"))
    await store.close()
    conn = sqlite3.connect(path)
    conn.executescript(
        "DROP TABLE verifications; DROP TABLE verification_requests; PRAGMA user_version=4;"
    )
    conn.close()

    store = SQLiteStore(path)
    kit, _ = _kit(store)
    room = await _room(kit)
    request, secret = await kit.request_verification("id-alice", room_id=room, channel_id="sms")
    verified = await kit.complete_verification(request.id, secret)
    kept = await store.get_identity("id-alice")
    await kit.close()

    conn = sqlite3.connect(path)
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()
    assert version == _SCHEMA_VERSION
    assert verified.status == VerificationStatus.VERIFIED
    assert kept is not None


async def test_a_failed_check_of_your_own_counts_against_the_request(
    store: ConversationStore,
) -> None:
    """A wrong PIN on the integrator's page: bounded like a wrong secret."""
    kit, events = _kit(store, max_attempts=2)
    room = await _room(kit)
    alice = _identity()
    request, secret = await kit.request_verification(alice, room_id=room, channel_id="sms")

    once = await kit.record_failed_attempt(request.id)
    twice = await kit.record_failed_attempt(request.id)
    with pytest.raises(VerificationRefusedError) as after:
        await kit.complete_verification(request.id, secret)

    assert (once.status, once.attempts) == (VerificationRequestStatus.OPEN, 1)
    assert twice.status == VerificationRequestStatus.FAILED
    assert after.value.reason == "not_open"
    assert (await kit.verification_of(alice)).status == VerificationStatus.UNVERIFIED
    assert events[-1].data.get("reason") == "failed"
    await kit.close()
