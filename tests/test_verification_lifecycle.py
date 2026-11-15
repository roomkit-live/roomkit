"""Identity verification hooks and the end of a verification when it happens (RFC §11.7.5).

A verification ends at its expiry by a timer, not on the sender's next
message; the timers are re-armed when a kit starts, and what expired while
nothing ran is ended then.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, tzinfo
from typing import Any

import pytest

from roomkit import (
    HookExecution,
    HookTrigger,
    RoomKit,
    VerificationPolicy,
    VerificationRefusedError,
    VerificationStatus,
)
from roomkit.core.mixins import verification as verification_mixin
from roomkit.models.verification import Verification
from roomkit.store.base import ConversationStore
from roomkit.store.memory import InMemoryStore


@pytest.fixture(autouse=True)
def _cheap_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """A secret's scrypt at a test's cost (55 ms a hash otherwise), stored and
    compared the same way."""
    monkeypatch.setattr(verification_mixin, "_SCRYPT_N", 2**4)


async def _past(when: datetime | None) -> None:
    """Until *when* passed and a timer due then had time to end what it ends."""
    assert when is not None
    await asyncio.sleep(max(0.0, (when - datetime.now(UTC)).total_seconds()) + 0.05)


async def _eventually(predicate: Callable[[], Any], timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


async def _kit(
    store: ConversationStore | None = None, **policy: float
) -> tuple[RoomKit, list[tuple[str, dict[str, Any]]]]:
    kit = RoomKit(store=store or InMemoryStore(), verification_policy=VerificationPolicy(**policy))
    if await kit.store.get_room("r1") is None:
        await kit.create_room(room_id="r1")
    fired: list[tuple[str, dict[str, Any]]] = []
    for trigger in (HookTrigger.ON_VERIFICATION_COMPLETED, HookTrigger.ON_VERIFICATION_ENDED):

        @kit.hook(trigger, execution=HookExecution.ASYNC)
        async def record(event: Any, ctx: Any, trigger: HookTrigger = trigger) -> None:
            fired.append((str(trigger), dict(event.content.data)))

    return kit, fired


async def _verified(kit: RoomKit, identity_id: str = "id-alice") -> Verification:
    request, secret = await kit.request_verification(identity_id, room_id="r1", channel_id="sms")
    return await kit.complete_verification(request.id, secret)


class _VerificationClock:
    """Only verification's clock and timer waits move when the test advances them."""

    def __init__(self) -> None:
        self._now = datetime.now(UTC)
        self._changed = asyncio.Event()

    def __getattr__(self, name: str) -> object:
        return getattr(asyncio, name)

    def now(self, tz: tzinfo) -> datetime:
        return self._now.astimezone(tz)

    async def sleep(self, delay: float) -> None:
        deadline = self._now + timedelta(seconds=delay)
        while self._now < deadline:
            await self._changed.wait()

    def past(self, deadline: datetime | None) -> None:
        assert deadline is not None and deadline >= self._now
        self._now = deadline + timedelta(microseconds=1)
        self._changed.set()
        self._changed = asyncio.Event()


async def test_a_completion_fires_its_hook_in_the_request_room() -> None:
    kit, fired = await _kit()

    await _verified(kit)

    await _eventually(lambda: fired)
    trigger, data = fired[0]
    assert trigger == HookTrigger.ON_VERIFICATION_COMPLETED
    assert data["identity_id"] == "id-alice" and data["channel_id"] == "sms"
    assert data["method"] == "link"
    await kit.close()


async def test_a_verification_ends_when_it_expires_not_on_the_next_message() -> None:
    kit, fired = await _kit(ttl=0.02)

    await _verified(kit)
    await _eventually(lambda: any(t == HookTrigger.ON_VERIFICATION_ENDED for t, _ in fired))

    ended = [d for t, d in fired if t == HookTrigger.ON_VERIFICATION_ENDED]
    assert ended[0]["reason"] == "expired" and ended[0]["channel_id"] == "sms"
    assert ended[0]["ended"] == "verification"
    stored = await kit.store.get_verification("id-alice")
    assert stored is not None and stored.status == VerificationStatus.EXPIRED
    await kit.close()


async def test_a_revocation_fires_the_end_once_and_stops_the_timer() -> None:
    kit, fired = await _kit(ttl=0.1)  # revoked well before it expires
    verification = await _verified(kit)

    await kit.revoke_verification("id-alice", reason="advisor")
    await _past(verification.expires_at)  # a timer left armed would end it again by now

    ended = [d["reason"] for t, d in fired if t == HookTrigger.ON_VERIFICATION_ENDED]
    assert ended == ["advisor"]
    await kit.close()


async def test_a_request_that_fails_fires_the_end() -> None:
    kit, fired = await _kit(max_attempts=1)
    request, _ = await kit.request_verification("id-alice", room_id="r1", channel_id="sms")

    with pytest.raises(VerificationRefusedError):
        await kit.complete_verification(request.id, "wrong")

    await _eventually(lambda: fired)
    assert fired[0][0] == HookTrigger.ON_VERIFICATION_ENDED and fired[0][1]["reason"] == "failed"
    assert fired[0][1]["ended"] == "request"  # nothing was verified to end
    await kit.close()


async def test_a_request_that_fails_while_verified_ends_the_verification() -> None:
    kit, fired = await _kit(max_attempts=1)
    await _verified(kit)
    again, _ = await kit.request_verification("id-alice", room_id="r1", channel_id="sms")

    with pytest.raises(VerificationRefusedError):
        await kit.complete_verification(again.id, "wrong")

    await _eventually(lambda: any(t == HookTrigger.ON_VERIFICATION_ENDED for t, _ in fired))
    ended = [d for t, d in fired if t == HookTrigger.ON_VERIFICATION_ENDED]
    assert ended[0]["reason"] == "failed" and ended[0]["ended"] == "verification"
    await kit.close()


async def test_a_verification_that_outlived_an_unanswered_request_still_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asked again, the person did not answer: the request's end is not the
    verification's, which is still announced when its own time runs out."""
    clock = _VerificationClock()
    monkeypatch.setattr(verification_mixin, "datetime", clock)
    monkeypatch.setattr(verification_mixin, "asyncio", clock)
    kit, fired = await _kit(ttl=0.2, request_ttl=0.05)
    try:
        verified = await _verified(kit)
        again, secret = await kit.request_verification("id-alice", room_id="r1", channel_id="sms")

        clock.past(again.expires_at)
        await _eventually(lambda: _ends(fired))  # the timer ends it without a late reply
        assert _ends(fired) == [("expired", "request")]
        state = await kit.verification_of("id-alice")
        assert state.status == VerificationStatus.VERIFIED and state.request_id is None
        assert state.expires_at == verified.expires_at
        with pytest.raises(VerificationRefusedError, match="expired"):
            await kit.complete_verification(again.id, secret)

        clock.past(verified.expires_at)
        await _eventually(lambda: len(_ends(fired)) == 2, timeout=3.0)
        assert _ends(fired) == [("expired", "request"), ("expired", "verification")]
        stored = await kit.store.get_verification("id-alice")
        assert stored is not None and stored.status == VerificationStatus.EXPIRED
    finally:
        await kit.close()


async def test_a_starting_kit_rearms_the_timers_of_verifications_in_force() -> None:
    store = InMemoryStore()
    first, _ = await _kit(store, ttl=0.1)
    await _verified(first)
    await first.close()  # its timer dies with it

    second, fired = await _kit(store, ttl=0.1)
    async with second:
        await _eventually(lambda: fired, timeout=2.0)

    assert fired[0][0] == HookTrigger.ON_VERIFICATION_ENDED
    assert fired[0][1]["reason"] == "expired"


async def test_what_expired_while_nothing_ran_is_ended_at_start() -> None:
    store = InMemoryStore()
    first, _ = await _kit(store, ttl=0.1)  # closed well before it expires
    verification = await _verified(first)
    await first.close()
    await _past(verification.expires_at)

    second, fired = await _kit(store)
    ended = await second.check_verifications()

    assert [v.identity_id for v in ended] == ["id-alice"]
    assert ended[0].status == VerificationStatus.EXPIRED
    await _eventually(lambda: fired)
    await second.close()


async def test_a_closed_kit_ends_nothing_afterwards() -> None:
    kit, fired = await _kit(ttl=0.1)  # closed well before it expires
    verification = await _verified(kit)
    await _eventually(lambda: fired)  # the completion

    await kit.close()
    await _past(verification.expires_at)  # its timer, had it lived on, would have fired

    assert [t for t, _ in fired] == [HookTrigger.ON_VERIFICATION_COMPLETED]


class _StoreWithoutVerification(InMemoryStore):
    """A store written before verification existed."""

    async def list_verifications_in_force(self) -> list[Verification]:
        return await ConversationStore.list_verifications_in_force(self)


async def test_a_kit_whose_store_keeps_no_verification_still_starts() -> None:
    kit = RoomKit(store=_StoreWithoutVerification())

    async with kit:
        assert await kit.store.get_room("nope") is None


class _StoreNotReady(InMemoryStore):
    """A store that cannot answer yet (a PostgresStore before ``init()``)."""

    async def list_verifications_in_force(self) -> list[Verification]:
        raise RuntimeError("PostgresStore.init() must be called before use")


async def test_a_kit_whose_store_cannot_answer_still_starts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Resuming verifications never stops a kit from starting: a verification
    whose time passed already reads EXPIRED, only its end is announced late."""
    kit = RoomKit(store=_StoreNotReady())

    with caplog.at_level("WARNING", logger="roomkit.framework"):
        async with kit:
            pass

    assert "Verifications were not resumed" in caplog.text


def _ends(fired: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, str]]:
    return [(d["reason"], d["ended"]) for t, d in fired if t == HookTrigger.ON_VERIFICATION_ENDED]


async def test_a_request_left_unanswered_ends_when_it_expires() -> None:
    """Its end is announced when it happens, and the person reads as having
    no request open: the next message calls for a new one (RFC §11.7.5)."""
    kit, fired = await _kit(request_ttl=0.02)
    await kit.request_verification("id-alice", room_id="r1", channel_id="sms")

    await _eventually(lambda: _ends(fired), timeout=2.0)

    assert _ends(fired) == [("expired", "request")]
    state = await kit.verification_of("id-alice")
    assert state.status == VerificationStatus.UNVERIFIED and state.request_id is None
    await kit.close()


async def test_a_verified_person_s_unanswered_request_leaves_the_verification() -> None:
    kit, fired = await _kit(ttl=5.0, request_ttl=0.02)
    await _verified(kit)
    await kit.request_verification("id-alice", room_id="r1", channel_id="sms")

    await _eventually(lambda: _ends(fired), timeout=2.0)

    assert _ends(fired) == [("expired", "request")]
    state = await kit.verification_of("id-alice")
    assert state.status == VerificationStatus.VERIFIED and state.request_id is None
    await kit.close()


async def test_a_request_that_expired_while_nothing_ran_reads_as_ended() -> None:
    """No timer runs for it in this process: it reads as ended, and the new
    request announces it as expired, not as cancelled."""
    store = InMemoryStore()
    first, _ = await _kit(store, request_ttl=0.05)
    request, _ = await first.request_verification("id-alice", room_id="r1", channel_id="sms")
    await first.close()  # its timer dies with it
    await _past(request.expires_at)

    # The new request lasts: the end announced is the old one's, not its own.
    second, fired = await _kit(store, request_ttl=5.0)
    state = await second.verification_of("id-alice")
    assert state.status == VerificationStatus.UNVERIFIED and state.request_id is None

    await second.request_verification("id-alice", room_id="r1", channel_id="sms")
    await _eventually(lambda: _ends(fired))
    assert _ends(fired)[0] == ("expired", "request")
    await second.close()


async def test_a_completed_request_ends_nothing_at_its_expiry() -> None:
    kit, fired = await _kit(ttl=5.0, request_ttl=0.05)
    request, secret = await kit.request_verification("id-alice", room_id="r1", channel_id="sms")
    await kit.complete_verification(request.id, secret)
    await _past(request.expires_at)  # its timer, had it not been stopped, would have fired

    assert _ends(fired) == []
    await kit.close()
