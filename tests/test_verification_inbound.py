"""The inbound pipeline labels, holds and consumes codes (RFC §10.1 step 5b, §11.7.3-4)."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import pytest

from roomkit import (
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomKit,
    TextContent,
    VerificationPolicy,
    VerificationStatus,
)
from roomkit.channels import SMSChannel
from roomkit.channels.ai import AIChannel
from roomkit.core.mixins import verification as verification_mixin
from roomkit.core.mixins.verification import HELD_KEY
from roomkit.identity.mock import MockIdentityResolver
from roomkit.models.identity import Identity
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.sms.mock import MockSMSProvider
from tests.test_verification_lifecycle import VerificationClock

ALICE = "+15550000001"


_DEFAULT = VerificationPolicy()


@pytest.fixture(autouse=True)
def _cheap_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    """scrypt at a test's cost (its production factor is pinned in
    tests/test_identity_verification.py): a short TTL is not spent hashing."""
    monkeypatch.setattr(verification_mixin, "_SCRYPT_N", 2**4)


def _kit(policy: VerificationPolicy | None = _DEFAULT) -> tuple[RoomKit, MockAIProvider]:
    ai = MockAIProvider(responses=["ok"] * 10)
    kit = RoomKit(
        identity_resolver=MockIdentityResolver(mapping={ALICE: Identity(id="id-alice")}),
        verification_policy=policy,
    )
    kit.register_channel(SMSChannel("sms", provider=MockSMSProvider()))
    kit.register_channel(AIChannel("ai", provider=ai))

    @kit.hook(HookTrigger.ON_ROOM_CREATED)
    async def agent_joins(event: Any, ctx: Any) -> None:
        await kit.attach_channel(ctx.room.id, "ai")

    return kit, ai


async def _text(kit: RoomKit, body: str) -> Any:
    return await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id=ALICE, content=TextContent(body=body))
    )


async def _settle(kit: RoomKit) -> None:
    """Until the ASYNC observers the message fired have run; the agent's turn
    ran within process_inbound."""
    await kit.hook_engine.drain_observers()


def _prompt(ai: MockAIProvider) -> str:
    return " ".join(str(m.content) for m in ai.calls[-1].messages)


async def test_an_unverified_message_is_held_from_the_agent() -> None:
    kit, ai = _kit()
    asked: list[str] = []

    @kit.hook(HookTrigger.ON_VERIFICATION_REQUIRED)
    async def required(event: Any, ctx: Any) -> HookResult:
        asked.append(event.content.body)
        return HookResult.allow()  # watching only: the message stays held

    result = await _text(kit, "Did my transfer go through?")
    await _settle(kit)

    stored = await kit.store.get_event(result.event.id)
    assert stored is not None
    assert stored.visibility == "transport" and HELD_KEY in stored.metadata
    assert stored.source.verification == VerificationStatus.UNVERIFIED
    assert asked == ["Did my transfer go through?"]
    assert ai.calls == []
    await kit.close()


async def test_a_hook_lets_a_general_question_through_labelled() -> None:
    kit, ai = _kit()

    @kit.hook(HookTrigger.ON_VERIFICATION_REQUIRED)
    async def general_questions(event: Any, ctx: Any) -> HookResult:
        if "hours" in event.content.body:
            return HookResult.allow_unverified()
        return HookResult.allow()

    held = await _text(kit, "My account number is 12345, what's my balance?")
    passed = await _text(kit, "What are your opening hours?")
    await _settle(kit)

    stored = await kit.store.get_event(passed.event.id)
    assert stored is not None and stored.visibility == "all"
    assert stored.source.verification == VerificationStatus.UNVERIFIED
    assert len(ai.calls) == 1
    assert "opening hours" in _prompt(ai)
    assert "12345" not in _prompt(ai)  # the held message is not in the agent's context
    assert (await kit.store.get_event(held.event.id)).visibility == "transport"  # type: ignore[union-attr]
    await kit.close()


async def test_a_hook_blocks_a_message_before_it_is_stored() -> None:
    kit, ai = _kit()

    @kit.hook(HookTrigger.ON_VERIFICATION_REQUIRED)
    async def refuse(event: Any, ctx: Any) -> HookResult:
        return HookResult.block("verify first")

    result = await _text(kit, "Close my account")

    assert result.blocked and result.reason == "verify first"
    assert result.event is None
    await kit.close()


async def test_a_verified_sender_reaches_the_agent() -> None:
    kit, ai = _kit()
    first = await _text(kit, "hello")
    room_id = first.event.room_id
    request, secret = await kit.request_verification("id-alice", room_id=room_id, channel_id="sms")
    await kit.complete_verification(request.id, secret)
    ai.calls.clear()

    result = await _text(kit, "Did my transfer go through?")
    await _settle(kit)

    stored = await kit.store.get_event(result.event.id)
    assert stored is not None and stored.visibility == "all"
    assert stored.source.verification == VerificationStatus.VERIFIED
    assert ai.calls
    await kit.close()


async def test_a_code_reply_is_consumed_and_verifies() -> None:
    kit, ai = _kit()
    first = await _text(kit, "hello")
    room_id = first.event.room_id
    request, code = await kit.request_verification(
        "id-alice", room_id=room_id, channel_id="sms", method="code"
    )
    wrong = str((int(code) + 1) % 10**6).zfill(6)

    refused = await _text(kit, wrong)
    accepted = await _text(kit, f" {code} ")

    assert refused.blocked and refused.reason == "verification_code"
    assert accepted.blocked and accepted.reason == "verification_code"
    events = await kit.store.list_events(room_id, limit=50)
    bodies = " ".join(getattr(e.content, "body", "") for e in events)
    assert code not in bodies and wrong not in bodies
    assert (await kit.verification_of("id-alice")).status == VerificationStatus.VERIFIED
    stored = await kit.store.get_verification_request(request.id)
    assert stored is not None and stored.attempts == 1
    await kit.close()


async def test_digits_with_no_code_request_open_are_an_ordinary_message() -> None:
    kit, _ = _kit()

    result = await _text(kit, "123456")

    assert not result.blocked and result.event is not None
    assert result.event.visibility == "transport"  # held like any unverified message
    await kit.close()


async def test_a_kit_without_a_policy_keeps_its_pipeline() -> None:
    kit, ai = _kit(policy=None)

    result = await _text(kit, "Did my transfer go through?")
    await _settle(kit)

    assert result.event.source.verification is None
    assert result.event.visibility == "all"
    assert ai.calls
    await kit.close()


async def test_a_policy_that_does_not_hold_only_labels() -> None:
    kit, ai = _kit(policy=VerificationPolicy(hold_unverified=False))

    result = await _text(kit, "Did my transfer go through?")
    await _settle(kit)

    assert result.event.source.verification == VerificationStatus.UNVERIFIED
    assert result.event.visibility == "all"
    assert ai.calls
    await kit.close()


async def test_holding_never_widens_what_a_message_reached() -> None:
    """The SMS binding writes to the advisor's console only: held, it stays so."""
    from roomkit.channels import WebSocketChannel

    kit, ai = _kit()
    kit.register_channel(WebSocketChannel("advisor"))
    kit.register_channel(WebSocketChannel("other"))
    first = await _text(kit, "hello")
    room_id = first.event.room_id
    await kit.attach_channel(room_id, "advisor")
    await kit.attach_channel(room_id, "other")
    binding = await kit.store.get_binding(room_id, "sms")
    assert binding is not None
    await kit.store.update_binding(binding.model_copy(update={"visibility": "advisor"}))

    result = await _text(kit, "Did my transfer go through?")

    stored = await kit.store.get_event(result.event.id)
    assert stored is not None and stored.visibility == "advisor"
    await kit.close()


async def test_an_unverified_edit_cannot_rewrite_a_message() -> None:
    from roomkit.models.event import EditContent

    kit, ai = _kit()
    first = await _text(kit, "hello")
    room_id = first.event.room_id
    request, secret = await kit.request_verification("id-alice", room_id=room_id, channel_id="sms")
    await kit.complete_verification(request.id, secret)
    original = await _text(kit, "Please transfer 100 dollars")
    await kit.revoke_verification("id-alice")

    edit = await kit.process_inbound(
        InboundMessage(
            channel_id="sms",
            sender_id=ALICE,
            event_type="edit",
            content=EditContent(
                target_event_id=original.event.id,
                new_content=TextContent(body="Please transfer 10000 dollars"),
            ),
        )
    )

    assert edit.blocked and edit.reason == "verification_required"
    kept = await kit.store.get_event(original.event.id)
    assert kept is not None and kept.content.body == "Please transfer 100 dollars"  # type: ignore[union-attr]
    await kit.close()


async def test_a_verification_in_one_organization_does_not_label_another() -> None:
    """Alice proved who she is to organization A; writing to organization B's
    room, she is not verified there."""
    ai = MockAIProvider(responses=["ok"] * 10)
    kit = RoomKit(
        identity_resolver=MockIdentityResolver(
            mapping={ALICE: Identity(id="id-alice", organization_id="org-a")}
        ),
        verification_policy=VerificationPolicy(),
    )
    kit.register_channel(SMSChannel("sms", provider=MockSMSProvider()))
    kit.register_channel(AIChannel("ai", provider=ai))
    message = InboundMessage(channel_id="sms", sender_id=ALICE, content=TextContent(body="hi"))
    await kit.process_inbound(message, room_id="room-a", organization_id="org-a")
    request, secret = await kit.request_verification(
        "id-alice", room_id="room-a", channel_id="sms", organization_id="org-a"
    )
    await kit.complete_verification(request.id, secret, organization_id="org-a")
    await kit.create_room(room_id="room-b", organization_id="org-b")
    await kit.attach_channel("room-b", "ai")

    result = await kit.process_inbound(
        InboundMessage(
            channel_id="sms", sender_id=ALICE, content=TextContent(body="What's my balance?")
        ),
        room_id="room-b",
        organization_id="org-b",
    )
    await _settle(kit)

    assert result.event.source.verification == VerificationStatus.UNVERIFIED
    assert result.event.visibility == "transport"
    assert ai.calls == []
    await kit.close()


async def test_a_hook_reads_whose_verification_to_ask_for() -> None:
    """The first message's sender is not a participant yet: the hook still
    learns whom to verify, and that a request is open on the next message."""
    kit, _ = _kit()
    opened: list[str] = []

    @kit.hook(HookTrigger.ON_VERIFICATION_REQUIRED)
    async def ask(event: Any, ctx: Any) -> HookResult:
        state = await kit.sender_verification(event, ctx)
        assert state is not None
        if state.request_id is None:
            request, _ = await kit.request_verification(
                state.identity_id,
                room_id=ctx.room.id,
                channel_id=event.source.channel_id,
                window_start=event.created_at,
            )
            opened.append(request.identity_id)
        return HookResult.allow()

    await _text(kit, "Did my transfer go through?")
    await _text(kit, "Hello?")

    assert opened == ["id-alice"]
    assert (await kit.verification_of("id-alice")).status == VerificationStatus.PENDING
    await kit.close()


async def test_an_unlabelled_event_has_no_sender_verification() -> None:
    kit, _ = _kit(policy=None)
    seen: list[Any] = []

    @kit.hook(HookTrigger.BEFORE_BROADCAST, channel_ids={"sms"})
    async def read(event: Any, ctx: Any) -> HookResult:
        seen.append(await kit.sender_verification(event, ctx))
        return HookResult.allow()

    await _text(kit, "hello")

    assert seen == [None]
    await kit.close()


async def test_a_new_link_goes_out_once_the_first_expired_unanswered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guide's hook asks for a request when none is open: one left
    unanswered past its expiry is not open, so the next message asks again."""
    clock = VerificationClock()
    monkeypatch.setattr(verification_mixin, "datetime", clock)
    monkeypatch.setattr(verification_mixin, "asyncio", clock)
    kit, _ = _kit(VerificationPolicy(request_ttl=0.2))
    links: list[str] = []
    expiries: list[datetime] = []

    @kit.hook(HookTrigger.ON_VERIFICATION_REQUIRED)
    async def required(event: Any, ctx: Any) -> HookResult:
        state = await kit.sender_verification(event, ctx)
        if state is not None and state.request_id is None:
            request, _secret = await kit.request_verification(
                state.identity_id, room_id=ctx.room.id, channel_id="sms"
            )
            links.append(request.id)
            expiries.append(request.expires_at)
        return HookResult.allow()

    await _text(kit, "hello")
    # The first request's time runs out, and the clock moves no further: the
    # second is read pending, however long the test takes.
    clock.past(expiries[0])

    async def first_link_expired() -> bool:
        state = await kit.verification_of("id-alice")
        return state is not None and state.request_id is None

    async with asyncio.timeout(5):  # its expiry timer ends the request unanswered
        while not await first_link_expired():
            await asyncio.sleep(0.01)
    await _text(kit, "still there?")

    assert len(links) == 2
    state = await kit.verification_of("id-alice")
    assert state.status == VerificationStatus.PENDING and state.request_id == links[1]
    await kit.close()
