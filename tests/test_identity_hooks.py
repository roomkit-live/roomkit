"""Tests for identity hook firing points (Area 3.3)."""

from __future__ import annotations

from roomkit.core.framework import RoomKit
from roomkit.identity.base import IdentityResolver
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    HookExecution,
    HookTrigger,
    IdentificationStatus,
)
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.identity import Identity, IdentityResult
from tests.test_framework import SimpleChannel


class AmbiguousResolver(IdentityResolver):
    """Resolver that always returns ambiguous/pending."""

    async def resolve(self, message: InboundMessage, context: RoomContext) -> IdentityResult:
        return IdentityResult(
            status=IdentificationStatus.PENDING,
            candidates=[
                Identity(id="id1", display_name="Alice"),
                Identity(id="id2", display_name="Bob"),
            ],
        )


class UnknownResolver(IdentityResolver):
    """Resolver that always returns rejected/unknown."""

    async def resolve(self, message: InboundMessage, context: RoomContext) -> IdentityResult:
        return IdentityResult(
            status=IdentificationStatus.REJECTED,
            message="Unknown sender",
        )


class IdentifiedResolver(IdentityResolver):
    """Resolver that always identifies successfully."""

    async def resolve(self, message: InboundMessage, context: RoomContext) -> IdentityResult:
        return IdentityResult(
            status=IdentificationStatus.IDENTIFIED,
            identity=Identity(id="known_user"),
        )


class TestIdentityAmbiguousHook:
    async def test_ambiguous_hook_fires(self) -> None:
        """ON_IDENTITY_AMBIGUOUS fires when resolver returns PENDING."""
        fired: list[RoomEvent] = []

        kit = RoomKit(identity_resolver=AmbiguousResolver())

        async def on_ambiguous(event: RoomEvent, ctx: RoomContext) -> None:
            fired.append(event)

        from roomkit.core.hooks import HookRegistration

        kit.hook_engine.register(
            HookRegistration(
                trigger=HookTrigger.ON_IDENTITY_AMBIGUOUS,
                execution=HookExecution.ASYNC,
                fn=on_ambiguous,
                name="test_ambiguous",
            )
        )

        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="unknown_user",
            content=TextContent(body="hello"),
        )
        await kit.process_inbound(msg)
        assert len(fired) == 1

    async def test_identified_does_not_fire_ambiguous(self) -> None:
        """ON_IDENTITY_AMBIGUOUS does NOT fire when resolver succeeds."""
        fired: list[RoomEvent] = []

        kit = RoomKit(identity_resolver=IdentifiedResolver())

        async def on_ambiguous(event: RoomEvent, ctx: RoomContext) -> None:
            fired.append(event)

        from roomkit.core.hooks import HookRegistration

        kit.hook_engine.register(
            HookRegistration(
                trigger=HookTrigger.ON_IDENTITY_AMBIGUOUS,
                execution=HookExecution.ASYNC,
                fn=on_ambiguous,
                name="test_ambiguous",
            )
        )

        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        await kit.process_inbound(msg)
        assert len(fired) == 0


class TestIdentityUnknownHook:
    async def test_unknown_hook_fires(self) -> None:
        """ON_IDENTITY_UNKNOWN fires when resolver returns REJECTED."""
        fired: list[RoomEvent] = []

        kit = RoomKit(identity_resolver=UnknownResolver())

        async def on_unknown(event: RoomEvent, ctx: RoomContext) -> None:
            fired.append(event)

        from roomkit.core.hooks import HookRegistration

        kit.hook_engine.register(
            HookRegistration(
                trigger=HookTrigger.ON_IDENTITY_UNKNOWN,
                execution=HookExecution.ASYNC,
                fn=on_unknown,
                name="test_unknown",
            )
        )

        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="stranger",
            content=TextContent(body="hello"),
        )
        await kit.process_inbound(msg)
        assert len(fired) == 1

    async def test_identified_does_not_fire_unknown(self) -> None:
        """ON_IDENTITY_UNKNOWN does NOT fire when resolver succeeds."""
        fired: list[RoomEvent] = []

        kit = RoomKit(identity_resolver=IdentifiedResolver())

        async def on_unknown(event: RoomEvent, ctx: RoomContext) -> None:
            fired.append(event)

        from roomkit.core.hooks import HookRegistration

        kit.hook_engine.register(
            HookRegistration(
                trigger=HookTrigger.ON_IDENTITY_UNKNOWN,
                execution=HookExecution.ASYNC,
                fn=on_unknown,
                name="test_unknown",
            )
        )

        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        await kit.process_inbound(msg)
        assert len(fired) == 0
