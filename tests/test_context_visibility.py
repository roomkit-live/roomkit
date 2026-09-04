"""Visibility holds on the next turn too (RFC §7.5 rule 8).

The broadcast keeps `visibility`'s promise; for a long time the turn after
broke it. A channel the visibility hid an event from was not called at
broadcast — correct — but the event stayed in ``RoomContext.recent_events``,
which the memory provider re-read verbatim on the next turn and handed to the
model. These tests are that leak, closed and kept closed.
"""

from __future__ import annotations

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.channels.websocket import WebSocketChannel
from roomkit.core.framework import RoomKit
from roomkit.core.visibility import effective_visibility, visible_events
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    Access,
    ChannelCategory,
    ChannelType,
    EventStatus,
    HookTrigger,
    Visibility,
)
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.hook import HookResult
from roomkit.models.room import Room
from roomkit.providers.ai.mock import MockAIProvider

MARKER = "SECRET-MARKER"


def _prompted(provider: MockAIProvider) -> str:
    """Everything the model was ever shown, flattened."""
    return " ".join(str(m.content) for call in provider.calls for m in call.messages)


async def _room(
    *, source_visibility: str = Visibility.ALL, ai_visibility: str = Visibility.ALL
) -> tuple[RoomKit, MockAIProvider]:
    """ws1 (the source under test) + ws2 (a plainly visible second transport) + ai1."""
    kit = RoomKit()
    provider = MockAIProvider(responses=["ok"])
    kit.register_channel(WebSocketChannel("ws1"))
    kit.register_channel(WebSocketChannel("ws2"))
    kit.register_channel(AIChannel("ai1", provider=provider))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ws1", visibility=source_visibility)
    await kit.attach_channel("r1", "ws2")
    await kit.attach_channel(
        "r1", "ai1", category=ChannelCategory.INTELLIGENCE, visibility=ai_visibility
    )
    return kit, provider


class TestHiddenEventsStayHidden:
    """The card's success criterion: four scopes x set on the event / on the binding."""

    @pytest.mark.parametrize("scope", ["transport", "none", "internal", "ws2"])
    @pytest.mark.parametrize("on_binding", [False, True])
    async def test_hidden_event_never_reaches_the_model(
        self, scope: str, on_binding: bool
    ) -> None:
        kit, provider = await _room(source_visibility=scope if on_binding else Visibility.ALL)

        await kit.process_inbound(
            InboundMessage(
                channel_id="ws1",
                sender_id="u1",
                content=TextContent(body=MARKER),
                **({} if on_binding else {"visibility": scope}),
            )
        )
        # The AI is correctly skipped at broadcast — that half already worked.
        assert provider.calls == []

        # A second turn on a plainly visible channel, so ws1's binding keeps
        # its scope throughout: the next turn is what used to leak.
        await kit.process_inbound(
            InboundMessage(channel_id="ws2", sender_id="u2", content=TextContent(body="hello"))
        )

        assert provider.calls, "the second turn must reach the model"
        assert MARKER not in _prompted(provider)
        await kit.close()

    async def test_ordinary_history_still_reaches_the_model(self) -> None:
        # The guard against over-filtering: "all" is the common case and must
        # arrive, or the fix would have traded a leak for amnesia.
        kit, provider = await _room()

        await kit.process_inbound(
            InboundMessage(channel_id="ws1", sender_id="u1", content=TextContent(body=MARKER))
        )
        await kit.process_inbound(
            InboundMessage(channel_id="ws2", sender_id="u2", content=TextContent(body="hello"))
        )

        assert MARKER in _prompted(provider)
        await kit.close()

    async def test_an_agent_still_recalls_its_own_whispers(self) -> None:
        # RFC §7.4 assistant pattern: an AI bound visibility="ws1" produces
        # events whose scope excludes itself. Filtering without the source
        # exemption would erase its own turns from its own prompt.
        kit, provider = await _room(ai_visibility="ws1")
        provider.responses = ["MY-OWN-ANSWER"]

        for body in ("first", "second"):
            await kit.process_inbound(
                InboundMessage(channel_id="ws1", sender_id="u1", content=TextContent(body=body))
            )

        assert "MY-OWN-ANSWER" in _prompted(provider)
        await kit.close()

    async def test_a_detached_source_does_not_erase_its_history(self) -> None:
        # No binding left to resolve: the event's own scope is the whole
        # answer. Ordinary history survives the detach; an internal event
        # stays hidden because it says so on the event itself.
        kit, provider = await _room()

        await kit.process_inbound(
            InboundMessage(channel_id="ws1", sender_id="u1", content=TextContent(body=MARKER))
        )
        await kit.send_event(
            "r1", "ws1", TextContent(body="INTERNAL-MARKER"), visibility=Visibility.INTERNAL
        )
        assert await kit.detach_channel("r1", "ws1")

        await kit.process_inbound(
            InboundMessage(channel_id="ws2", sender_id="u2", content=TextContent(body="hello"))
        )

        prompted = _prompted(provider)
        assert MARKER in prompted
        assert "INTERNAL-MARKER" not in prompted
        await kit.close()

    async def test_hooks_still_see_the_whole_timeline(self) -> None:
        # Host code, in the integrator's process, holding the store anyway:
        # filtering it would forbid nothing and break legitimate readers.
        kit, _ = await _room()
        seen: list[str] = []

        @kit.hook(HookTrigger.BEFORE_BROADCAST)
        async def capture(event: RoomEvent, ctx: RoomContext) -> HookResult:
            seen.extend(
                str(e.content.body) for e in ctx.recent_events if hasattr(e.content, "body")
            )
            return HookResult.allow()

        await kit.process_inbound(
            InboundMessage(
                channel_id="ws1",
                sender_id="u1",
                content=TextContent(body=MARKER),
                visibility=Visibility.NONE,
            )
        )
        await kit.process_inbound(
            InboundMessage(channel_id="ws2", sender_id="u2", content=TextContent(body="hello"))
        )

        assert MARKER in seen
        await kit.close()


class TestRefusedEventsStayRefused:
    """A message a hook blocked is stored BLOCKED and delivered to nobody
    (RFC §10.1 step 10); it does not come back to the model as history one
    turn later (§7.5 rule 8), on any store, because the filter is per reader."""

    async def _spam_filtered_room(self) -> tuple[RoomKit, MockAIProvider]:
        kit, provider = await _room()

        @kit.hook(HookTrigger.BEFORE_BROADCAST)
        async def refuse_spam(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if MARKER in getattr(event.content, "body", ""):
                return HookResult.block("spam")
            return HookResult.allow()

        return kit, provider

    async def test_a_message_a_hook_blocked_never_reaches_the_model(self) -> None:
        kit, provider = await self._spam_filtered_room()
        await kit.process_inbound(
            InboundMessage(channel_id="ws1", sender_id="u1", content=TextContent(body="hello"))
        )
        refused = await kit.process_inbound(
            InboundMessage(channel_id="ws1", sender_id="u1", content=TextContent(body=MARKER))
        )
        assert refused.blocked is True
        events = await kit.store.list_events("r1")
        assert [e.status for e in events if getattr(e.content, "body", "") == MARKER] == [
            EventStatus.BLOCKED
        ]

        await kit.process_inbound(
            InboundMessage(channel_id="ws1", sender_id="u1", content=TextContent(body="and now?"))
        )
        regenerated = await kit.regenerate_response("r1")

        assert regenerated is not None and regenerated.event is not None
        # The refused message is in the store (audit) and out of every prompt.
        assert MARKER in " ".join(str(getattr(e.content, "body", "")) for e in events)
        assert MARKER not in _prompted(provider)
        assert "hello" in _prompted(provider)
        await kit.close()

    async def test_a_muted_agent_loses_its_own_silenced_answers(self) -> None:
        """RFC §7.5 rule 2 stores a muted channel's answers BLOCKED
        (``source_muted``); nobody received them, so they are not history the
        agent may continue from — the room's other turns still are."""
        kit, provider = await _room()
        await kit.mute("r1", "ai1")

        await kit.process_inbound(
            InboundMessage(channel_id="ws1", sender_id="u1", content=TextContent(body="first"))
        )
        await kit.process_inbound(
            InboundMessage(channel_id="ws1", sender_id="u1", content=TextContent(body="second"))
        )

        events = await kit.store.list_events("r1")
        silenced = [e for e in events if e.source.channel_id == "ai1"]
        assert [e.status for e in silenced] == [EventStatus.BLOCKED] * 2
        assert {e.blocked_by for e in silenced} == {"source_muted"}
        # The brain kept tracking: two generations, the second one carrying
        # the user's first turn and none of the agent's silenced answers.
        assert len(provider.calls) == 2
        last = provider.calls[-1].messages
        assert [m.content for m in last] == ["first", "second"]
        assert all(m.role == "user" for m in last)
        await kit.close()

    async def test_a_read_only_source_message_never_reaches_the_model(self) -> None:
        """RFC §10.1 step 11: a read-only source's message is stored BLOCKED
        (``source_read_only``) and never broadcast; the next turn does not
        hand it to the model either."""
        kit, provider = await _room()
        await kit.set_access("r1", "ws1", Access.READ_ONLY)

        refused = await kit.process_inbound(
            InboundMessage(channel_id="ws1", sender_id="u1", content=TextContent(body=MARKER))
        )
        await kit.process_inbound(
            InboundMessage(channel_id="ws2", sender_id="u2", content=TextContent(body="hello"))
        )

        assert refused.blocked is True
        assert refused.event is not None and refused.event.blocked_by == "source_read_only"
        assert "hello" in _prompted(provider)
        assert MARKER not in _prompted(provider)
        await kit.close()

    async def test_hooks_still_see_the_blocked_record(self) -> None:
        kit, _provider = await self._spam_filtered_room()
        seen: list[tuple[str, EventStatus]] = []

        @kit.hook(HookTrigger.BEFORE_BROADCAST)
        async def capture(event: RoomEvent, ctx: RoomContext) -> HookResult:
            seen.extend(
                (str(e.content.body), e.status)
                for e in ctx.recent_events
                if hasattr(e.content, "body")
            )
            return HookResult.allow()

        await kit.process_inbound(
            InboundMessage(channel_id="ws1", sender_id="u1", content=TextContent(body=MARKER))
        )
        await kit.process_inbound(
            InboundMessage(channel_id="ws2", sender_id="u2", content=TextContent(body="hello"))
        )

        assert (MARKER, EventStatus.BLOCKED) in seen
        await kit.close()


def _binding(channel_id: str, **kwargs: object) -> ChannelBinding:
    return ChannelBinding(
        channel_id=channel_id, room_id="r1", channel_type=ChannelType.WEBSOCKET, **kwargs
    )


def _event(
    source: str,
    body: str,
    visibility: str = Visibility.ALL,
    status: EventStatus = EventStatus.DELIVERED,
) -> RoomEvent:
    return RoomEvent(
        room_id="r1",
        source=EventSource(channel_id=source, channel_type=ChannelType.WEBSOCKET),
        content=TextContent(body=body),
        visibility=visibility,
        status=status,
    )


class TestEffectiveVisibility:
    """The event's own scope wins; the binding answers for the default."""

    def test_a_non_default_event_scope_wins(self) -> None:
        event = _event("ws1", "x", visibility=Visibility.NONE)
        assert effective_visibility(event, _binding("ws1", visibility="all")) == Visibility.NONE

    def test_the_binding_answers_for_the_default(self) -> None:
        # The stored event keeps "all": the router's stamp lands on a copy made
        # after the commit, so storage never sees the binding's scope.
        event = _event("ws1", "x")
        assert effective_visibility(event, _binding("ws1", visibility="transport")) == "transport"

    def test_a_detached_source_leaves_the_event_speaking_for_itself(self) -> None:
        assert effective_visibility(_event("ws1", "x"), None) == Visibility.ALL
        assert (
            effective_visibility(_event("ws1", "x", visibility=Visibility.INTERNAL), None)
            == Visibility.INTERNAL
        )


class TestVisibleEvents:
    def _context(self, *events: RoomEvent, **bindings: str) -> RoomContext:
        return RoomContext(
            room=Room(id="r1"),
            bindings=[_binding(cid, visibility=vis) for cid, vis in bindings.items()],
            recent_events=list(events),
        )

    def test_a_binding_scope_hides_the_event_from_a_later_reader(self) -> None:
        ctx = self._context(
            _event("ws1", "hidden"), _event("ws2", "shown"), ws1="ws2", ws2="all", reader="all"
        )
        assert [str(e.content.body) for e in visible_events(ctx, "reader")] == ["shown"]

    def test_a_channel_always_keeps_its_own_events(self) -> None:
        ctx = self._context(_event("reader", "mine"), reader="ws1", ws1="all")
        assert [str(e.content.body) for e in visible_events(ctx, "reader")] == ["mine"]

    def test_an_unbound_reader_sees_only_what_it_produced(self) -> None:
        ctx = self._context(_event("ws1", "theirs"), _event("ghost", "mine"), ws1="all")
        assert [str(e.content.body) for e in visible_events(ctx, "ghost")] == ["mine"]

    def test_a_blocked_event_reaches_no_reader(self) -> None:
        ctx = self._context(
            _event("ws1", "refused", status=EventStatus.BLOCKED),
            _event("ws1", "accepted"),
            ws1="all",
            reader="all",
        )
        assert [str(e.content.body) for e in visible_events(ctx, "reader")] == ["accepted"]

    def test_a_channel_does_not_keep_its_own_blocked_event(self) -> None:
        # The own-events exception is about what a channel may know; a turn
        # the room refused is not one it may continue from.
        ctx = self._context(
            _event("reader", "mine"),
            _event("reader", "mine, refused", status=EventStatus.BLOCKED),
            reader="all",
        )
        assert [str(e.content.body) for e in visible_events(ctx, "reader")] == ["mine"]
        ghost = self._context(_event("ghost", "mine, refused", status=EventStatus.BLOCKED))
        assert visible_events(ghost, "ghost") == []
