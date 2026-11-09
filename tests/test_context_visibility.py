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
from roomkit.core.visibility import committed_visibility, visible_events
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    Access,
    ChannelCategory,
    ChannelType,
    EventStatus,
    EventType,
    HookExecution,
    HookTrigger,
    Visibility,
)
from roomkit.models.event import (
    DeleteContent,
    EditContent,
    EventSource,
    MediaContent,
    RoomEvent,
    TextContent,
)
from roomkit.models.hook import HookResult
from roomkit.models.room import Room
from roomkit.models.store_filter import EventFilter
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
    """The card's success criterion: each hidden scope x set on the event / on the binding."""

    @pytest.mark.parametrize("scope", ["transport", "none", "ws2"])
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
        # answer. Ordinary history survives the detach; an event for no
        # channel stays hidden because it says so on the event itself.
        kit, provider = await _room()

        await kit.process_inbound(
            InboundMessage(channel_id="ws1", sender_id="u1", content=TextContent(body=MARKER))
        )
        await kit.send_event(
            "r1", "ws1", TextContent(body="INTERNAL-MARKER"), visibility=Visibility.NONE
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
        # A default timeline read serves what the room received (RFC §14.1);
        # the audit reader asks for the refused row explicitly.
        received = await kit.store.list_events("r1")
        assert MARKER not in [getattr(e.content, "body", "") for e in received]
        events = await kit.store.list_events("r1", event_filter=EventFilter(include_blocked=True))
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

    @pytest.mark.parametrize("muted", [False, True])
    async def test_an_agent_that_cannot_write_loses_its_own_refused_answers(
        self, muted: bool
    ) -> None:
        """RFC §7.5 rule 2 stores the answers of an agent that cannot write
        BLOCKED (a read-only agent's stream is read to its end, each row
        ``source_read_only``; a muted one's is closed unread); nobody received
        them, so they are not history the agent may continue from — the
        room's other turns still are."""
        kit, provider = await _room()
        await kit.set_access("r1", "ai1", Access.READ_ONLY)
        if muted:
            await kit.mute("r1", "ai1")

        await kit.process_inbound(
            InboundMessage(channel_id="ws1", sender_id="u1", content=TextContent(body="first"))
        )
        await kit.process_inbound(
            InboundMessage(channel_id="ws1", sender_id="u1", content=TextContent(body="second"))
        )

        received = await kit.store.list_events("r1")
        assert [e for e in received if e.source.channel_id == "ai1"] == []
        events = await kit.store.list_events("r1", event_filter=EventFilter(include_blocked=True))
        refused = [e for e in events if e.source.channel_id == "ai1"]
        assert [e.status for e in refused] == [EventStatus.BLOCKED] * 2
        assert {e.blocked_by for e in refused} == {"source_read_only"}
        # The brain kept tracking: two generations, the second one carrying
        # the user's first turn and none of the agent's refused answers.
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


class TestCommittedVisibility:
    """An event is committed with its own visibility intersected with its
    source binding's (RFC §7.3)."""

    @staticmethod
    def _committed(own: str, binding: str, **bindings: str) -> str:
        source = _binding("ws1", visibility=binding)
        others = [_binding(cid, visibility="all") for cid in bindings]
        return committed_visibility(own, source, others)

    def test_all_is_neutral(self) -> None:
        assert self._committed("all", "transport") == "transport"

    def test_the_event_s_own_scope_cannot_widen_its_binding(self) -> None:
        assert self._committed("all", "ws2", ws2="all") == "ws2"
        assert self._committed("ws2,ws3", "ws2", ws2="all", ws3="all") == "ws2"

    def test_internal_absorbs_and_keeps_its_mark(self) -> None:
        assert self._committed("internal", "transport") == Visibility.INTERNAL

    def test_none_absorbs(self) -> None:
        assert self._committed("none", "all") == Visibility.NONE

    def test_transport_and_intelligence_exclude_each_other(self) -> None:
        assert self._committed("transport", "intelligence") == Visibility.NONE

    def test_disjoint_lists_reach_nobody(self) -> None:
        assert self._committed("ws2", "ws3", ws2="all", ws3="all") == Visibility.NONE

    def test_a_category_keeps_the_listed_channels_of_that_category(self) -> None:
        source = _binding("ws1", visibility="ws2,ai1")
        bindings = [_binding("ws2"), _binding("ai1", category=ChannelCategory.INTELLIGENCE)]
        assert committed_visibility("transport", source, bindings) == "ws2"

    def test_a_detached_source_leaves_the_event_speaking_for_itself(self) -> None:
        assert committed_visibility(Visibility.INTERNAL, None, []) == Visibility.INTERNAL


class TestVisibleEvents:
    def _context(self, *events: RoomEvent, **bindings: str) -> RoomContext:
        return RoomContext(
            room=Room(id="r1"),
            bindings=[_binding(cid, visibility=vis) for cid, vis in bindings.items()],
            recent_events=list(events),
        )

    def test_the_stored_visibility_is_the_whole_answer(self) -> None:
        """A binding changed after the commit changes nothing of the past
        (RFC §7.5 rule 8): its events keep the visibility stamped on them."""
        ctx = self._context(
            _event("ws1", "hidden", visibility="ws2"),
            _event("ws1", "shown"),
            ws1="ws2",
            reader="all",
        )
        assert [str(e.content.body) for e in visible_events(ctx, "reader")] == ["shown"]

    def test_a_channel_always_keeps_its_own_events(self) -> None:
        ctx = self._context(_event("reader", "mine"), reader="ws1", ws1="all")
        assert [str(e.content.body) for e in visible_events(ctx, "reader")] == ["mine"]

    def test_an_unbound_reader_is_handed_no_event(self) -> None:
        ctx = self._context(_event("ws1", "theirs"), _event("ghost", "mine"), ws1="all")
        assert visible_events(ctx, "ghost") == []

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


def _last_prompt(provider: MockAIProvider) -> str:
    """What the model was shown on its latest turn, flattened."""
    return " ".join(str(m.content) for m in provider.calls[-1].messages)


async def _say(kit: RoomKit, channel: str, sender: str, body: str) -> RoomEvent:
    result = await kit.process_inbound(
        InboundMessage(
            channel_id=channel,
            sender_id=sender,
            content=TextContent(body=body),
            raw_payload={"text": body},
        )
    )
    assert result.event is not None
    return result.event


async def _change(kit: RoomKit, content: EditContent | DeleteContent) -> None:
    kind = EventType.EDIT if isinstance(content, EditContent) else EventType.DELETE
    result = await kit.process_inbound(
        InboundMessage(channel_id="ws1", sender_id="u1", event_type=kind, content=content)
    )
    assert result.blocked is False


class TestDeletedContentStaysDeleted:
    """RFC §7.5 rule 8, §10.3: a soft-deleted event stays in every history a
    channel is handed, its content withheld; its stored record keeps it."""

    async def test_an_agent_s_turn_after_a_delete_does_not_read_the_deleted_text(
        self,
    ) -> None:
        kit, provider = await _room()
        said = await _say(kit, "ws1", "u1", MARKER)
        await _change(kit, DeleteContent(target_event_id=said.id))

        await _say(kit, "ws2", "u2", "hello")

        assert MARKER not in _last_prompt(provider)
        stored = await kit.store.get_event(said.id)
        assert stored is not None and stored.content == TextContent(body=MARKER)
        assert stored.source.raw_payload == {"text": MARKER}
        await kit.close()

    async def test_the_deleted_event_keeps_its_place_in_the_history(self) -> None:
        kit, _provider = await _room()
        said = await _say(kit, "ws1", "u1", MARKER)
        await _change(kit, DeleteContent(target_event_id=said.id))

        history = visible_events(await kit._build_context("r1"), "ai1")

        held = next(e for e in history if e.id == said.id)
        assert (held.index, held.content, held.source.raw_payload) == (
            said.index,
            TextContent(body=""),
            {},
        )
        assert held.metadata.get("deleted") is True
        timeline = await kit.store.list_events("r1")
        assert [e.content for e in timeline if e.type == EventType.DELETE] == [
            DeleteContent(target_event_id=said.id)
        ]
        await kit.close()

    async def test_hooks_still_read_what_was_deleted(self) -> None:
        """The hooks are the host's, which owns the record (RFC §10.3)."""
        kit, _provider = await _room()
        seen: list[RoomContext] = []

        @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC)
        async def audit(event: RoomEvent, ctx: RoomContext) -> None:
            seen.append(ctx)

        said = await _say(kit, "ws1", "u1", MARKER)
        await _change(kit, DeleteContent(target_event_id=said.id))
        await _say(kit, "ws2", "u2", "hello")

        held = [e for e in seen[-1].recent_events if e.id == said.id]
        assert [e.content for e in held] == [TextContent(body=MARKER)]
        await kit.close()

    async def test_an_edit_s_new_content_is_still_read(self) -> None:
        kit, provider = await _room()
        said = await _say(kit, "ws1", "u1", "draft")
        await _change(
            kit,
            EditContent(target_event_id=said.id, new_content=TextContent(body=MARKER)),
        )

        await _say(kit, "ws2", "u2", "hello")

        assert MARKER in _last_prompt(provider)
        await kit.close()

    async def test_an_edit_of_a_deleted_event_has_its_new_content_withheld(self) -> None:
        kit, provider = await _room()
        said = await _say(kit, "ws1", "u1", "draft")
        await _change(
            kit,
            EditContent(target_event_id=said.id, new_content=TextContent(body=MARKER)),
        )
        await _change(kit, DeleteContent(target_event_id=said.id))

        await _say(kit, "ws2", "u2", "hello")

        assert MARKER not in _last_prompt(provider)
        await kit.close()


class TestWithheldContent:
    def _context(self, *events: RoomEvent) -> RoomContext:
        return RoomContext(
            room=Room(id="r1"), bindings=[_binding("reader")], recent_events=list(events)
        )

    def test_a_deleted_target_is_withheld_with_its_edits(self) -> None:
        target = _event("ws1", MARKER).model_copy(update={"metadata": {"deleted": True}})
        edit = _event("ws1", "").model_copy(
            update={
                "type": EventType.EDIT,
                "content": EditContent(
                    target_event_id=target.id,
                    new_content=MediaContent(url="https://x.test/a.png", mime_type="image/png"),
                ),
            }
        )
        delete = _event("ws1", "").model_copy(
            update={"type": EventType.DELETE, "content": DeleteContent(target_event_id=target.id)}
        )

        held = visible_events(self._context(target, edit, delete), "reader")

        assert [e.id for e in held] == [target.id, edit.id, delete.id]
        assert held[0].content == TextContent(body="")
        assert held[1].content == EditContent(
            target_event_id=target.id, new_content=TextContent(body="")
        )
        assert held[2].content == delete.content

    @pytest.mark.parametrize("status", [EventStatus.BLOCKED, EventStatus.DELIVERED])
    def test_a_delete_row_alone_withholds_nothing(self, status: EventStatus) -> None:
        """Only the mark an accepted delete's commit writes on its target
        withholds it: a row refused, or one stored without step 8a (a host's
        ``commit_event``, a child room's trace), checked no authority."""
        target = _event("ws1", MARKER)
        delete = _event("ws1", "", status=status).model_copy(
            update={"type": EventType.DELETE, "content": DeleteContent(target_event_id=target.id)}
        )

        held = visible_events(self._context(target, delete), "reader")

        assert held[0].content == target.content

    async def test_a_delete_row_committed_outside_the_pipeline_hides_nothing(self) -> None:
        kit, _provider = await _room()
        said = await _say(kit, "ws1", "u1", MARKER)
        row = RoomEvent(
            room_id="r1",
            type=EventType.DELETE,
            source=EventSource(channel_id="ws2", channel_type=ChannelType.WEBSOCKET),
            content=DeleteContent(target_event_id=said.id),
            status=EventStatus.DELIVERED,
        )
        await kit.commit_event("r1", row)

        # A history holding the row, as a store whose conversation read serves
        # every row would hand it.
        context = await kit._build_context("r1")
        context = context.model_copy(update={"recent_events": [*context.recent_events, row]})
        history = visible_events(context, "ai1")

        assert [e.content for e in history if e.id == said.id] == [TextContent(body=MARKER)]
        await kit.close()
