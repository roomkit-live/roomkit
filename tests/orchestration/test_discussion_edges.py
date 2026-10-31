"""A discussion at its edges (RFC §19.7.5): who counts as a person and who as
asked, turns of their own, the depth limit, waiting, uninstalling, failures
and cuts, visibility and tenants.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.core.framework import RoomKit
from roomkit.core.hooks import HookRegistration
from roomkit.memory.sliding_window import SlidingWindowMemory
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    Access,
    ChannelCategory,
    ChannelType,
    EventType,
    HookExecution,
    HookTrigger,
    ParticipantRole,
)
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.hook import HookResult, InjectedEvent
from roomkit.models.participant import Participant
from roomkit.orchestration.strategies.discussion import (
    Discussion,
    SpeakQueueChange,
    SpeakQueueEvent,
    _driver,
)
from roomkit.providers.ai.base import AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from tests.orchestration.test_strategy_discussion import _provider, _settle, _timeline
from tests.test_framework import SimpleChannel


class _People(SimpleChannel):
    """A transport that says who sent each message, as real transports do."""

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            source=EventSource(
                channel_id=self.channel_id,
                channel_type=self.channel_type,
                participant_id=message.sender_id,
            ),
            content=message.content,
            metadata=dict(message.metadata),
        )


async def _room(kit: RoomKit, agents: dict[str, MockAIProvider], **options: Any) -> None:
    kit.register_channel(_People("chat"))
    channels = [AIChannel(name, provider=p) for name, p in agents.items()]
    await kit.create_room(room_id="r1", orchestration=Discussion(channels, **options))
    await kit.attach_channel("r1", "chat")


async def _say(
    kit: RoomKit,
    body: str,
    *,
    sender: str = "ops",
    addressed_to: list[str] | None = None,
    event_type: EventType = EventType.MESSAGE,
    metadata: dict[str, Any] | None = None,
    chain_depth: int = 0,
) -> Any:
    return await kit.process_inbound(
        InboundMessage(
            channel_id="chat",
            sender_id=sender,
            content=TextContent(body=body),
            addressed_to=addressed_to,
            event_type=event_type,
            metadata=metadata or {},
            chain_depth=chain_depth,
        )
    )


OPS = {"sender_name": "ops"}
"""The on-call's messages, named by their transport."""


async def _person(kit: RoomKit, pid: str, name: str, role: ParticipantRole) -> None:
    await kit.store.add_participant(
        Participant(id=pid, room_id="r1", channel_id="chat", display_name=name, role=role)
    )


def _queue(kit: RoomKit) -> Any:
    queue = kit.speak_queue("r1")
    assert queue is not None
    return queue


def _follow(kit: RoomKit) -> list[tuple[SpeakQueueChange, tuple[str, ...]]]:
    seen: list[tuple[SpeakQueueChange, tuple[str, ...]]] = []

    @kit.hook(HookTrigger.ON_SPEAK_QUEUE, execution=HookExecution.ASYNC)
    async def follow(event: SpeakQueueEvent, context: RoomContext) -> None:
        seen.append((event.change, event.channel_ids))

    return seen


async def _seen(
    seen: list[tuple[SpeakQueueChange, tuple[str, ...]]],
    change: tuple[SpeakQueueChange, tuple[str, ...]],
) -> None:
    """Wait for *change* to reach the hook :func:`_follow` set, which runs on its own."""
    async with asyncio.timeout(5):
        while change not in seen:
            await asyncio.sleep(0.01)


# -- Instructions --


async def test_an_instruction_to_several_agents_gives_each_its_turn() -> None:
    a, b = _provider("a done"), _provider("b done")
    kit = RoomKit()
    await _room(kit, {"a": a, "b": b})

    await _say(kit, "summarise", addressed_to=["a", "b"], event_type=EventType.INSTRUCTION)
    await _settle(kit)

    assert (len(a.calls), len(b.calls)) == (1, 1)
    await kit.close()


async def test_an_instruction_past_the_depth_limit_is_dropped_and_reported() -> None:
    a = _provider("a done")
    kit = RoomKit()
    seen = _follow(kit)
    await _room(kit, {"a": a, "b": _provider()}, max_depth=3)

    await _say(
        kit, "hand back", addressed_to=["a"], event_type=EventType.INSTRUCTION, chain_depth=2
    )
    await _settle(kit)
    await _seen(seen, (SpeakQueueChange.INSTRUCTION_DROPPED, ("a",)))

    assert len(a.calls) == 0
    assert (SpeakQueueChange.INSTRUCTION_DROPPED, ("a",)) in seen
    assert _queue(kit).queue == () and not _queue(kit).waiting
    await kit.close()


class _Directing(AIChannel):
    """An agent whose answer is an INSTRUCTION to ``b``: not the application's."""

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if event.source.channel_id == self.channel_id:
            return ChannelOutput.empty()
        instruction = RoomEvent(
            room_id=event.room_id,
            type=EventType.INSTRUCTION,
            source=EventSource(channel_id=self.channel_id, channel_type=ChannelType.AI),
            content=TextContent(body="Summarise the incident for the customer."),
            addressed_to=["b"],
        )
        return ChannelOutput(responded=True, response_events=[instruction])


async def test_an_agent_s_answer_typed_instruction_gives_no_turn() -> None:
    """Only the application issues instructions (RFC §8.5): one agent's
    answer does not queue another for a turn in the application's name."""
    b = _provider("b answers")
    kit = RoomKit()
    kit.register_channel(_People("chat"))
    agents = [_Directing("a", provider=_provider()), AIChannel("b", provider=b)]
    await kit.create_room(room_id="r1", orchestration=Discussion(agents))
    await kit.attach_channel("r1", "chat")

    await _say(kit, "@a go")
    await _settle(kit)

    assert b.calls == []
    assert _queue(kit).queue == ()
    await kit.close()


async def test_a_hook_s_injected_instruction_gives_no_turn() -> None:
    """A hook directs an agent with ``send_event``, never by injecting an
    instruction, which is dropped before any pass (RFC §9.4)."""
    a, b = _provider("a answers"), _provider("b answers")
    kit = RoomKit()
    seen = _follow(kit)
    await _room(kit, {"a": a, "b": b})

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def direct(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if event.source.channel_id != "chat":
            return HookResult.allow()
        if event.type == EventType.INSTRUCTION:
            return HookResult.allow()
        instruction = RoomEvent(
            room_id=event.room_id,
            type=EventType.INSTRUCTION,
            source=event.source,
            content=TextContent(body="Summarise the incident."),
            addressed_to=["a"],
        )
        return HookResult.allow(
            injected=[InjectedEvent(event=instruction, target_channel_ids=["a"])]
        )

    await _say(kit, "@b go")
    await _settle(kit)

    assert (len(a.calls), len(b.calls)) == (0, 1)
    assert [who for change, who in seen if change == SpeakQueueChange.QUEUED] == [("b",)]
    await kit.close()


async def test_an_instruction_to_an_agent_the_room_lacks_reports_it_unavailable() -> None:
    """An agent the kit registers and the room does not bind is asked nowhere,
    and the instruction goes on (RFC §10.1.1 step 1)."""
    elsewhere = _provider()
    kit = RoomKit()
    await _room(kit, {"a": _provider()})
    kit.register_channel(AIChannel("elsewhere", provider=elsewhere))

    result = await _say(kit, "x", addressed_to=["elsewhere"], event_type=EventType.INSTRUCTION)

    assert (result.blocked, result.unavailable_targets) == (False, ["elsewhere"])
    assert elsewhere.calls == []
    await kit.close()


async def test_the_last_turn_given_ends_the_discussion_at_once() -> None:
    release = asyncio.Event()

    class _Held(MockAIProvider):
        async def generate(self, context: Any) -> AIResponse:
            await release.wait()
            return await super().generate(context)

    kit = RoomKit()
    await _room(kit, {"a": _Held(ai_responses=[AIResponse(content="a")])}, max_turns=1)
    await _say(kit, "@a go")
    async with asyncio.timeout(5):
        while _queue(kit).speaking != "a":
            await asyncio.sleep(0.01)

    # Over once max_turns turns were given: the running turn ends as it would.
    assert _queue(kit).over
    refused = await _say(kit, "more", addressed_to=["a"], event_type=EventType.INSTRUCTION)
    assert (refused.blocked, refused.reason) == (True, "discussion_over")
    release.set()
    await _settle(kit)
    await kit.close()


async def test_a_message_deleted_while_queued_gives_no_turn() -> None:
    """RFC §10.3: a message deleted while a turn waits to answer it is no
    longer said: the turn is not given, and no agent reads its text."""
    release = asyncio.Event()

    class _Held(MockAIProvider):
        async def generate(self, context: Any) -> AIResponse:
            await release.wait()
            return await super().generate(context)

    b = _provider("b here")
    kit = RoomKit()
    await _room(kit, {"a": _Held(ai_responses=[AIResponse(content="a")]), "b": b})
    await _say(kit, "@a go")
    async with asyncio.timeout(5):
        while _queue(kit).speaking != "a":
            await asyncio.sleep(0.01)
    queued = (await _say(kit, "@b the code is 4417")).event
    assert _queue(kit).queue == ("b",)

    await kit.delete_event("r1", queued.id)
    release.set()
    await _settle(kit)

    assert b.calls == []
    await kit.close()


# -- People --


async def test_a_delivery_is_not_a_persons_answer() -> None:
    a = _provider("@ops which region?", "thanks")
    kit = RoomKit()
    await _room(kit, {"a": a}, people=["ops"])
    await _say(kit, "@a deploy failed", metadata=OPS)
    await _settle(kit)
    assert _queue(kit).waiting

    await kit.deliver("r1", "background job finished", channel_id="chat")
    await _settle(kit)

    assert len(a.calls) == 1
    assert _queue(kit).asked == (("a", "ops"),) and _queue(kit).waiting
    await kit.close()


async def test_a_name_a_transport_stamped_is_one_agents_address() -> None:
    a, b = _provider("@Alice which region?", "rolling back"), _provider("b here")
    kit = RoomKit()
    await _room(kit, {"a": a, "b": b})

    await _say(kit, "@a deploy failed", sender="u1", metadata={"sender_name": "Alice"})
    await _settle(kit)
    assert _queue(kit).asked == (("a", "Alice"),)

    await _say(kit, "eu-west", sender="u1", metadata={"sender_name": "Alice"})
    await _settle(kit)
    assert (len(a.calls), len(b.calls)) == (2, 0)
    await kit.close()


async def test_a_sender_taking_anothers_name_does_not_answer_for_them() -> None:
    a, b = _provider("@Alice do you approve the rollback?", "noted"), _provider("b here")
    kit = RoomKit()
    await _room(kit, {"a": a, "b": b}, addressed_only=True)
    await _person(kit, "alice", "Alice", ParticipantRole.MEMBER)
    await _person(kit, "mallory", "Mallory", ParticipantRole.MEMBER)
    await _say(kit, "@a can we roll back?", sender="alice")
    await _settle(kit)
    assert _queue(kit).asked == (("a", "Alice"),)

    # Mallory stamps Alice's name: the transcript labels her "Alice (2)".
    await _say(kit, "yes, approved", sender="mallory", metadata={"sender_name": "Alice"})
    await _settle(kit)

    # Not Alice's answer: what was asked of her stands, nobody is woken.
    assert len(a.calls) == 1
    assert _queue(kit).asked == (("a", "Alice"),)
    await kit.close()


async def test_a_name_two_people_answer_to_records_no_ask() -> None:
    """``Alice Martin`` and ``AliceMartin`` are two people the register tells
    apart, addressed by one name: neither may answer for the other."""
    a = _provider("@AliceMartin do you approve?", "noted")
    kit = RoomKit()
    await _room(kit, {"a": a}, addressed_only=True)
    await _person(kit, "alice", "Alice Martin", ParticipantRole.MEMBER)
    await _person(kit, "mallory", "AliceMartin", ParticipantRole.MEMBER)

    await _say(kit, "@a can we roll back?", sender="alice")
    await _settle(kit)

    assert _queue(kit).asked == ()
    await kit.close()


async def test_a_name_too_long_to_address_is_not_cut_into_another() -> None:
    long_name = "Bartholomew-Fitzgerald-Winchester-III"
    a = _provider(f"@{long_name} do you approve?", "noted")
    kit = RoomKit()
    await _room(kit, {"a": a}, addressed_only=True)
    await _person(kit, "bart", long_name, ParticipantRole.MEMBER)
    await _person(kit, "mallory", long_name[:32], ParticipantRole.MEMBER)

    await _say(kit, "@a can we roll back?", sender="bart")
    await _settle(kit)
    await _say(kit, "yes", sender="mallory")
    await _settle(kit)

    assert _queue(kit).asked == () and len(a.calls) == 1
    await kit.close()


async def test_a_person_with_an_accented_name_can_be_addressed() -> None:
    a = _provider("@Hélène which region?")
    kit = RoomKit()
    await _room(kit, {"a": a})
    await _person(kit, "h", "Hélène", ParticipantRole.MEMBER)

    await _say(kit, "@a deploy failed", sender="h")
    await _settle(kit)

    assert _queue(kit).asked == (("a", "Hélène"),)
    await kit.close()


async def test_a_turn_names_the_message_it_answers_and_who_asked() -> None:
    a = _provider("a here")
    kit = RoomKit()
    await _room(kit, {"a": a, "b": _provider()})
    await _person(kit, "alice", "Alice", ParticipantRole.MEMBER)

    await _say(kit, "@a what is the error rate?", sender="alice")
    await _settle(kit)

    notes = next(str(m.content) for m in a.calls[0].messages if "[Discussion:" in str(m.content))
    assert "This turn answers the message from “Alice”, “@a what is the error rate?”" in notes
    assert "asked by “Alice”" in notes
    await kit.close()


# -- Waiting and listening --


async def test_a_room_with_nobody_to_wait_for_is_idle_after_a_depth_stop() -> None:
    a, b = _provider("@b go"), _provider("@a go")
    kit = RoomKit()
    await _room(kit, {"a": a, "b": b}, max_depth=3)
    await _person(kit, "bot1", "Pager", ParticipantRole.BOT)

    await _say(kit, "@a start", sender="bot1")
    await _settle(kit)
    assert not _queue(kit).waiting

    await _say(kit, "@b new alert", sender="bot1")
    await _settle(kit)
    assert len(b.calls) == 2
    await kit.close()


async def test_a_human_agent_is_someone_the_discussion_waits_for() -> None:
    """An AGENT is a human advisor (RFC §5), a person of the discussion
    (§19.7.5 rule 2): after a depth stop, the room waits for them."""
    a, b = _provider("@b go"), _provider("@a go")
    kit = RoomKit()
    await _room(kit, {"a": a, "b": b}, max_depth=3)
    await _person(kit, "adv1", "Claire", ParticipantRole.AGENT)

    await _say(kit, "@a start", sender="adv1")
    await _settle(kit)
    assert _queue(kit).waiting
    await kit.close()


async def test_a_persons_message_ends_the_wait_even_asking_no_one() -> None:
    a = _provider("@ops may I?", "@b please look")
    b = _provider("b looked")
    kit = RoomKit()
    await _room(kit, {"a": a, "b": b}, people=["ops"], addressed_only=True)
    await _say(kit, "@a go", metadata=OPS)
    await _settle(kit)
    assert _queue(kit).waiting
    # An instruction is given while waiting; what it asks of b waits.
    await _say(kit, "ask b", addressed_to=["a"], event_type=EventType.INSTRUCTION)
    await _settle(kit)
    assert _queue(kit).queue == ("b",) and len(b.calls) == 0

    await _say(kit, "just a note for the team", metadata=OPS)
    await _settle(kit)
    assert len(b.calls) == 1
    await kit.close()


async def test_an_agent_that_only_listens_ignores_a_message_that_does_not_name_it() -> None:
    a, b = _provider("a here"), _provider("b here")
    kit = RoomKit()
    await _room(kit, {"a": a, "b": b})
    await kit.listen_only("r1", ["b"])

    await _say(kit, "what happened?")
    await _settle(kit)

    assert (len(a.calls), len(b.calls)) == (1, 0)
    await kit.close()


async def test_listen_only_on_a_turn_given_but_not_started_cuts_it() -> None:
    a = _provider("a should not be heard")
    kit = RoomKit()
    held = asyncio.Event()

    @kit.hook(HookTrigger.ON_SPEAK_QUEUE, execution=HookExecution.ASYNC)
    async def hold(event: SpeakQueueEvent, context: RoomContext) -> None:
        if event.change == SpeakQueueChange.TURN_GIVEN:
            await kit.listen_only("r1", ["a"])
            held.set()

    await _room(kit, {"a": a})
    await _say(kit, "@a go")
    await asyncio.wait_for(held.wait(), 5)
    await _settle(kit)

    assert [e for e in await _timeline(kit) if e.source.channel_id == "a"] == []
    await kit.close()


async def test_talking_again_ends_a_wait_that_was_for_it() -> None:
    a, b = _provider("@b look"), _provider("b looked")
    kit = RoomKit()
    await _room(kit, {"a": a, "b": b})
    await kit.listen_only("r1", ["b"])
    await _say(kit, "@a go")
    await _settle(kit)
    assert _queue(kit).waiting

    await kit.talk_again("r1", ["b"])
    await _settle(kit)

    assert len(b.calls) == 1
    await kit.close()


# -- Uninstall, regeneration --


async def test_uninstalling_forgets_the_discussion_so_a_new_one_starts_fresh() -> None:
    kit = RoomKit()
    first = Discussion([AIChannel("a", provider=_provider("a"))], max_turns=1)
    await kit.create_room(room_id="r1", orchestration=first)
    kit.register_channel(_People("chat"))
    await kit.attach_channel("r1", "chat")
    await kit.listen_only("r1", ["a"])
    await _say(kit, "@a go")
    await _settle(kit)
    assert _queue(kit).over

    await kit.uninstall_strategy("r1")
    second = Discussion([kit.channels["a"]], max_turns=5)  # type: ignore[list-item]
    await kit.install_strategy("r1", second)

    queue = _queue(kit)
    assert (queue.over, queue.listening, queue.queue) == (False, frozenset(), ())
    await kit.close()


async def test_installing_a_discussion_over_one_keeps_its_queue() -> None:
    kit = RoomKit()
    await _room(kit, {"a": _provider("a"), "b": _provider("b"), "c": _provider("c")})
    await kit.listen_only("r1", ["b"])
    await _say(kit, "@a go")
    await _settle(kit)
    assert kit._discussions["r1"].state.turns_given == 1

    replaced = Discussion([kit.channels["a"], kit.channels["b"]])  # type: ignore[list-item]
    await kit.install_strategy("r1", replaced)

    # Who listens and the turns given stay; c, which the install attached, leaves.
    assert _queue(kit).listening == frozenset({"b"})
    assert kit._discussions["r1"].state.turns_given == 1
    assert kit.room_strategy("r1") is replaced
    assert {b.channel_id for b in await kit.store.list_bindings("r1")} == {"chat", "a", "b"}
    # The record covers both installs: uninstalling gives the room back whole.
    assert await kit.uninstall_strategy("r1") is True
    assert {b.channel_id for b in await kit.store.list_bindings("r1")} == {"chat"}
    await kit.close()


async def test_removing_an_agent_the_host_bound_itself_is_refused() -> None:
    kit = RoomKit()
    kit.register_channel(_People("chat"))
    a, c = AIChannel("a", provider=_provider("a")), AIChannel("c", provider=_provider("c"))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "chat")
    kit.register_channel(c)
    await kit.attach_channel("r1", "c", category=ChannelCategory.INTELLIGENCE)
    first = Discussion([a, c])
    await kit.install_strategy("r1", first)

    with pytest.raises(ValueError, match="c is an intelligence channel that is not one"):
        await kit.install_strategy("r1", Discussion([a]))

    # The room keeps the discussion it held, its agents and its record.
    assert kit.room_strategy("r1") is first
    assert kit._discussions["r1"].config.agents == ["a", "c"]
    assert {b.channel_id for b in await kit.store.list_bindings("r1")} == {"chat", "a", "c"}
    await kit.close()


async def test_two_installs_over_a_discussion_at_once_are_one_too_many() -> None:
    kit = RoomKit()
    await _room(kit, {"a": _provider("a"), "b": _provider("b")})
    load_room_context = kit.store.load_room_context

    async def slow(room_id: str) -> Any:
        await asyncio.sleep(0.01)  # the first install is still running
        return await load_room_context(room_id)

    kit.store.load_room_context = slow  # type: ignore[method-assign]
    first = Discussion([kit.channels["a"]])  # type: ignore[list-item]
    second = Discussion([kit.channels["a"], kit.channels["b"]])  # type: ignore[list-item]

    results = await asyncio.gather(
        kit.install_strategy("r1", first),
        kit.install_strategy("r1", second),
        return_exceptions=True,
    )

    assert [isinstance(r, ValueError) for r in results] == [False, True]
    # The record names the discussion the room runs.
    assert kit.room_strategy("r1") is first
    assert kit._discussions["r1"].strategy is first
    await kit.close()


async def test_a_replacement_that_fails_once_the_discussion_runs_it_is_recorded() -> None:
    kit = RoomKit()
    await _room(kit, {"a": _provider("a")})
    replaced = Discussion([kit.channels["a"], AIChannel("b", provider=_provider("b"))])  # type: ignore[list-item]
    attach_channel = kit.attach_channel

    async def failing(room_id: str, channel_id: str, **options: Any) -> Any:
        if channel_id == "b":
            raise RuntimeError("store down")
        return await attach_channel(room_id, channel_id, **options)

    kit.attach_channel = failing  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="store down"):
        await kit.install_strategy("r1", replaced)

    # The record and the discussion agree: installing again finishes it.
    assert kit.room_strategy("r1") is replaced
    assert kit._discussions["r1"].strategy is replaced
    kit.attach_channel = attach_channel  # type: ignore[method-assign]
    await kit.install_strategy("r1", replaced)
    assert "b" in {b.channel_id for b in await kit.store.list_bindings("r1")}
    await kit.close()


async def test_a_second_direct_install_takes_the_place_of_the_first() -> None:
    kit = RoomKit()
    await kit.create_room(room_id="r1")
    kit.register_channel(_People("chat"))
    await kit.attach_channel("r1", "chat")
    a = AIChannel("a", provider=_provider("a"))
    await Discussion([a]).install(kit, "r1")
    await kit.listen_only("r1", ["a"])
    second = Discussion([a], max_turns=3)

    await second.install(kit, "r1")

    assert kit._discussions["r1"].strategy is second
    assert _queue(kit).listening == frozenset({"a"})
    await kit.close()


async def test_uninstalling_from_a_speak_queue_hook_releases_the_room() -> None:
    kit = RoomKit()
    strategy = Discussion([AIChannel("a", provider=_provider("a"))])

    @kit.hook(HookTrigger.ON_SPEAK_QUEUE, execution=HookExecution.ASYNC)
    async def leave(event: SpeakQueueEvent, context: RoomContext) -> None:
        if event.change == SpeakQueueChange.TURN_ENDED:
            await strategy.uninstall(kit, "r1")

    await kit.create_room(room_id="r1", orchestration=strategy)
    kit.register_channel(_People("chat"))
    await kit.attach_channel("r1", "chat")
    await _say(kit, "@a go")
    async with asyncio.timeout(5):
        while kit.speak_queue("r1") is not None:
            await asyncio.sleep(0.01)
    await kit.close()


async def test_uninstalling_from_done_releases_the_room() -> None:
    kit = RoomKit()

    async def done(room_id: str) -> bool:
        await strategy.uninstall(kit, room_id)
        return True

    strategy = Discussion([AIChannel("a", provider=_provider("a"))], done=done)
    await kit.create_room(room_id="r1", orchestration=strategy)
    kit.register_channel(_People("chat"))
    await kit.attach_channel("r1", "chat")
    await _say(kit, "@a go")
    async with asyncio.timeout(5):
        while kit.speak_queue("r1") is not None:
            await asyncio.sleep(0.01)
    result = await _say(kit, "hello")
    assert [e.content.body for e in result.response_events] == ["a"]
    await kit.close()


async def test_regenerating_after_removing_the_answer_regenerates() -> None:
    a = _provider("first", "second")
    kit = RoomKit()
    await _room(kit, {"a": a})
    await _say(kit, "@a hello")
    await _settle(kit)
    trigger = await kit.regenerate_target("r1")
    assert trigger is not None
    answer = next(e for e in await _timeline(kit) if e.source.channel_id == "a")
    await kit.store.delete_event("r1", answer.id)

    result = await kit.regenerate_response("r1", trigger_id=trigger.id)
    assert result is not None and not result.blocked
    await _settle(kit)
    assert len(a.calls) == 2
    await kit.close()


async def test_regenerating_twice_before_the_turn_queues_it_once() -> None:
    gate = asyncio.Event()

    class _Slow(MockAIProvider):
        async def generate(self, context: Any) -> AIResponse:
            await gate.wait()
            return await super().generate(context)

    a = _Slow(ai_responses=[AIResponse(content="x")])
    kit = RoomKit()
    await _room(kit, {"a": a})
    await _say(kit, "@a hello")
    async with asyncio.timeout(5):
        while _queue(kit).speaking != "a":
            await asyncio.sleep(0.01)
    for _ in range(5):
        await kit.regenerate_response("r1")
    assert _queue(kit).queue == ("a",)
    gate.set()
    await _settle(kit)
    assert len(a.calls) == 2
    await kit.close()


# -- Failures --


async def test_a_store_failure_in_the_discussion_never_loses_the_message() -> None:
    kit = RoomKit()
    await _room(kit, {"a": _provider("a")})
    store = kit.store
    patch = store.patch_room_metadata
    failed: list[bool] = []

    async def flaky(room_id: str, updates: dict[str, Any]) -> Any:
        if "_speak_queue" in updates and not failed:
            failed.append(True)
            raise RuntimeError("store down")
        return await patch(room_id, updates)

    store.patch_room_metadata = flaky  # type: ignore[method-assign]
    result = await _say(kit, "@a hello")

    assert result.event is not None and failed == [True]
    stored = [e.content.body for e in await _timeline(kit)]
    assert "@a hello" in stored
    await kit.close()


async def test_a_failure_reading_the_queue_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_driver, "_RETRY_MAX", 0.01)  # the retry's wait, on a test's scale
    a = _provider("a")
    kit = RoomKit()
    await _room(kit, {"a": a})
    store = kit.store
    get_event = store.get_event
    failed: list[bool] = []

    async def flaky(event_id: str) -> Any:
        if not failed:
            failed.append(True)
            raise RuntimeError("store down")
        return await get_event(event_id)

    store.get_event = flaky  # type: ignore[method-assign]
    await _say(kit, "@a hello")
    async with asyncio.timeout(5):
        while len(a.calls) < 1:
            await asyncio.sleep(0.02)
    await kit.close()


async def test_concurrent_installs_on_one_room_are_one_too_many() -> None:
    kit = RoomKit()
    await kit.create_room(room_id="r1")
    load_room_context = kit.store.load_room_context

    async def slow(room_id: str) -> Any:
        await asyncio.sleep(0.01)  # the first install is still running
        return await load_room_context(room_id)

    kit.store.load_room_context = slow  # type: ignore[method-assign]
    first = Discussion([AIChannel("a", provider=_provider())])
    second = Discussion([kit.channels.get("a") or AIChannel("a", provider=_provider())])

    results = await asyncio.gather(
        first.install(kit, "r1"), second.install(kit, "r1"), return_exceptions=True
    )
    assert sum(isinstance(r, ValueError) for r in results) == 1
    await kit.close()


# -- Visibility and tenants --


async def test_a_name_a_binding_hides_the_message_from_queues_nobody() -> None:
    a, b = _provider("a here"), _provider("b here")
    kit = RoomKit()
    seen = _follow(kit)
    kit.register_channel(_People("chat"))
    channels = [AIChannel("a", provider=a), AIChannel("b", provider=b)]
    await kit.create_room(room_id="r1", orchestration=Discussion(channels))
    await kit.attach_channel("r1", "chat", visibility="a")

    await _say(kit, "@a @b hello")
    await _settle(kit)
    await _seen(seen, (SpeakQueueChange.TURN_ENDED, ("a",)))  # after what was queued

    assert (len(a.calls), len(b.calls)) == (1, 0)
    assert all("b" not in agents for change, agents in seen if change == SpeakQueueChange.QUEUED)
    await kit.close()


async def test_the_host_calls_are_scoped_to_a_tenant() -> None:
    kit = RoomKit()
    await kit.create_room(room_id="r1", organization_id="acme")
    await Discussion([AIChannel("a", provider=_provider())]).install(kit, "r1")

    assert kit.speak_queue("r1", organization_id="acme") is not None
    assert kit.speak_queue("r1", organization_id="other") is None
    with pytest.raises(ValueError, match="holds no discussion"):
        await kit.listen_only("r1", ["a"], organization_id="other")
    await kit.close()


async def test_names_are_read_after_every_host_hook() -> None:
    b = _provider("b here")
    kit = RoomKit()
    await _room(kit, {"a": _provider(), "b": b}, addressed_only=True)

    async def redact(event: RoomEvent, context: RoomContext) -> HookResult:
        if isinstance(event.content, TextContent) and "@b" in event.content.body:
            content = TextContent(body=event.content.body.replace("@b", "[redacted]"))
            return HookResult.modify(event.model_copy(update={"content": content}))
        return HookResult.allow()

    kit.hook_engine.add_room_hook(
        "r1",
        HookRegistration(
            trigger=HookTrigger.BEFORE_BROADCAST,
            execution=HookExecution.SYNC,
            fn=redact,
            priority=1_000_000,
            name="redact",
        ),
    )
    await _say(kit, "@b secret")
    await _settle(kit)

    assert len(b.calls) == 0
    await kit.close()


async def test_a_name_stamped_like_a_nameless_senders_channel_is_not_them() -> None:
    """A sender with no name is labelled by its channel (``@chat``), a form no
    name takes: a later sender stamping the name ``chat`` is someone else."""
    a = _provider("@chat may I roll back?", "noted")
    kit = RoomKit()
    await _room(kit, {"a": a}, people=["chat"], addressed_only=True)
    await _say(kit, "@a deploy failed", sender="u1")
    await _settle(kit)
    assert _queue(kit).asked == (("a", "@chat"),)

    await _say(kit, "yes", sender="u2", metadata={"sender_name": "chat"})
    await _settle(kit)

    assert _queue(kit).asked == (("a", "@chat"),) and len(a.calls) == 1
    await kit.close()


async def test_a_person_named_like_an_agent_is_still_a_person_in_the_notes() -> None:
    a = _provider("a here")
    kit = RoomKit()
    await _room(kit, {"a": a, "b": _provider()})
    await _person(kit, "pb", "b", ParticipantRole.MEMBER)

    await _say(kit, "@a hello", sender="pb")
    await _settle(kit)

    notes = next(str(m.content) for m in a.calls[0].messages if "[Discussion:" in str(m.content))
    assert "asked by “b”" in notes and "asked by @b" not in notes
    await kit.close()


async def test_an_agents_memory_learns_every_message_it_may_see() -> None:
    """A memory that learns as messages arrive (an index, a summary) learns the
    whole conversation, as in a room with no discussion: not its own rows, and
    the message a turn answers once."""
    learned: list[str] = []

    class _Recording(SlidingWindowMemory):
        async def ingest(  # type: ignore[override]
            self, room_id: str, event: RoomEvent, *, channel_id: str | None = None
        ) -> None:
            learned.append(event.content.body if isinstance(event.content, TextContent) else "")

    sre = AIChannel("sre", provider=_provider("sre: spike at 14:05"), memory=_Recording())
    dev = AIChannel("dev", provider=_provider("dev: release 4.2"))
    kit = RoomKit()
    kit.register_channel(_People("chat"))
    await kit.create_room(room_id="r1", orchestration=Discussion([sre, dev], addressed_only=True))
    await kit.attach_channel("r1", "chat")

    for body in ("@dev what was deployed?", "the customer is Acme", "@sre metrics?"):
        await _say(kit, body)
        await _settle(kit)

    assert learned == [
        "@dev what was deployed?",
        "dev: release 4.2",
        "the customer is Acme",
        "@sre metrics?",
    ]
    await kit.close()


async def test_an_agent_the_room_never_delivers_to_learns_nothing_and_is_not_queued() -> None:
    """The router's delivery set decides who an event reaches: an agent bound
    write-only is neither handed the message nor queued by its name. A
    binding's direction does not filter an intelligence channel (RFC §4.2)."""
    learned: list[str] = []

    class _Recording(SlidingWindowMemory):
        async def ingest(  # type: ignore[override]
            self, room_id: str, event: RoomEvent, *, channel_id: str | None = None
        ) -> None:
            learned.append(event.id)

    a = AIChannel("a", provider=_provider("a here"))
    b = AIChannel("b", provider=_provider("b here"), memory=_Recording())
    kit = RoomKit()
    kit.register_channel(_People("chat"))
    kit.register_channel(b)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "chat")
    await kit.attach_channel(
        "r1", "b", category=ChannelCategory.INTELLIGENCE, access=Access.WRITE_ONLY
    )
    await Discussion([a, b]).install(kit, "r1")

    await _say(kit, "@b are you there?")
    await _settle(kit)

    assert learned == [] and len(b._provider.calls) == 0  # type: ignore[attr-defined]
    await kit.close()


async def test_a_name_a_sender_stamped_before_the_participant_records_no_ask() -> None:
    """A sender stamps ``Alice`` before the room records its participant
    Alice: two sources behind one name. Neither may answer for the other, so
    the agent's ask records no one, rather than whoever spoke first."""
    a = _provider("noted", "@Alice can you confirm?")
    kit = RoomKit()
    await _room(kit, {"a": a}, addressed_only=True)
    await _say(kit, "@a hello", sender="m", metadata={"sender_name": "Alice"})
    await _settle(kit)
    await _person(kit, "alice", "Alice", ParticipantRole.MEMBER)

    await _say(kit, "@a please check the deploy", sender="alice")
    await _settle(kit)

    assert _queue(kit).asked == ()
    await kit.close()
