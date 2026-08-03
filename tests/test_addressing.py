"""Tests for event addressing (RFC §19.3).

Addressing says who is *asked to act*; visibility says who may *see*. The
two are independent, and most of these tests exist to keep them that way.
"""

from __future__ import annotations

from typing import Any

from roomkit import AgentResponsePolicy, ChannelCategory, RoomKit
from roomkit.core.event_router import _solicits
from roomkit.models.channel import ChannelOutput
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType, EventType, Visibility
from roomkit.models.event import EventSource, RoomEvent, TextContent
from tests.test_framework import SimpleChannel


class _Talker(SimpleChannel):
    """An agent that always answers — the ping-pong engine of RFC B.4."""

    category = ChannelCategory.INTELLIGENCE
    channel_type = ChannelType.AI

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.solicited: list[str] = []

    async def on_event(self, event: RoomEvent, binding: Any, context: Any) -> Any:
        if event.source.channel_id == self.channel_id:
            return ChannelOutput.empty()
        self.solicited.append(self.extract_text(event) or event.id)
        return ChannelOutput(
            responded=True,
            response_events=[
                RoomEvent(
                    room_id=event.room_id,
                    type=EventType.MESSAGE,
                    source=EventSource(channel_id=self.channel_id, channel_type=ChannelType.AI),
                    content=TextContent(body=f"{self.channel_id} replies"),
                )
            ],
        )


class _RecordingAgent(SimpleChannel):
    """An intelligence channel that records what it was asked to act on."""

    category = ChannelCategory.INTELLIGENCE
    channel_type = ChannelType.AI

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.solicited: list[str] = []

    async def on_event(self, event: RoomEvent, binding: Any, context: Any) -> Any:
        if event.source.channel_id != self.channel_id:
            self.solicited.append(self.extract_text(event) or event.id)
        return await super().on_event(event, binding, context)


async def _room(*agent_ids: str) -> tuple[RoomKit, SimpleChannel, dict[str, _RecordingAgent]]:
    kit = RoomKit()
    human = SimpleChannel("human")
    kit.register_channel(human)
    agents = {aid: _RecordingAgent(aid) for aid in agent_ids}
    for agent in agents.values():
        kit.register_channel(agent)
    await kit.create_room(room_id="room-1")
    await kit.attach_channel("room-1", "human")
    for aid in agent_ids:
        await kit.attach_channel("room-1", aid, category=ChannelCategory.INTELLIGENCE)
    return kit, human, agents


def _event(**kwargs: Any) -> RoomEvent:
    return RoomEvent(
        room_id="room-1",
        type=EventType.MESSAGE,
        source=EventSource(channel_id="human", channel_type=ChannelType.SMS),
        content=TextContent(body="hi"),
        **kwargs,
    )


class TestSolicitation:
    def test_unaddressed_solicits_everyone(self) -> None:
        assert _solicits(_event(), "any-agent") is True

    def test_address_names_who_acts(self) -> None:
        event = _event(addressed_to=["codex"])
        assert _solicits(event, "codex") is True
        assert _solicits(event, "claude-code") is False

    def test_empty_address_solicits_nobody(self) -> None:
        # Distinct from None: "asked no one" is a decision, not an absence.
        assert _solicits(_event(addressed_to=[]), "codex") is False

    def test_address_outranks_the_router(self) -> None:
        # RFC §19.4 step 0 — a router cannot override what the sender asked.
        event = _event(
            addressed_to=["codex"],
            metadata={"_routed_to": "claude-code", "_always_process": ["supervisor"]},
        )
        assert _solicits(event, "codex") is True
        assert _solicits(event, "claude-code") is False
        assert _solicits(event, "supervisor") is False

    def test_router_still_decides_when_unaddressed(self) -> None:
        event = _event(metadata={"_routed_to": "claude-code", "_always_process": ["supervisor"]})
        assert _solicits(event, "claude-code") is True
        assert _solicits(event, "supervisor") is True
        assert _solicits(event, "codex") is False


class TestInboundAddressing:
    async def test_only_the_addressed_agent_acts(self) -> None:
        kit, human, agents = await _room("claude-code", "codex")

        await kit.process_inbound(
            InboundMessage(
                channel_id="human",
                sender_id="user",
                content=TextContent(body="review it"),
                addressed_to=["codex"],
            )
        )

        assert agents["codex"].solicited == ["review it"]
        assert agents["claude-code"].solicited == []
        await kit.close()

    async def test_unaddressed_reaches_every_agent(self) -> None:
        kit, human, agents = await _room("claude-code", "codex")

        await kit.process_inbound(
            InboundMessage(channel_id="human", sender_id="user", content=TextContent(body="hello"))
        )

        assert agents["codex"].solicited == ["hello"]
        assert agents["claude-code"].solicited == ["hello"]
        await kit.close()

    async def test_address_is_stored_on_the_event(self) -> None:
        # A transcript can show who was asked; a replay reproduces it.
        kit, human, agents = await _room("codex")

        await kit.process_inbound(
            InboundMessage(
                channel_id="human",
                sender_id="user",
                content=TextContent(body="go"),
                addressed_to=["codex"],
            )
        )

        events = await kit.store.list_events("room-1")
        inbound = next(e for e in events if e.source.channel_id == "human")
        assert inbound.addressed_to == ["codex"]
        await kit.close()

    async def test_addressing_does_not_narrow_visibility(self) -> None:
        # The other agent is not asked to act, but a transport channel still
        # receives the message — addressing is not a delivery filter.
        kit, human, agents = await _room("claude-code", "codex")
        witness = SimpleChannel("witness")
        kit.register_channel(witness)
        await kit.attach_channel("room-1", "witness")

        await kit.process_inbound(
            InboundMessage(
                channel_id="human",
                sender_id="user",
                content=TextContent(body="only codex please"),
                addressed_to=["codex"],
            )
        )

        assert [e.content.body for e in witness.delivered] == ["only codex please"]
        await kit.close()

    async def test_unknown_ids_solicit_nobody(self) -> None:
        kit, human, agents = await _room("codex")

        await kit.process_inbound(
            InboundMessage(
                channel_id="human",
                sender_id="user",
                content=TextContent(body="ghost"),
                addressed_to=["not-in-this-room"],
            )
        )

        assert agents["codex"].solicited == []
        await kit.close()

    async def test_visibility_and_addressing_compose(self) -> None:
        # An event hidden from a channel stays hidden even if addressed to it.
        kit, human, agents = await _room("codex")

        await kit.process_inbound(
            InboundMessage(
                channel_id="human",
                sender_id="user",
                content=TextContent(body="hidden"),
                addressed_to=["codex"],
                visibility=Visibility.TRANSPORT,
            )
        )

        assert agents["codex"].solicited == []
        await kit.close()


class TestAgentResponsePolicy:
    """RFC §19.3.1 — what an agent's own output solicits."""

    async def test_agent_chain_is_the_default(self) -> None:
        kit, human, agents = await _room("a", "b")
        room = await kit.get_room("room-1")
        assert room is not None
        assert room.agent_response_policy is AgentResponsePolicy.AGENT_CHAIN
        await kit.close()

    def test_chain_lets_an_agent_solicit_the_others(self) -> None:
        event = _event()
        assert (
            _solicits(
                event,
                "b",
                source_is_agent=True,
                policy=AgentResponsePolicy.AGENT_CHAIN,
            )
            is True
        )

    def test_addressed_only_silences_unaddressed_agent_output(self) -> None:
        # The two lines every multi-agent app used to hand-write.
        event = _event()
        assert (
            _solicits(
                event,
                "b",
                source_is_agent=True,
                policy=AgentResponsePolicy.ADDRESSED_ONLY,
            )
            is False
        )

    def test_addressed_only_still_honours_an_explicit_address(self) -> None:
        # An agent MAY address another agent and be answered by it alone.
        event = _event(addressed_to=["b"])
        for channel, expected in (("b", True), ("c", False)):
            assert (
                _solicits(
                    event,
                    channel,
                    source_is_agent=True,
                    policy=AgentResponsePolicy.ADDRESSED_ONLY,
                )
                is expected
            )

    def test_addressed_only_leaves_human_messages_alone(self) -> None:
        # The policy governs agent-sourced events, nothing else.
        event = _event()
        assert (
            _solicits(
                event,
                "b",
                source_is_agent=False,
                policy=AgentResponsePolicy.ADDRESSED_ONLY,
            )
            is True
        )

    async def test_policy_is_per_room_over_the_kit_default(self) -> None:
        kit = RoomKit(agent_response_policy=AgentResponsePolicy.ADDRESSED_ONLY)
        inherited = await kit.create_room(room_id="inherited")
        overridden = await kit.create_room(
            room_id="overridden",
            agent_response_policy=AgentResponsePolicy.AGENT_CHAIN,
        )

        assert inherited.agent_response_policy is AgentResponsePolicy.ADDRESSED_ONLY
        assert overridden.agent_response_policy is AgentResponsePolicy.AGENT_CHAIN
        await kit.close()

    async def test_addressed_only_stops_the_ping_pong_end_to_end(self) -> None:
        kit = RoomKit(agent_response_policy=AgentResponsePolicy.ADDRESSED_ONLY)
        human = SimpleChannel("human")
        kit.register_channel(human)
        talkers = {aid: _Talker(aid) for aid in ("a", "b")}
        for talker in talkers.values():
            kit.register_channel(talker)
        await kit.create_room(room_id="room-1")
        await kit.attach_channel("room-1", "human")
        for aid in talkers:
            await kit.attach_channel("room-1", aid, category=ChannelCategory.INTELLIGENCE)

        await kit.process_inbound(
            InboundMessage(
                channel_id="human",
                sender_id="user",
                content=TextContent(body="go"),
                addressed_to=["a"],
            )
        )

        # "a" answered once; "b" never heard a thing, so nothing bounced back.
        assert talkers["a"].solicited == ["go"]
        assert talkers["b"].solicited == []
        await kit.close()
