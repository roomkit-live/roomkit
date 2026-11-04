"""An agent of a discussion is addressed by its handle (RFC §19.7.5 rule 4).

A host whose channel ids cannot be names (``agent:<uuid>``) still has agents
that address each other: each agent has a handle, given by the host, else its
channel id when that can be a name, else its name. Notes, the transcript and
the dispatch policy present agents by their handles; the queue and
``addressed_to`` keep channel ids.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from roomkit import Agent, RoomKit
from roomkit.channels._discussion_turn import discussion_handles
from roomkit.channels.ai import AIChannel
from roomkit.core.locks import InMemoryLockManager
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.orchestration.strategies.discussion import (
    Discussion,
    DispatchCandidate,
    MockDispatchPolicy,
)
from roomkit.orchestration.strategies.discussion._config import CONFIG_KEY, DiscussionConfig
from roomkit.orchestration.strategies.discussion._room import DiscussionRoom
from roomkit.orchestration.strategies.discussion.dispatch.classifier import _member, question
from roomkit.providers.ai.base import AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.store.memory import InMemoryStore
from tests.orchestration.test_discussion_background import _starts_task, _until_said
from tests.orchestration.test_strategy_discussion import (
    NOTES,
    _provider,
    _room,
    _say,
    _settle,
    _texts,
    _timeline,
)
from tests.test_framework import SimpleChannel

JEV = "agent:3f2a1b2c-0d4e-4f5a-8b6c-7d8e9f0a1b2c"
COUNSEL = "agent:9b1c2d3e-4f5a-6b7c-8d9e-0f1a2b3c4d5e"


def _agents(jev: MockAIProvider, counsel: MockAIProvider) -> list[Agent]:
    return [
        Agent(JEV, provider=jev, name="Jev", description="reads the contract"),
        Agent(COUNSEL, provider=counsel, name="Counsel", description="checks the law"),
    ]


async def _keyed_room(kit: RoomKit, strategy: Discussion) -> None:
    kit.register_channel(SimpleChannel("ops"))
    await kit.create_room(room_id="r1", orchestration=strategy)
    await kit.attach_channel("r1", "ops")


async def _answers(kit: RoomKit) -> list[RoomEvent]:
    return [e for e in await _timeline(kit) if e.source.channel_id in (JEV, COUNSEL)]


# -- Which handle an agent has ---------------------------------------------------


def test_a_handle_is_the_hosts_else_the_channel_id_else_the_name() -> None:
    strategy = Discussion(
        [
            AIChannel("dev", provider=_provider()),
            Agent("agent:1", provider=_provider(), name="Research Desk"),
            AIChannel("agent:2", provider=_provider()),
            Agent("sre", provider=_provider(), name="Sam"),
        ],
        handles={"agent:2": "ops", "sre": "SiteReliability"},
    )
    assert strategy.handles() == {
        "dev": "dev",
        "agent:1": "ResearchDesk",
        "agent:2": "ops",
        "sre": "SiteReliability",
    }


def test_an_agent_with_a_name_keeps_its_channel_id_as_its_handle() -> None:
    """``name`` is a display name: a channel id that can name the agent stays its handle."""
    strategy = Discussion([Agent("dev", provider=_provider(), name="Developer")])
    assert strategy.handles() == {"dev": "dev"}


@pytest.mark.parametrize(
    ("agents", "handles", "refused"),
    [
        ({"agent:1": None}, None, "cannot be addressed by its channel id"),
        ({"agent:1": "!!"}, None, "cannot be addressed by its channel id"),
        ({"dev": None}, {"dev": "two words"}, "cannot name an agent"),
        ({"dev": None}, {"dev": "all"}, "cannot name an agent"),
        ({"dev": None}, {"dev": "x" * 33}, "cannot name an agent"),
        ({"dev": None}, {"sre": "sre"}, "not agents"),
        ({"agent:1": "Jev", "agent:2": "jev"}, None, "share the handle"),
        ({"dev": None, "sre": None}, {"sre": "DEV"}, "share the handle"),
    ],
)
def test_an_agent_without_a_handle_or_two_alike_are_refused_at_construction(
    agents: dict[str, str | None], handles: dict[str, str] | None, refused: str
) -> None:
    """*agents* maps each agent's channel id to its name."""
    channels = [Agent(agent_id, provider=_provider(), name=n) for agent_id, n in agents.items()]
    with pytest.raises(ValueError, match=refused):
        Discussion(channels, handles=handles)


def test_a_configuration_stored_without_handles_addresses_agents_by_channel_id() -> None:
    """A room stored without handles reads names by channel id."""
    config = DiscussionConfig.stored({CONFIG_KEY: {"agents": ["a", "b"], "max_depth": 5}})
    assert config is not None and config.handles is None
    room = DiscussionRoom.following(RoomKit(), "r1", config)
    assert room.handles == {"a": "a", "b": "b"}
    assert room.label("b") == "@b"


@pytest.mark.parametrize(
    "stored",
    [
        {"agents": [], "max_depth": 5},
        {"agents": ["a", "a"], "max_depth": 5},
        {"agents": ["a"], "max_depth": 0},
        {"agents": ["a"], "max_depth": 5, "max_turns": 0},
        {"agents": ["a"], "max_depth": 5, "silent_token": "  "},
        {"agents": ["a"], "max_depth": 5, "rules": "  "},
        {"agents": ["a"], "max_depth": 5, "everyone": ["ghost"]},
        {"agents": ["a"], "max_depth": 5, "handles": {"ghost": "ops"}},
        {"agents": ["a", "b"], "max_depth": 5, "handles": {"a": "ops", "b": "OPS"}},
        # An agent stored with no handle is addressed by its channel id.
        {"agents": ["a", "b"], "max_depth": 5, "handles": {"a": "b"}},
        {"agents": ["a"], "max_depth": 5, "handles": {"a": "two\nlines"}},
        {"agents": ["a"], "max_depth": 5, "handles": {"a": "all"}},
        {"agents": ["a"]},
        # A store holds what Discussion() wrote: JSON's own types, no lax reading.
        {"agents": ["a"], "max_depth": "5"},
        {"agents": ["a"], "max_depth": 5.0},
        {"agents": ["a"], "max_depth": True},
        {"agents": ["a"], "max_depth": 5, "addressed_only": "yes"},
    ],
)
async def test_what_the_constructor_refuses_stored_reads_as_no_discussion(
    stored: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """Something below the kit's API wrote it: no error, which would break
    every message of the room, and no discussion holding the room either."""
    agent = _provider("from a")
    kit = RoomKit()
    kit.register_channel(SimpleChannel("ops"))
    kit.register_channel(AIChannel("a", provider=agent))
    await kit.store.create_room(Room(id="r1", metadata={CONFIG_KEY: stored}))
    await kit.attach_channel("r1", "ops")
    await kit.attach_channel("r1", "a", category=ChannelCategory.INTELLIGENCE)

    with caplog.at_level(logging.WARNING, logger="roomkit.orchestration.discussion"):
        assert DiscussionConfig.stored({CONFIG_KEY: stored}) is None
    result = await _say(kit, "hello")

    assert "A stored discussion reads as none" in caplog.text
    assert [e.content.body for e in result.response_events] == ["from a"]
    assert kit.speak_queue("r1") is None
    await kit.close()


async def test_a_depth_of_1_the_kit_writes_for_a_discussion_is_read_back() -> None:
    """A discussion with no ``max_depth`` stores the kit's limit, which may be 1
    where the constructor refuses it for a discussion's own."""
    kit = RoomKit(max_chain_depth=1)
    await _room(kit, {"a": _provider("from a")})
    room = await kit.get_room("r1")

    config = DiscussionConfig.stored(room.metadata)

    assert config is not None and config.max_depth == 1
    await kit.close()


@pytest.mark.parametrize(
    ("config", "handles"),
    [
        ({"agents": ["b"], "handles": {"b": "ops"}, "max_depth": 5}, {"b": "ops"}),
        # A channel that is no agent is no agent's handle: a person stays one.
        ({"agents": ["b"], "handles": {"sms": "b"}, "max_depth": 5}, {}),
        ({"handles": {"b": "ops"}, "max_depth": 5}, {}),
        # What cannot name an agent: a line break would leave the transcript's label.
        ({"agents": ["b"], "handles": {"b": "two\nlines"}, "max_depth": 5}, {}),
        ({"agents": ["b"], "handles": {"b": 3}, "max_depth": 5}, {}),
        # What reads as no discussion labels nobody, a handle of an agent included.
        ({"agents": ["sms", "b"], "handles": {"sms": "ops"}}, {}),
        ({"agents": ["sms"], "handles": {"sms": "ops"}, "max_depth": 0}, {}),
    ],
)
def test_a_stored_handle_counts_for_an_agent_and_if_it_can_name_one(
    config: dict[str, Any], handles: dict[str, str]
) -> None:
    assert discussion_handles({CONFIG_KEY: config}) == handles


async def test_a_discussion_handed_to_the_kit_holds_no_room() -> None:
    """Only the strategy writes a discussion: one a room's creation names is
    dropped, the sender keeps its own label and the room answers."""
    agent = _provider("from b")
    kit = RoomKit()
    kit.register_channel(SimpleChannel("ops"))
    kit.register_channel(AIChannel("b", provider=agent))
    forged = {"agents": ["b"], "handles": {"ops": "b"}, "max_depth": 5}
    room = await kit.create_room(room_id="r1", metadata={CONFIG_KEY: forged})
    await kit.attach_channel("r1", "ops")
    await kit.attach_channel("r1", "b", category=ChannelCategory.INTELLIGENCE)

    result = await _say(kit, "hello")

    assert room.metadata == {}
    assert kit.speak_queue("r1") is None
    assert [e.content.body for e in result.response_events] == ["from b"]
    assert len(agent.calls) == 1
    await kit.close()


async def test_a_metadata_update_leaves_the_stored_discussion_as_it_is() -> None:
    """A read-modify-write writes its own keys: naming a discussion's keys
    with another value, it stops no turn and marks no process as replaced."""
    agent = _provider("from a")
    kit = RoomKit()
    await _room(kit, {"a": agent})
    room = await kit.get_room("r1")
    changed = {**room.metadata[CONFIG_KEY], "max_turns": 9}

    updated = await kit.update_room_metadata(
        "r1",
        {**room.metadata, CONFIG_KEY: changed, "_speak_queue": {"over": True}, "crm": "c_42"},
    )
    await _say(kit, "@a hello")
    await _settle(kit)

    assert updated.metadata["crm"] == "c_42"
    assert updated.metadata[CONFIG_KEY] == room.metadata[CONFIG_KEY]
    assert len(agent.calls) == 1
    await kit.close()


# -- What a handle does ------------------------------------------------------------


async def test_an_agent_naming_another_by_its_handle_queues_it() -> None:
    jev = _provider("Clause 4 caps the liability. @counsel is that cap lawful?")
    counsel = _provider("Lawful, within the statutory floor.")
    kit = RoomKit()
    await _keyed_room(kit, Discussion(_agents(jev, counsel), everyone=[JEV]))

    await _say(kit, "@Jev what does clause 4 say?")
    await _settle(kit)

    asked, answered = await _answers(kit)
    assert (asked.source.channel_id, answered.source.channel_id) == (JEV, COUNSEL)
    # The address keeps channel ids.
    assert asked.addressed_to == [COUNSEL]
    assert (len(jev.calls), len(counsel.calls)) == (1, 1)
    await kit.close()


async def test_notes_and_the_transcript_present_agents_by_their_handles() -> None:
    jev = _provider("Clause 4 caps the liability. @Counsel is that cap lawful?")
    counsel = _provider("Lawful.")
    kit = RoomKit()
    await _keyed_room(kit, Discussion(_agents(jev, counsel), everyone=[JEV]))

    await _say(kit, "@Jev what does clause 4 say?")
    await _settle(kit)

    texts = _texts(counsel)
    notes = next(t for t in texts if NOTES in t)
    assert "You are @Counsel," in notes
    assert "- @Jev: Jev" in notes
    assert "the message from @Jev" in notes and "asked by @Jev" in notes
    assert "as in @Jev" in notes
    # Jev's turn, in the conversation Counsel reads, carries Jev's handle.
    assert any('@Jev: "Clause 4 caps the liability.' in t for t in texts)
    # No channel id anywhere: neither "agent:3f2a…" nor the transcript's "agent-3f2a…".
    assert not any("3f2a" in t or "9b1c" in t for t in texts)
    await kit.close()


async def test_a_person_named_like_a_handle_is_addressed_as_the_agent() -> None:
    jev, counsel = _provider("Jev here"), _provider("Counsel here")
    kit = RoomKit()
    await _keyed_room(kit, Discussion(_agents(jev, counsel), people=["Jev", "ops"]))

    await _say(kit, "@jev please")
    await _settle(kit)

    assert [e.source.channel_id for e in await _answers(kit)] == [JEV]
    await kit.close()


async def test_the_dispatch_policy_reads_agents_by_their_handles() -> None:
    policy = MockDispatchPolicy([[COUNSEL], [JEV]])
    jev = _provider("Clause 4 caps it. @Counsel lawful?", "ok")
    counsel = _provider("Lawful.")
    kit = RoomKit()
    await _keyed_room(kit, Discussion(_agents(jev, counsel), dispatch=policy))

    await _say(kit, "@Jev clause 4?")
    await _settle(kit)
    await _say(kit, "thanks, and the next step?")
    await _settle(kit)

    (turn,) = policy.turns
    assert [(c.channel_id, c.handle) for c in turn.candidates] == [
        (JEV, "Jev"),
        (COUNSEL, "Counsel"),
    ]
    by_agent = {e.source.channel_id: e.id for e in turn.recent}
    assert turn.speakers[by_agent[JEV]] == "@Jev"
    assert turn.speakers[by_agent[COUNSEL]] == "@Counsel"
    await kit.close()


def test_the_classifier_is_asked_about_each_candidate_by_its_handle() -> None:
    candidate = DispatchCandidate(channel_id=COUNSEL, name="Counsel", handle="Counsel")
    assert question(candidate).instructions.startswith("Should @Counsel (Counsel) take")
    # A candidate with no handle is presented by its channel id.
    unnamed = question(DispatchCandidate(channel_id="sre"))
    assert unnamed.instructions.startswith("Should @sre take")


async def test_a_process_that_follows_the_discussion_reads_handles() -> None:
    store, locks = InMemoryStore(), InMemoryLockManager()
    jev, counsel = _provider("Jev here"), _provider("Counsel here")
    host = RoomKit(store=store, lock_manager=locks)
    await _keyed_room(host, Discussion(_agents(jev, counsel), everyone=[JEV]))
    follower = RoomKit(store=store, lock_manager=locks)
    follower.register_channel(SimpleChannel("ops"))

    result: Any = await follower.process_inbound(
        InboundMessage(
            channel_id="ops",
            sender_id="oncall",
            content=TextContent(body="@Counsel is clause 4 lawful?"),
        )
    )
    await _settle(host)

    stored = await store.get_event(result.event.id)
    assert stored is not None and stored.addressed_to == [COUNSEL]
    assert (len(jev.calls), len(counsel.calls)) == (0, 1)
    await follower.close()
    await host.close()


async def _until_speaking(kit: RoomKit, agent: str) -> None:
    async with asyncio.timeout(5):
        while (queue := kit.speak_queue("r1")) is None or queue.speaking != agent:
            await asyncio.sleep(0.01)


class _Holding(MockAIProvider):
    """A model whose turn runs until the process stops."""

    async def generate(self, context: Any) -> AIResponse:
        self.calls.append(context)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


async def test_a_stored_turn_of_an_agent_no_longer_held_does_not_stall_the_discussion() -> None:
    """The queue outlives the process (rule 16): a reinstalled discussion
    without that agent drops its turn and goes on."""
    store, locks = InMemoryStore(), InMemoryLockManager()
    first = RoomKit(store=store, lock_manager=locks)
    first.register_channel(SimpleChannel("ops"))
    team = [AIChannel("a", provider=_Holding()), AIChannel("b", provider=_provider("b"))]
    await first.create_room(room_id="r1", orchestration=Discussion(team))
    await first.attach_channel("r1", "ops")
    await _say(first, "@a @b status?")
    await _until_speaking(first, "a")
    await first.close()

    second = RoomKit(store=store, lock_manager=locks)
    second.register_channel(SimpleChannel("ops"))
    a = _provider("a answers")
    await second.detach_channel("r1", "b")
    await Discussion([AIChannel("a", provider=a), AIChannel("c", provider=_provider())]).install(
        second, "r1"
    )
    await _say(second, "@a are you there?")
    await _settle(second)

    assert len(a.calls) == 1
    queue = second.speak_queue("r1")
    assert queue is not None and "b" not in queue.queue
    await second.close()


async def test_a_background_task_names_its_agent_by_handle() -> None:
    """The worker's brief, the tool's answer and the relayed result name the
    agent as the room addresses it, never by an ``agent:<uuid>`` channel id."""
    comms_id = "agent:7c1e0b2a-3d4f-4a5b-8c6d-9e0f1a2b3c4d"
    comms = MockAIProvider(ai_responses=_starts_task("On it.", "Findings: 12.", "12 tickets."))
    kit = RoomKit()
    kit.register_channel(SimpleChannel("ops"))
    agents = [Agent(comms_id, provider=comms, name="Comms"), Agent("sre", provider=_provider())]
    kit.setup_background(agents[0])
    await kit.create_room(room_id="r1", orchestration=Discussion(agents))
    await kit.attach_channel("r1", "ops")

    await _say(kit, "@comms please go through the support tickets")
    await _until_said(kit, 2)

    told = ["\n".join(str(m.content) for m in call.messages) for call in comms.calls]
    assert any("in the background for @Comms" in t for t in told)
    assert any("its result comes back to Comms" in t for t in told)
    assert "[Background task from Comms completed." in told[-1]
    assert not any("7c1e" in t for t in told)
    await kit.close()


def test_the_classifier_team_lists_each_candidate_by_its_handle() -> None:
    candidate = DispatchCandidate(channel_id=COUNSEL, name="Counsel", handle="Counsel")
    member = _member(candidate, waits=False)
    assert member == {"agent": "@Counsel", "name": "Counsel"}


async def test_an_agent_naming_a_person_records_the_ask_under_its_channel_id() -> None:
    jev = _provider("@ops may I quote clause 4 in full?")
    kit = RoomKit()
    await _keyed_room(kit, Discussion(_agents(jev, _provider()), everyone=[JEV]))

    await _say(kit, "@Jev what does clause 4 say?")
    await _settle(kit)

    queue = kit.speak_queue("r1")
    assert queue is not None and queue.asked == ((JEV, "@ops"),) and queue.waiting
    await kit.close()
