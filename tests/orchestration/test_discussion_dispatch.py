"""A discussion's dispatch policy (RFC §19.7.5 rule 18).

For a person's message that names no agent and answers none, which rule 8
gives to ``everyone``, the policy decides which agents take it, in which
order, or that none does: decided once, by the process holding the lease,
off the room lock, bounded, falling back to every candidate.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import Agent, RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.classifiers.base import ClassifierError
from roomkit.classifiers.mock import MockClassifier
from roomkit.core.locks import InMemoryLockManager
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import EventType, HookExecution, HookTrigger
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.orchestration.strategies.discussion import (
    ClassifierDispatchPolicy,
    Discussion,
    DispatchCandidate,
    DispatchDecision,
    DispatchDecisionEvent,
    DispatchPolicy,
    DispatchTurn,
    MockDispatchPolicy,
    SpeakQueueChange,
    SpeakQueueEvent,
    _queue,
)
from roomkit.orchestration.strategies.discussion._config import CONFIG_KEY
from roomkit.orchestration.strategies.discussion._shared import STATE_KEY
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.store.memory import InMemoryStore
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

AGENTS = ("investigator", "dev", "sre")


def _team(**answers: str) -> dict[str, MockAIProvider]:
    return {name: _provider(answers.get(name, f"{name} here")) for name in AGENTS}


def _decisions(kit: RoomKit) -> list[DispatchDecisionEvent]:
    seen: list[DispatchDecisionEvent] = []

    @kit.hook(HookTrigger.ON_DISPATCH_DECISION, execution=HookExecution.ASYNC)
    async def record(event: DispatchDecisionEvent, ctx: Any) -> None:
        seen.append(event)

    return seen


async def _answers(kit: RoomKit) -> list[str]:
    return [
        e.source.channel_id
        for e in await _timeline(kit)
        if e.source.channel_id in AGENTS and e.type == EventType.MESSAGE
    ]


async def test_the_policy_picks_who_takes_an_unaddressed_message() -> None:
    team = _team()
    policy = MockDispatchPolicy([["sre"]])
    kit = RoomKit()
    await _room(kit, team, dispatch=policy)

    await _say(kit, "the error rate is climbing")
    await _settle(kit)

    assert await _answers(kit) == ["sre"]
    assert [len(p.calls) for p in team.values()] == [0, 0, 1]
    (turn,) = policy.turns
    assert turn.room_id == "r1"
    assert isinstance(turn.event.content, TextContent)
    assert turn.event.content.body == "the error rate is climbing"
    assert [c.channel_id for c in turn.candidates] == list(AGENTS)
    assert turn.speakers[turn.event.id] == "@ops"
    await kit.close()


async def test_the_agents_answer_in_the_order_decided() -> None:
    kit = RoomKit()
    await _room(kit, _team(), dispatch=MockDispatchPolicy([["sre", "investigator"]]))

    await _say(kit, "what is going on?")
    await _settle(kit)

    assert await _answers(kit) == ["sre", "investigator"]
    await kit.close()


async def test_an_empty_decision_asks_no_agent() -> None:
    team = _team()
    kit = RoomKit()
    await _room(kit, team, dispatch=MockDispatchPolicy([[]]))

    await _say(kit, "thanks, that helps")
    await _settle(kit)

    assert await _answers(kit) == []
    queue = kit.speak_queue("r1")
    assert queue is not None and queue.queue == () and not queue.waiting
    await kit.close()


async def test_a_name_and_an_answer_owed_are_never_decided() -> None:
    team = {
        "investigator": _provider("@ops which region?", "on it"),
        "dev": _provider("dev here"),
        "sre": _provider("sre here"),
    }
    policy = MockDispatchPolicy([[]])
    kit = RoomKit()
    await _room(kit, team, dispatch=policy, people=["ops"])

    await _say(kit, "@investigator checkout fails")
    await _settle(kit)
    await _say(kit, "eu-west")  # answers the investigator's question
    await _settle(kit)

    assert policy.turns == []
    assert await _answers(kit) == ["investigator", "investigator"]
    await kit.close()


async def test_a_failing_policy_asks_every_candidate() -> None:
    kit = RoomKit()
    seen = _decisions(kit)
    await _room(kit, _team(), dispatch=MockDispatchPolicy(error=RuntimeError("down")))

    await _say(kit, "anyone?")
    await _settle(kit)

    assert sorted(await _answers(kit)) == sorted(AGENTS)
    (report,) = seen
    assert report.decision.reason == "fallback"
    assert report.decision.agents == AGENTS
    await kit.close()


class _Broken(DispatchPolicy):
    async def decide(self, turn: DispatchTurn) -> DispatchDecision:
        return None  # type: ignore[return-value]


async def test_a_policy_that_returns_no_decision_asks_every_candidate() -> None:
    kit = RoomKit()
    seen = _decisions(kit)
    await _room(kit, _team(), dispatch=_Broken())

    await _say(kit, "anyone?")
    await _settle(kit)

    assert sorted(await _answers(kit)) == sorted(AGENTS)
    assert seen[0].decision.reason == "fallback"
    await kit.close()


async def test_a_decision_it_cannot_read_asks_every_candidate() -> None:
    kit = RoomKit()
    seen = _decisions(kit)
    unreadable = DispatchDecision(agents=None)  # type: ignore[arg-type]
    await _room(kit, _team(), dispatch=MockDispatchPolicy([unreadable]))

    await _say(kit, "anyone?")
    await _settle(kit)

    assert sorted(await _answers(kit)) == sorted(AGENTS)
    assert seen[0].decision.reason == "fallback"
    await kit.close()


async def test_a_policy_past_its_bound_asks_every_candidate() -> None:
    kit = RoomKit()
    seen = _decisions(kit)
    policy = MockDispatchPolicy([["sre"]], delay=1.0)
    await _room(kit, _team(), dispatch=policy, dispatch_timeout=0.05)

    await _say(kit, "anyone?")
    await _settle(kit)

    assert sorted(await _answers(kit)) == sorted(AGENTS)
    (report,) = seen
    assert report.decision.reason == "fallback" and report.duration_ms == 50
    await kit.close()


async def test_a_decision_keeps_to_the_candidates_each_once() -> None:
    team = _team()
    kit = RoomKit()
    seen = _decisions(kit)
    policy = MockDispatchPolicy([["ghost", "sre", "dev", "sre"]])
    await _room(kit, team, dispatch=policy)
    await kit.listen_only("r1", ["dev"])

    await _say(kit, "status?")
    await _settle(kit)

    # The agent that only listens is no candidate: no policy picks it.
    assert [c.channel_id for c in policy.turns[0].candidates] == ["investigator", "sre"]
    assert await _answers(kit) == ["sre"]
    assert seen[0].decision.agents == ("sre",)
    assert seen[0].candidates == ("investigator", "sre")
    await kit.close()


async def test_everyone_orders_and_bounds_the_candidates() -> None:
    policy = MockDispatchPolicy()
    kit = RoomKit()
    await _room(kit, _team(), dispatch=policy, everyone=["sre", "investigator"])

    await _say(kit, "who is on it?")
    await _settle(kit)

    assert [c.channel_id for c in policy.turns[0].candidates] == ["sre", "investigator"]
    assert await _answers(kit) == ["sre", "investigator"]
    await kit.close()


async def test_a_decision_reports_the_message_candidates_and_cost() -> None:
    kit = RoomKit()
    seen = _decisions(kit)
    decision = DispatchDecision(("dev",), "deploys are dev's", {"dev": 0.9})
    await _room(kit, _team(), dispatch=MockDispatchPolicy([decision]))

    result = await _say(kit, "was anything deployed?")
    await _settle(kit)

    (report,) = seen
    assert report.room_id == "r1" and report.event.id == result.event.id
    assert report.candidates == AGENTS
    assert report.decision == decision
    assert report.duration_ms >= 0
    await kit.close()


async def test_the_queue_says_who_a_decision_queued() -> None:
    kit = RoomKit()
    changes: list[SpeakQueueEvent] = []

    @kit.hook(HookTrigger.ON_SPEAK_QUEUE, execution=HookExecution.ASYNC)
    async def follow(event: SpeakQueueEvent, ctx: Any) -> None:
        changes.append(event)

    await _room(kit, _team(), dispatch=MockDispatchPolicy([["dev"]]))
    result = await _say(kit, "deploy?")
    await _settle(kit)
    await asyncio.sleep(0.05)

    queued = [c for c in changes if c.change == SpeakQueueChange.QUEUED]
    assert [(c.channel_ids, c.event_id) for c in queued] == [(("dev",), result.event.id)]
    await kit.close()


async def test_a_message_waiting_for_a_decision_keeps_its_place() -> None:
    kit = RoomKit()
    await _room(kit, _team(), dispatch=MockDispatchPolicy([["sre"]], delay=0.2))

    await _say(kit, "the error rate is climbing")
    await asyncio.sleep(0.05)  # the decision runs
    await _say(kit, "@dev and the deploy?")
    async with asyncio.timeout(5):
        while len(await _answers(kit)) < 2:
            await asyncio.sleep(0.02)

    # The first message's pick comes before the agent a later one named.
    assert await _answers(kit) == ["sre", "dev"]
    await kit.close()


async def test_a_pick_merged_with_a_later_message_answers_the_later_one() -> None:
    team = _team()
    kit = RoomKit()
    await _room(kit, team, dispatch=MockDispatchPolicy([["sre"]], delay=0.2))

    await _say(kit, "the error rate is climbing")
    await asyncio.sleep(0.05)  # the decision runs
    await _say(kit, "@sre check the deploy")
    async with asyncio.timeout(5):
        while not team["sre"].calls:
            await asyncio.sleep(0.02)
    await _settle(kit)

    # One turn for both, answering the latest person's message (rule 8).
    assert len(team["sre"].calls) == 1
    notes = next(text for text in _texts(team["sre"]) if NOTES in text)
    assert "check the deploy" in notes and "error rate" not in notes
    await kit.close()


async def test_a_decision_another_process_took_first_is_neither_applied_nor_reported() -> None:
    team = _team()
    kit = RoomKit()
    seen = _decisions(kit)
    await _room(kit, team, dispatch=MockDispatchPolicy([["sre"]], delay=0.2))

    await _say(kit, "anyone?")
    await asyncio.sleep(0.05)  # the decision runs
    room = kit._discussions["r1"]
    async with room.editing() as state:  # as a process that took the lease over would
        state.dispatching = []
    await asyncio.sleep(0.3)
    await _settle(kit)

    assert seen == [] and not team["sre"].calls
    await kit.close()


async def test_closing_while_deciding_leaves_the_message_waiting() -> None:
    store = InMemoryStore()
    kit = RoomKit(store=store)
    await _room(kit, _team(), dispatch=MockDispatchPolicy([["sre"]], delay=1.0))

    result = await _say(kit, "anyone?")
    await asyncio.sleep(0.05)  # the decision runs
    await kit.close()

    room = await store.get_room("r1")
    assert room is not None
    waiting = room.metadata[STATE_KEY]["dispatching"]
    assert [p["event_id"] for p in waiting] == [result.event.id]


async def test_past_the_messages_that_may_wait_the_oldest_asks_its_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_queue, "KEPT_PENDING", 2)
    store, locks = InMemoryStore(), InMemoryLockManager()
    installer = RoomKit(store=store, lock_manager=locks)
    await _room(installer, _team(), dispatch=MockDispatchPolicy([[]]))
    await installer.close()  # no process left to decide: messages wait
    follower = RoomKit(store=store, lock_manager=locks)
    follower.register_channel(SimpleChannel("ops"))

    first = await _say(follower, "one")
    await _say(follower, "two")
    await _say(follower, "three")

    room = await store.get_room("r1")
    assert room is not None
    waiting = room.metadata[STATE_KEY]["dispatching"]
    assert first.event.id not in [p["event_id"] for p in waiting] and len(waiting) == 2
    queue = follower.speak_queue("r1")
    assert queue is not None and queue.queue == AGENTS
    await follower.close()


async def test_the_recent_conversation_holds_what_a_candidate_may_read() -> None:
    policy = MockDispatchPolicy([[]])
    kit = RoomKit()
    await _room(kit, _team(), dispatch=policy, everyone=["sre"])
    kit.register_channel(SimpleChannel("side"))
    await kit.attach_channel("r1", "side", visibility="dev")

    await _say(kit, "@investigator first look")
    await _settle(kit)
    await kit.process_inbound(
        InboundMessage(
            channel_id="side",
            sender_id="someone",
            content=TextContent(body="only dev may read this"),
        )
    )
    await _settle(kit)
    await _say(kit, "so?")
    await _settle(kit)

    turn = policy.turns[-1]
    said = [e.content.body for e in turn.recent if isinstance(e.content, TextContent)]
    assert "@investigator first look" in said and "investigator here" in said
    assert "only dev may read this" not in said
    assert "so?" not in said
    by_agent = next(e for e in turn.recent if e.source.channel_id == "investigator")
    assert turn.speakers[by_agent.id] == "@investigator"
    await kit.close()


async def test_a_done_discussion_decides_nothing() -> None:
    policy = MockDispatchPolicy([["sre"]])
    over = {"now": False}
    kit = RoomKit()
    await _room(kit, _team(), dispatch=policy, done=lambda room_id: over["now"])

    over["now"] = True
    await _say(kit, "anyone?")
    await _settle(kit)

    assert policy.turns == [] and await _answers(kit) == []
    queue = kit.speak_queue("r1")
    assert queue is not None and queue.over
    await kit.close()


async def test_a_dispatched_message_stays_unaddressed() -> None:
    kit = RoomKit()
    await _room(kit, _team(), dispatch=MockDispatchPolicy([["dev"]]))

    result = await _say(kit, "deploy?")
    await _settle(kit)

    stored = await kit.store.get_event(result.event.id)
    assert stored is not None and stored.addressed_to is None
    room = await kit.store.get_room("r1")
    assert room is not None and room.metadata[CONFIG_KEY]["dispatch"] is True
    await kit.close()


async def test_identity_reaches_the_candidates() -> None:
    policy = MockDispatchPolicy([[]])
    kit = RoomKit()
    kit.register_channel(SimpleChannel("ops"))
    agents = [
        Agent("sre", provider=_provider(), name="Sam", description="runs the platform"),
        AIChannel("dev", provider=_provider()),
    ]
    await kit.create_room(room_id="r1", orchestration=Discussion(agents, dispatch=policy))
    await kit.attach_channel("r1", "ops")

    await _say(kit, "hello")
    await _settle(kit)

    sre, dev = policy.turns[0].candidates
    assert (sre.name, sre.description) == ("Sam", "runs the platform")
    assert (dev.name, dev.role, dev.description) == (None, None, None)
    await kit.close()


def test_a_dispatch_policy_does_not_go_with_addressed_only() -> None:
    agents = [AIChannel("a", provider=_provider())]
    with pytest.raises(ValueError, match="addressed_only"):
        Discussion(agents, addressed_only=True, dispatch=MockDispatchPolicy())
    with pytest.raises(ValueError, match="dispatch_timeout"):
        Discussion(agents, dispatch=MockDispatchPolicy(), dispatch_timeout=0)


# -- Across processes --


async def test_the_lease_holder_decides_what_a_follower_routed() -> None:
    store, locks = InMemoryStore(), InMemoryLockManager()
    policy = MockDispatchPolicy([["dev"]])
    holder = RoomKit(store=store, lock_manager=locks)
    holder.register_channel(SimpleChannel("ops"))
    team = _team()
    channels = [AIChannel(n, provider=p) for n, p in team.items()]
    await holder.create_room(room_id="r1", orchestration=Discussion(channels, dispatch=policy))
    await holder.attach_channel("r1", "ops")
    follower = RoomKit(store=store, lock_manager=locks)
    follower.register_channel(SimpleChannel("ops"))

    await _say(follower, "deploy?")
    await _settle(holder)

    assert len(policy.turns) == 1
    assert [len(p.calls) for p in team.values()] == [0, 1, 0]
    await follower.close()
    await holder.close()


# -- The classifier policy --


def _turn(*candidates: str, text: str = "the error rate is climbing") -> DispatchTurn:
    def event(channel: str, body: str) -> RoomEvent:
        return RoomEvent(
            room_id="r1",
            source=EventSource(channel_id=channel, channel_type="websocket"),
            content=TextContent(body=body),
        )

    earlier = event("investigator", "logs show 502s")
    message = event("ops", text)
    return DispatchTurn(
        room_id="r1",
        event=message,
        recent=(earlier,),
        speakers={earlier.id: "@investigator", message.id: "Alice"},
        candidates=tuple(
            DispatchCandidate(c, name=c.title(), description=f"{c} work") for c in candidates
        ),
    )


async def test_the_classifier_picks_the_likeliest_above_the_threshold() -> None:
    classifier = MockClassifier({"agent_0": 0.2, "agent_1": 0.7, "agent_2": 0.9})
    policy = ClassifierDispatchPolicy(classifier)

    decision = await policy.decide(_turn("investigator", "dev", "sre"))

    assert decision.agents == ("sre", "dev")
    assert decision.judgments == {"investigator": 0.2, "dev": 0.7, "sre": 0.9}
    assert decision.reason == "above threshold"


async def test_the_classifier_keeps_to_max_agents_and_may_pick_none() -> None:
    classifier = MockClassifier(
        [{"agent_0": 0.8, "agent_1": 0.9, "agent_2": 0.6}, {"agent_0": 0.1}]
    )
    policy = ClassifierDispatchPolicy(classifier, max_agents=1)

    assert (await policy.decide(_turn("a", "b", "c"))).agents == ("b",)
    nobody = await policy.decide(_turn("a", "b", "c", text="thanks!"))
    assert nobody.agents == () and nobody.reason == "nobody above threshold"


async def test_the_classifier_reads_the_team_the_conversation_and_the_message() -> None:
    classifier = MockClassifier()
    await ClassifierDispatchPolicy(classifier).decide(_turn("sre"))

    ((state, questions),) = classifier.calls
    assert state == {
        "team": [{"agent": "@sre", "name": "Sre", "can": "sre work"}],
        "conversation": [{"speaker": "@investigator", "text": "logs show 502s"}],
        "message": {"speaker": "Alice", "text": "the error rate is climbing"},
    }
    (question,) = questions.values()
    assert question.instructions.startswith("Should @sre (Sre: sre work) take the latest message")


async def test_a_classifier_failure_is_the_discussions_fallback() -> None:
    kit = RoomKit()
    seen = _decisions(kit)
    classifier = MockClassifier(error=ClassifierError("refused"))
    await _room(kit, _team(), dispatch=ClassifierDispatchPolicy(classifier))

    await _say(kit, "anyone?")
    await _settle(kit)

    assert sorted(await _answers(kit)) == sorted(AGENTS)
    assert seen[0].decision.reason == "fallback"
    await kit.close()


def test_the_classifier_policy_checks_its_bounds() -> None:
    with pytest.raises(ValueError, match="threshold"):
        ClassifierDispatchPolicy(MockClassifier(), threshold=0)
    with pytest.raises(ValueError, match="max_agents"):
        ClassifierDispatchPolicy(MockClassifier(), max_agents=0)
    with pytest.raises(ValueError, match="recent"):
        ClassifierDispatchPolicy(MockClassifier(), recent=-1)


def test_a_policy_is_a_dispatch_policy() -> None:
    assert isinstance(MockDispatchPolicy(), DispatchPolicy)
    assert isinstance(ClassifierDispatchPolicy(MockClassifier()), DispatchPolicy)
