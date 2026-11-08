"""A discussion served by several processes (RFC §19.7.5 rule 16).

Each "process" is a kit of its own, with its own agent objects; they share the
store and the lock manager, as processes share Postgres and its advisory
locks. One queue in the store, one process giving the turns under a lease.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

import pytest

from roomkit import Discussion, Loop, Pipeline, RoomKit, Supervisor, Swarm
from roomkit.channels._mark_copies import compile_mark_patterns
from roomkit.channels.ai import AIChannel
from roomkit.core.exceptions import RoomNotFoundError
from roomkit.core.locks import InMemoryLockManager
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelCategory,
    ChannelType,
    EventType,
    HookExecution,
    HookTrigger,
)
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.orchestration.strategies.discussion import (
    SpeakQueueChange,
    SpeakQueueEvent,
    _driver,
    _shared,
)
from roomkit.orchestration.strategies.discussion._config import DiscussionConfig
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.store.memory import InMemoryStore
from roomkit.tools.context import _current_loop_ctx
from tests.test_framework import SimpleChannel

AGENTS = ("investigator", "sre", "comms")
KEPT = ("investigator", "sre")


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Poll and lease on a test's time scale."""
    monkeypatch.setattr(_driver, "POLL_SECONDS", 0.05)
    monkeypatch.setattr(_shared, "LEASE_SECONDS", 0.6)
    monkeypatch.setattr(_shared, "_RENEW_BEFORE", 0.4)


class _Slow(MockAIProvider):
    """Answers after a while, and logs when it generates, across processes:
    ``+name`` as it starts, ``-name`` as it answers, ``xname`` once its turn
    is cut. A ``Cancel`` of its turn, or the cluster's *release*, ends the
    wait at once, as a streaming provider's next chunk would reach the
    steering check; the delay only bounds a turn nothing ends."""

    def __init__(
        self,
        name: str,
        log: list[str],
        delay: float = 0.02,
        release: asyncio.Event | None = None,
    ) -> None:
        super().__init__(ai_responses=[AIResponse(content=f"{name} answers")])
        self.who, self.log, self.delay = name, log, delay
        self.release = release or asyncio.Event()

    async def generate(self, context: Any) -> AIResponse:
        self.log.append(f"+{self.who}")
        loop = _current_loop_ctx.get()
        ends = [self.release, *([loop.cancel_event] if loop is not None else [])]
        waits = [asyncio.ensure_future(end.wait()) for end in ends]
        try:
            await asyncio.wait(waits, timeout=self.delay, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            self.log.append(f"x{self.who}")
            raise
        finally:
            for wait in waits:
                wait.cancel()
        cut = loop is not None and loop.cancel_event.is_set()
        self.log.append(f"{'x' if cut else '-'}{self.who}")
        return await super().generate(context)


class _Cluster:
    """Processes serving one room: one store, one lock manager."""

    def __init__(self) -> None:
        self.store = InMemoryStore()
        self.locks = InMemoryLockManager()
        self.log: list[str] = []
        self.kits: list[RoomKit] = []
        self.release = asyncio.Event()

    def strategy(
        self, delay: float = 0.02, names: Sequence[str] = AGENTS, **options: Any
    ) -> Discussion:
        agents = [AIChannel(n, provider=_Slow(n, self.log, delay, self.release)) for n in names]
        return Discussion(agents, **options)

    def kit(self) -> RoomKit:
        kit = RoomKit(store=self.store, lock_manager=self.locks)
        kit.register_channel(SimpleChannel("ops"))
        self.kits.append(kit)
        return kit

    async def process(self, *, install: bool, create: bool = False, **options: Any) -> RoomKit:
        kit = self.kit()
        if create:
            await kit.create_room(room_id="r1", orchestration=self.strategy(**options))
            await kit.attach_channel("r1", "ops")
        elif install:
            await kit.install_strategy("r1", self.strategy(**options))
        return kit

    async def hosted(self, bound: str) -> RoomKit:
        """A process that creates the room and binds *bound* itself before
        installing the discussion: the install attaches the other agents."""
        kit = self.kit()
        strategy = self.strategy()
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ops")
        kit.register_channel(next(a for a in strategy.agents() if a.channel_id == bound))
        await kit.attach_channel("r1", bound, category=ChannelCategory.INTELLIGENCE)
        await kit.install_strategy("r1", strategy)
        return kit

    async def stored(self) -> DiscussionConfig | None:
        room = await self.store.get_room("r1")
        return DiscussionConfig.stored(room.metadata if room is not None else None)

    async def turns_given(self) -> int:
        room = await self.store.get_room("r1")
        assert room is not None
        return int(room.metadata["_speak_queue"]["turns_given"])

    async def bound(self) -> set[str]:
        return {b.channel_id for b in await self.store.list_bindings("r1")}

    async def answers(self) -> list[str]:
        events = await self.store.list_events("r1", limit=100)
        return [
            e.source.channel_id
            for e in events
            if e.type == EventType.MESSAGE and e.source.channel_id in AGENTS
        ]

    def overlap(self) -> int:
        running = most = 0
        for line in self.log:
            running += 1 if line.startswith("+") else -1
            most = max(most, running)
        return most

    async def close(self) -> None:
        for kit in self.kits:
            await kit.close()


async def _say(kit: RoomKit, body: str, **extra: Any) -> Any:
    return await kit.process_inbound(
        InboundMessage(channel_id="ops", sender_id="ops", content=TextContent(body=body), **extra)
    )


async def _until(predicate: Any, timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while not await predicate():
            await asyncio.sleep(0.02)


async def _idle(kit: RoomKit) -> None:
    """Until *kit*, the process giving the turns, runs none and has none to
    give (an agent left queued only listens): a turn the test says nobody
    gets would have been given by then."""

    async def idle() -> bool:
        queue = kit.speak_queue("r1")
        return queue is not None and queue.speaking is None and set(queue.queue) <= queue.listening

    await _until(idle)


async def _turn_over(cluster: _Cluster, driver: Any, agent: str = "investigator") -> None:
    """Until the turn *driver* was giving has run to its end, cut or not: its
    generation returned or was cancelled (a stopped driver forgets its turn at
    once, the generation may still run), and an answer it was going to write is
    written by then."""

    async def generated() -> bool:
        return f"-{agent}" in cluster.log or f"x{agent}" in cluster.log

    async def over() -> bool:
        return driver.speaking is None

    await _until(generated)
    await asyncio.sleep(0.05)  # the write that follows a generation that returned
    await _until(over)


def _shorter_lease(monkeypatch: pytest.MonkeyPatch) -> float:
    """A lease that expires in a quarter of a second, for a test that waits it out."""
    monkeypatch.setattr(_shared, "LEASE_SECONDS", 0.25)
    monkeypatch.setattr(_shared, "_RENEW_BEFORE", 0.15)
    return 0.25


async def test_a_process_that_did_not_install_it_follows_the_discussion() -> None:
    cluster = _Cluster()
    a = await cluster.process(install=True, create=True, delay=0.1)
    follower = await cluster.process(install=False)

    result = await _say(follower, "checkout is failing, what happened?")

    # The follower asks nobody at broadcast: no dogpile.
    assert result.response_events == []

    async def three() -> bool:
        return len(await cluster.answers()) >= 3

    await _until(three)
    await _idle(a)
    assert sorted(await cluster.answers()) == sorted(AGENTS)
    assert cluster.overlap() == 1
    queue = follower.speak_queue("r1")
    assert queue is not None  # the follower knows the room holds a discussion
    await cluster.close()


async def test_two_processes_never_give_two_turns_at_once() -> None:
    cluster = _Cluster()
    # Turns long enough to overlap, were two processes to give one each.
    a = await cluster.process(install=True, create=True, delay=0.1)
    b = await cluster.process(install=True, delay=0.1)

    await _say(a, "@investigator what do the logs say?")
    await asyncio.sleep(0.02)
    await _say(b, "@sre and the metrics?")

    async def both() -> bool:
        return sorted(await cluster.answers()) == ["investigator", "sre"]

    await _until(both)
    assert cluster.overlap() == 1
    await cluster.close()


@pytest.mark.wallclock
async def test_a_turn_longer_than_the_lease_keeps_it() -> None:
    # The process giving a turn renews its lease while the turn runs, so no
    # other process gives a turn meanwhile (rule 16).
    cluster = _Cluster()
    # Past the lease of 0.6 s.
    a = await cluster.process(install=True, create=True, delay=0.8)
    b = await cluster.process(install=True, delay=0.8)
    await _say(a, "@investigator what do the logs say?")
    await _turn_started(cluster)

    await _say(b, "@sre and the metrics?")

    async def both() -> bool:
        return sorted(await cluster.answers()) == ["investigator", "sre"]

    await _until(both)
    assert cluster.overlap() == 1
    await cluster.close()


async def test_a_process_that_stopped_hands_the_room_over_once_its_lease_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _shorter_lease(monkeypatch)
    cluster = _Cluster()
    a = await cluster.process(install=True, create=True)
    b = await cluster.process(install=True)
    await _say(a, "@investigator hello")

    async def one() -> bool:
        return len(await cluster.answers()) == 1

    await _until(one)
    # A crashes: its driver dies without giving the lease back.
    held = a._discussions["r1"]
    assert held.driver is not None and held.driver._task is not None
    held.driver._task.cancel()
    held.closed = True

    await _say(b, "@sre are you there?")

    async def two() -> bool:
        return len(await cluster.answers()) == 2

    await _until(two)
    assert (await cluster.answers())[-1] == "sre"
    await cluster.close()


async def test_uninstalling_in_one_process_ends_the_discussion_in_the_others() -> None:
    cluster = _Cluster()
    strategy = cluster.strategy()
    a = RoomKit(store=cluster.store, lock_manager=cluster.locks)
    a.register_channel(SimpleChannel("ops"))
    cluster.kits.append(a)
    await a.create_room(room_id="r1", orchestration=strategy)
    await a.attach_channel("r1", "ops")
    follower = await cluster.process(install=False)
    await _say(follower, "@investigator hello")

    async def one() -> bool:
        return len(await cluster.answers()) == 1

    await _until(one)
    await strategy.uninstall(a, "r1")
    for agent in AGENTS:
        follower.register_channel(AIChannel(agent, provider=_Slow(agent, cluster.log, 0.0)))

    result = await _say(follower, "@sre plain room again?")

    # The follower left: no name is read any more and the room answers by its
    # policy again, the agents asked at broadcast.
    assert follower.speak_queue("r1") is None
    assert {e.source.channel_id for e in result.response_events} <= set(AGENTS)
    assert result.response_events
    await cluster.close()


async def test_an_instruction_moves_the_turns_to_the_process_holding_it() -> None:
    cluster = _Cluster()
    a = await cluster.process(install=True, create=True)
    b = await cluster.process(install=True)
    await _say(a, "@investigator hello")

    async def one() -> bool:
        return len(await cluster.answers()) == 1

    await _until(one)

    await _say(b, "summarise", addressed_to=["comms"], event_type=EventType.INSTRUCTION)

    async def two() -> bool:
        return len(await cluster.answers()) == 2

    await _until(two)
    assert (await cluster.answers())[-1] == "comms"
    queue = b.speak_queue("r1")
    state = b._discussions["r1"].state
    assert queue is not None and state.lease_holder == b._instance_id
    await cluster.close()


async def test_listen_only_in_one_process_cuts_the_turn_running_in_another() -> None:
    cluster = _Cluster()
    a = await cluster.process(install=True, create=True, delay=2.0)
    follower = await cluster.process(install=False)
    await _say(follower, "@investigator take your time")

    async def started() -> bool:
        return "+investigator" in cluster.log

    await _until(started)
    await follower.listen_only("r1", ["investigator"])

    async def ended() -> bool:
        queue = a.speak_queue("r1")
        return queue is not None and queue.speaking is None

    await _until(ended, timeout=3.0)
    assert await cluster.answers() == []
    await cluster.close()


async def test_an_agent_that_only_listens_answers_a_person_naming_it_in_another_process() -> None:
    # Rule 12: the named agent takes its one turn; listening already when the
    # turn was given, the process giving it does not cut it.
    cluster = _Cluster()
    a = await cluster.process(install=True, create=True, delay=0.3)
    follower = await cluster.process(install=False)
    await a.listen_only("r1", ["sre"])

    await _say(follower, "@sre what do the metrics say?")

    async def answered() -> bool:
        return await cluster.answers() == ["sre"]

    await _until(answered)
    queue = a.speak_queue("r1")
    assert queue is not None and queue.listening == frozenset({"sre"})
    await cluster.close()


async def test_a_restarted_process_takes_the_discussion_back() -> None:
    cluster = _Cluster()
    a = await cluster.process(install=True, create=True)
    await a.close()  # the process stops: the queue and its configuration stay stored
    b = await cluster.process(install=False)
    await _say(b, "@investigator are you there?")
    await asyncio.sleep(3 * _driver.POLL_SECONDS)  # a driver would have read it by now
    assert await cluster.answers() == []  # followed: no process gives the turns

    await b.install_strategy("r1", cluster.strategy())

    async def answered() -> bool:
        return await cluster.answers() == ["investigator"]

    await _until(answered)
    assert isinstance(b.room_strategy("r1"), Discussion)
    assert b.speak_queue("r1") is not None
    await cluster.close()


async def test_a_process_that_only_follows_uninstalls_the_discussion_for_all() -> None:
    cluster = _Cluster()
    a = await cluster.hosted(bound="investigator")
    stored = await cluster.stored()
    assert stored is not None and stored.attached == ["comms", "sre"]
    await a.close()
    b = cluster.kit()

    assert await b.uninstall_strategy("r1") is True

    # Gone for every process, the agents its install attached with it; the
    # agent the host bound itself stays.
    assert await cluster.stored() is None
    assert await cluster.bound() == {"ops", "investigator"}
    assert b.speak_queue("r1") is None and b.room_strategy("r1") is None
    assert await b.uninstall_strategy("r1") is False
    b.register_channel(AIChannel("investigator", provider=_Slow("investigator", cluster.log, 0.0)))
    result = await _say(b, "anyone?")
    # The room answers by its policy again.
    assert [e.source.channel_id for e in result.response_events] == ["investigator"]
    await cluster.close()


async def _two_processes_then_a_change(cluster: _Cluster) -> tuple[RoomKit, RoomKit]:
    """A and B serve the room, sre only listens and one turn was given; then
    A installs the discussion without comms."""
    a = await cluster.process(install=True, create=True)
    b = await cluster.process(install=True)
    await b.listen_only("r1", ["sre"])
    await _say(a, "@investigator hello")

    async def one() -> bool:
        return len(await cluster.answers()) == 1

    await _until(one)
    await a.install_strategy("r1", cluster.strategy(names=KEPT))

    async def b_follows() -> bool:
        return b._discussions["r1"].config.agents == list(KEPT)

    await _until(b_follows)
    return a, b


async def test_changing_the_agents_keeps_the_queue_and_lets_the_removed_agent_go() -> None:
    cluster = _Cluster()
    a, b = await _two_processes_then_a_change(cluster)

    stored = await cluster.stored()
    assert stored is not None and stored.agents == list(KEPT)
    assert await cluster.turns_given() == 1
    assert "comms" not in await cluster.bound()
    view_a, view_b = a.speak_queue("r1"), b.speak_queue("r1")
    assert view_a is not None and view_b is not None
    assert view_a.listening == view_b.listening == frozenset({"sre"})
    assert a._discussions["r1"].drives
    await cluster.close()


async def test_a_process_whose_host_installed_the_old_agents_gives_no_turn() -> None:
    cluster = _Cluster()
    a, b = await _two_processes_then_a_change(cluster)
    assert not b._discussions["r1"].drives

    # B routes by the new configuration: @comms names no agent any more, and
    # the message asks everyone the discussion holds that does not only listen.
    await _say(b, "@comms are you there?")

    async def two() -> bool:
        return len(await cluster.answers()) == 2

    await _until(two)
    await _idle(a)
    assert await cluster.answers() == ["investigator", "investigator"]
    assert "+comms" not in cluster.log
    await cluster.close()


async def test_installing_the_stored_agents_lets_a_process_give_turns_again() -> None:
    cluster = _Cluster()
    _, b = await _two_processes_then_a_change(cluster)
    held = b._discussions["r1"]

    await b.install_strategy("r1", cluster.strategy(names=KEPT))

    assert held.drives and b._discussions["r1"] is held
    await cluster.close()


async def test_an_instruction_or_a_result_reaching_a_process_that_gives_no_turn_is_not_kept() -> (
    None
):
    cluster = _Cluster()
    _, b = await _two_processes_then_a_change(cluster)
    seen: list[tuple[SpeakQueueChange, tuple[str, ...]]] = []

    @b.hook(HookTrigger.ON_SPEAK_QUEUE, execution=HookExecution.ASYNC)
    async def follow(event: SpeakQueueEvent, context: Any) -> None:
        seen.append((event.change, event.channel_ids))

    await _say(b, "summarise", addressed_to=["investigator"], event_type=EventType.INSTRUCTION)
    result = RoomEvent(
        room_id="r1",
        source=EventSource(channel_id="investigator", channel_type=ChannelType.AI),
        content=TextContent(body="Result of the task"),
    )

    # B cannot give the turn that would take them: it reports the
    # instruction dropped, and leaves the result for its caller to post.
    assert await b._queue_discussion_post("r1", result) is False

    async def reported() -> bool:
        return (SpeakQueueChange.INSTRUCTION_DROPPED, ("investigator",)) in seen

    await _until(reported)
    assert all(change != SpeakQueueChange.QUEUED for change, _ in seen)
    assert b._discussions["r1"].instructions == {}
    await cluster.close()


async def test_what_a_process_held_when_another_install_replaced_its_own_is_not_kept() -> None:
    # A takes a task's result and an instruction for sre behind a slow turn;
    # B's install then replaces A's (sre kept), so A gives no turn: it posts
    # the result and reports the instruction dropped (rules 16 and 19).
    cluster = _Cluster()
    a = await cluster.process(install=True, create=True, delay=1.0)
    b = await cluster.process(install=True, delay=1.0)
    seen: list[tuple[SpeakQueueChange, tuple[str, ...]]] = []

    @a.hook(HookTrigger.ON_SPEAK_QUEUE, execution=HookExecution.ASYNC)
    async def follow(event: SpeakQueueEvent, context: Any) -> None:
        seen.append((event.change, event.channel_ids))

    await _say(a, "@investigator take your time")
    await _turn_started(cluster)
    result = RoomEvent(
        room_id="r1",
        source=EventSource(channel_id="sre", channel_type=ChannelType.AI),
        content=TextContent(body="Result of the task"),
        addressed_to=[],
    )
    assert await a._queue_discussion_post("r1", result) is True
    await _say(a, "summarise", addressed_to=["sre"], event_type=EventType.INSTRUCTION)
    await b.install_strategy("r1", cluster.strategy(names=KEPT, delay=1.0))
    cluster.release.set()  # the slow turn ends behind the install

    async def posted() -> list[RoomEvent]:
        events = await cluster.store.list_events("r1", limit=200)
        return [e for e in events if e.content == result.content]

    async def handed_back() -> bool:
        dropped = (SpeakQueueChange.INSTRUCTION_DROPPED, ("sre",)) in seen
        return dropped and bool(await posted())

    await _until(handed_back)
    await asyncio.sleep(3 * _driver.POLL_SECONDS)  # a second post or sre's turn would be in
    assert len(await posted()) == 1
    assert a._discussions["r1"].instructions == {}
    assert "+sre" not in cluster.log  # the instruction's turn was given nowhere
    await cluster.close()


async def test_a_quiet_follower_routes_host_calls_by_a_new_configuration() -> None:
    cluster = _Cluster()
    a = await cluster.process(install=True, create=True, names=KEPT)
    follower = await cluster.process(install=False)
    await _say(follower, "@investigator hello")  # it follows from here on, then goes quiet
    await a.install_strategy("r1", cluster.strategy())  # comms joins

    await follower.listen_only("r1", ["comms"])

    room = await cluster.store.get_room("r1")
    assert room is not None and room.metadata["_speak_queue"]["listening"] == ["comms"]
    assert follower._discussions["r1"].config.agents == list(AGENTS)
    await cluster.close()


async def test_a_direct_install_in_a_process_that_follows_takes_the_turns() -> None:
    cluster = _Cluster()
    a = await cluster.process(install=True, create=True)
    await a.close()
    b = await cluster.process(install=False)
    await _say(b, "@investigator are you there?")  # followed: queued, nobody gives it

    await cluster.strategy().install(b, "r1")

    async def answered() -> bool:
        return await cluster.answers() == ["investigator"]

    await _until(answered)
    await cluster.close()


@pytest.mark.parametrize("organization_id", ["org-a", None], ids=["scoped", "unscoped"])
async def test_a_process_that_never_read_the_room_claims_it_for_no_other_strategy(
    organization_id: str | None,
) -> None:
    """A claim reads the room as an install does: a discussion another
    process stored is this room's here too, and another strategy is refused
    (RFC §19.7, §19.7.5 rule 16)."""
    cluster = _Cluster()
    a = cluster.kit()
    await a.create_room(room_id="r1", organization_id="org-a")
    await a.install_strategy("r1", cluster.strategy(), organization_id="org-a")
    b = cluster.kit()

    swarm = Swarm(agents=["sales", "billing"], entry="sales")

    with pytest.raises(ValueError, match="holds a discussion"):
        await b.claim_room_strategy("r1", swarm, organization_id=organization_id)
    await cluster.close()


async def test_removing_the_agent_whose_turn_runs_cuts_it_in_the_process_giving_it() -> None:
    cluster = _Cluster()
    a = await cluster.process(install=True, create=True, delay=2.0)
    b = await cluster.process(install=True, delay=2.0)
    await _say(a, "@comms take your time")

    async def started() -> bool:
        return "+comms" in cluster.log

    await _until(started)
    await b.install_strategy("r1", cluster.strategy(names=KEPT, delay=2.0))

    async def ended() -> bool:
        queue = a.speak_queue("r1")
        return queue is not None and queue.speaking is None

    await _until(ended, timeout=3.0)
    assert await cluster.answers() == []
    await cluster.close()


async def test_uninstalling_in_another_process_leaves_no_record_behind() -> None:
    cluster = _Cluster()
    a = await cluster.process(install=True, create=True)
    b = await cluster.process(install=True)

    assert await b.uninstall_strategy("r1") is True

    async def left() -> bool:
        return a.room_strategy("r1") is None

    await _until(left)
    assert a.speak_queue("r1") is None
    await a.install_strategy("r1", cluster.strategy())  # the room takes a discussion again
    assert isinstance(a.room_strategy("r1"), Discussion)
    await cluster.close()


async def _turn_started(cluster: _Cluster, agent: str = "investigator") -> None:
    async def started() -> bool:
        return f"+{agent}" in cluster.log

    await _until(started)


async def _uninstall(kit: RoomKit, *, locked: bool) -> None:
    """Uninstall as a host does, under the room's lock when *locked*: it
    returns within a second whatever the turn running (RFC §19.7)."""
    async with asyncio.timeout(1.0):
        if not locked:
            assert await kit.uninstall_strategy("r1") is True
            return
        # A host serialises its own room changes under the room's lock.
        async with kit.lock_manager.locked("r1"):
            assert await kit.uninstall_strategy("r1") is True


@pytest.mark.parametrize("locked", [False, True], ids=["unlocked", "under_the_lock"])
@pytest.mark.parametrize("who", ["giving_the_turn", "other_installer", "follower"])
async def test_uninstalling_while_a_turn_runs_cuts_it_and_records_nothing(
    who: str, locked: bool
) -> None:
    cluster = _Cluster()
    a = await cluster.process(install=True, create=True, delay=1.0)
    b = await cluster.process(install=True, delay=1.0)
    follower = await cluster.process(install=False)
    await _say(follower, "@investigator take your time")
    await _turn_started(cluster)
    giving = next(k for k in (a, b) if k._discussions["r1"].driver.speaking == "investigator")
    driver = giving._discussions["r1"].driver
    kit = {"giving_the_turn": giving, "other_installer": b if giving is a else a}.get(
        who, follower
    )

    await _uninstall(kit, locked=locked)
    await _turn_over(cluster, driver)

    room = await cluster.store.get_room("r1")
    assert room is not None
    assert room.metadata.get("_speak_queue") is None
    assert room.metadata.get("_discussion") is None
    assert await cluster.answers() == []  # cut wherever it ran

    async def gone_everywhere() -> bool:
        return a.room_strategy("r1") is None and b.room_strategy("r1") is None

    await _until(gone_everywhere)  # each installer drops it once its driver reads the room
    await cluster.close()


async def test_uninstalling_under_the_room_lock_does_not_wait_for_an_observer_taking_it() -> None:
    cluster = _Cluster()
    a = await cluster.process(install=True, create=True, delay=1.0)

    @a.hook(HookTrigger.ON_SPEAK_QUEUE, execution=HookExecution.ASYNC)
    async def note(event: SpeakQueueEvent, context: Any) -> None:
        await a.update_room_metadata("r1", {"last_change": event.change.value})

    await _say(a, "@investigator take your time")
    await _turn_started(cluster)

    await _uninstall(a, locked=True)

    assert await cluster.stored() is None
    await cluster.close()


async def test_uninstalling_under_the_room_lock_returns_while_the_turn_runs_a_tool(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The cut turn stored its text before the call; its end, written on that
    # message, takes the room lock the host holds.
    called = asyncio.Event()

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        called.set()
        await asyncio.sleep(3.0)
        return "found"

    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="Looking.",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="c1", name="lookup", arguments={})],
            ),
            AIResponse(content="Done."),
        ],
        streaming=True,
    )
    tool = AITool(name="lookup", description="Look up", parameters={"type": "object"})
    agent = AIChannel(
        "investigator", provider=provider, tools=[tool], tool_handler=lookup, tool_search=False
    )
    kit = RoomKit()
    kit.register_channel(SimpleChannel("ops"))
    await kit.create_room(room_id="r1", orchestration=Discussion([agent]))
    await kit.attach_channel("r1", "ops")
    await _say(kit, "@investigator check")
    await asyncio.wait_for(called.wait(), 5.0)

    async def text_stored() -> bool:
        events = await kit.store.list_events("r1")
        return any(e.source.channel_id == "investigator" for e in events)

    await _until(text_stored)  # the cut turn's text, stored before its call

    caplog.set_level(logging.ERROR, logger="roomkit")
    await _uninstall(kit, locked=True)

    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR] == []
    await kit.close()


async def test_installing_under_the_room_lock_returns_mid_turn() -> None:
    cluster = _Cluster()
    a = cluster.kit()
    await a.create_room(room_id="r1")
    await a.attach_channel("r1", "ops")
    # The first install compiles the mark patterns (about 0.9 s, once per
    # process); the bounds below are about the installs, not that.
    await compile_mark_patterns()
    async with asyncio.timeout(1.0), a.lock_manager.locked("r1"):
        await a.install_strategy("r1", cluster.strategy(delay=1.0))
    await _say(a, "@investigator take your time")
    await _turn_started(cluster)
    driver = a._discussions["r1"].driver

    # Over the discussion, the speaking agent left out: its turn is cut.
    async with asyncio.timeout(1.0), a.lock_manager.locked("r1"):
        await a.install_strategy("r1", cluster.strategy(names=("sre", "comms")))
    await _turn_over(cluster, driver)

    assert await cluster.answers() == []
    stored = await cluster.stored()
    assert stored is not None and stored.agents == ["sre", "comms"]
    await cluster.close()


async def test_a_process_that_stops_during_a_turn_records_its_end_and_keeps_the_queue() -> None:
    cluster = _Cluster()
    a = await cluster.process(install=True, create=True, delay=1.5)
    await _say(a, "@investigator and @sre, what happened?")

    async def started() -> bool:
        return "+investigator" in cluster.log

    await _until(started)
    await a.close()

    room = await cluster.store.get_room("r1")
    assert room is not None
    queue = room.metadata["_speak_queue"]
    assert queue["last_speaker"] == "investigator"  # its end recorded
    assert [e["agent"] for e in queue["entries"]] == ["sre"]  # the queue kept (rule 16)
    assert await cluster.stored() is not None


async def test_a_process_whose_install_another_replaced_is_current_again_once_it_installs() -> (
    None
):
    # B's host read comms switched off a moment: it installs without it,
    # then stops. A follows B's agents and gives no turn: the room waits.
    cluster = _Cluster()
    a = await cluster.process(install=True, create=True)
    b = await cluster.process(install=True, names=KEPT)
    assert await b.room_strategy_is_current("r1", Discussion) is True
    assert await a.room_strategy_is_current("r1", Discussion) is False
    await b.close()  # its close gives the lease up: A may take it at once
    await _say(a, "@investigator are you there?")
    await asyncio.sleep(3 * _driver.POLL_SECONDS)  # A's driver has read it by now
    assert await cluster.answers() == []

    # A's host finds it false and installs the agents it holds.
    await a.install_strategy("r1", cluster.strategy())
    assert await a.room_strategy_is_current("r1", Discussion) is True

    async def answered() -> bool:
        return await cluster.answers() == ["investigator"]

    await _until(answered)
    await cluster.close()


async def test_a_process_that_installed_none_or_whose_discussion_went_is_not_current() -> None:
    cluster = _Cluster()
    a = await cluster.process(install=True, create=True)
    follower = await cluster.process(install=False)
    await _say(follower, "hello")  # it follows the stored discussion
    assert await follower.room_strategy_is_current("r1", Discussion) is False
    assert await a.room_strategy_is_current("r1", Discussion) is True

    assert await follower.uninstall_strategy("r1") is True

    # Read from the room as it is now, before A's driver polls; A takes the
    # uninstall at once, so its host sees the discussion gone.
    assert await a.room_strategy_is_current("r1", Discussion) is False
    assert a.room_strategy("r1") is None and a.speak_queue("r1") is None
    await cluster.close()


async def test_an_uninstall_elsewhere_is_taken_at_once_even_under_the_lock_mid_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A's driver does not look again before A's host asks.
    monkeypatch.setattr(_driver, "POLL_SECONDS", 5.0)
    cluster = _Cluster()
    a = await cluster.process(install=True, create=True, delay=1.0)
    b = await cluster.process(install=True)
    await _say(a, "@investigator take your time")
    await _turn_started(cluster)
    driver = a._discussions["r1"].driver
    assert await b.uninstall_strategy("r1") is True

    async with asyncio.timeout(1.0), a.lock_manager.locked("r1"):
        assert await a.room_strategy_is_current("r1", Discussion) is False
    assert a.room_strategy("r1") is None

    # A host that still wants the discussion would install it; this one
    # follows the uninstall, and nothing of A's cut turn comes back.
    await _turn_over(cluster, driver)
    room = await cluster.store.get_room("r1")
    assert room is not None and room.metadata.get("_speak_queue") is None
    assert await cluster.stored() is None
    assert await cluster.answers() == []
    await cluster.close()


async def test_the_same_configuration_installed_by_another_process_keeps_both_current() -> None:
    cluster = _Cluster()
    a = await cluster.process(install=True, create=True)
    b = await cluster.process(install=True)

    assert await a.room_strategy_is_current("r1", Discussion) is True
    assert await b.room_strategy_is_current("r1", Discussion) is True
    await cluster.close()


async def test_the_kind_asked_about_is_the_kind_installed() -> None:
    class Panel(Discussion):
        """A host's own discussion."""

    cluster = _Cluster()
    a = await cluster.process(install=True, create=True)

    assert await a.room_strategy_is_current("r1", Discussion) is True
    assert await a.room_strategy_is_current("r1", Panel) is False
    assert await a.uninstall_strategy("r1") is True
    assert await a.room_strategy_is_current("r1", Discussion) is False
    await cluster.close()


@pytest.mark.parametrize("kind", [Pipeline, Swarm, Loop, Supervisor, "instance", "not_a_strategy"])
async def test_asking_about_a_strategy_not_stored_with_the_room_is_refused(kind: Any) -> None:
    cluster = _Cluster()
    a = await cluster.process(install=True, create=True)
    asked = {"instance": a.room_strategy("r1"), "not_a_strategy": str}.get(kind, kind)

    with pytest.raises(TypeError, match="not a strategy stored with the room"):
        await a.room_strategy_is_current("r1", asked)
    await cluster.close()


async def test_a_room_of_another_organization_is_not_asked_about() -> None:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("ops"))
    await kit.create_room(room_id="r1", organization_id="org-a")
    await kit.install_strategy("r1", _Cluster().strategy(), organization_id="org-a")

    assert await kit.room_strategy_is_current("r1", Discussion, organization_id="org-a")
    with pytest.raises(RoomNotFoundError):
        await kit.room_strategy_is_current("r1", Discussion, organization_id="org-b")
    await kit.close()
