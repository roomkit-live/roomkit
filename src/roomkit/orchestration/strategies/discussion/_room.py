"""One room's discussion in this process: what fills its queue, its hooks (RFC §19.7.5).

The kit calls :meth:`DiscussionRoom.on_committed` for every event the room
commits and :meth:`DiscussionRoom.queue_instruction` for an instruction. Every
change of the queue goes through :meth:`DiscussionRoom.editing`, which reads
and writes the stored queue under the room lock, so every process serving the
room keeps one queue (rule 16). A process where the host installed the
discussion has a :class:`~._driver.TurnDriver`, which gives the turns while it
holds the lease; a process that only follows the discussion routes and queues
by the stored configuration and never gives a turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import sys
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any

from roomkit._media import content_words
from roomkit.core.hooks import HookRegistration
from roomkit.core.locks import _has_room_lock
from roomkit.core.task_utils import spared_by_close, start_named, wait_for_unspared
from roomkit.models.enums import EventStatus, EventType, HookExecution, HookTrigger
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.hook import HookResult
from roomkit.models.steering import Cancel
from roomkit.turns.wire import CommittedEvent

from ._config import DiscussionConfig
from ._driver import TurnDriver
from ._names import SilentToken, read_names
from ._people import (
    PeopleIndex,
    is_person,
    names_of,
    one_person,
    people_index,
    person_label,
    records_people,
    sees,
)
from ._posts import ResultPosts
from ._queue import Ask, Entry, Pending, SpeakQueueState
from ._shared import SharedQueue
from .models import SpeakQueue, SpeakQueueChange, SpeakQueueEvent

if TYPE_CHECKING:
    from roomkit.core.lanes import DeliveryPlan
    from roomkit.models.context import RoomContext

    from .strategy import Discussion

logger = logging.getLogger("roomkit.orchestration.discussion")

SILENT = "discussion_silent"
"""The ``blocked_by`` of an agent's row that says nothing (rule 13)."""

NAMES = "discussion_names"

_LAST = sys.maxsize
"""Priority of the names hook: after every other, so names are read in the
text the event commits with (rule 4); the silence check just before it."""

_STOP_WAIT = 5.0
"""How long stopping waits for the announcements still queued."""

_Change = tuple[SpeakQueueChange, Sequence[str], str | None]
"""What :meth:`DiscussionRoom.fire` announces."""


class DiscussionRoom:
    """The discussion a room holds, as this process takes part in it."""

    def __init__(
        self, kit: Any, room_id: str, config: DiscussionConfig, strategy: Discussion | None
    ) -> None:
        self.kit = kit
        self.room_id = room_id
        self.strategy = strategy
        self.own = config if strategy is not None else None
        """The configuration of the discussion the host installed in this
        process, if it installed one."""
        self._configure(config)
        self.organization_id: str | None = None
        self.shared = SharedQueue(kit, room_id)
        self.instructions: dict[str, RoomEvent] = {}
        self.turn_rows: list[RoomEvent] = []
        """What the speaking agent committed during its turn, read at its end."""
        self.closed = False
        self.forgotten = False
        """Uninstalled, here or as another process was read to have done: the
        queue goes, so the end of a turn cut on the way is not recorded (rule
        1), and this room gives no turn again, a discussion installed since
        included (rule 16)."""
        self.installs = 0
        """How many installs of this process's host took this room's place: a
        read begun before one tells nothing of it (rule 16)."""
        self.driver: TurnDriver | None = TurnDriver(self) if strategy is not None else None
        self._last_fire: asyncio.Task[None] | None = None
        self._announcing: set[asyncio.Task[None]] = set()
        self._holding: tuple[asyncio.Task[Any] | None, list[_Change]] | None = None
        """The task inside an edit of the queue and what it fired: one at a
        time, under the room lock."""
        self.posts = ResultPosts(kit, room_id, self._floor_free)
        """The task results to post as the agents' messages (rule 19)."""

    @classmethod
    def installed(cls, kit: Any, room_id: str, strategy: Discussion) -> DiscussionRoom:
        """The discussion the host installs in this process: it may give turns."""
        return cls(kit, room_id, DiscussionConfig.of(strategy, kit._max_chain_depth), strategy)

    @classmethod
    def following(cls, kit: Any, room_id: str, config: DiscussionConfig) -> DiscussionRoom:
        """A discussion another process installed: this one routes and queues."""
        return cls(kit, room_id, config, None)

    def _configure(self, config: DiscussionConfig) -> None:
        """Route and queue by *config*, each agent the channel the kit runs
        under its id, else the host's strategy's."""
        self.config = config
        given = self.strategy.agents() if self.strategy else []
        own = {a.channel_id: a for a in given if not isinstance(a, str)}
        # The kit's channel runs the turns: a strategy installed over another
        # may bring new objects under ids the kit has registered already.
        self.agents: dict[str, Any] = {
            agent_id: self.kit.channels.get(agent_id) or own.get(agent_id)
            for agent_id in config.agents
        }
        stored = config.handles or {}
        self.handles = {agent_id: stored.get(agent_id, agent_id) for agent_id in config.agents}
        """The name each agent is addressed by, by channel id (rule 4)."""
        self.by_handle = {handle: agent_id for agent_id, handle in self.handles.items()}
        self.silent = SilentToken(config.silent_token)
        self.max_depth = config.max_depth

    @property
    def drives(self) -> bool:
        """Whether this process may give the turns: its host installed the
        discussion the room stores (rule 16)."""
        return self.own is not None and self.own.same_as(self.config)

    def follow(self, config: DiscussionConfig) -> None:
        """Route and queue by *config*, the configuration the room stores now,
        which another process may have replaced (rule 16)."""
        if config == self.config:
            return
        if config.same_as(self.config):
            self.config = config
        else:
            self._configure(config)

    async def read(self) -> SpeakQueueState:
        """The stored queue, read now, without the lock; the room routed by
        the configuration stored with it from then on."""
        state = await self.shared.read()
        self._follow_stored()
        return state

    def _follow_stored(self) -> None:
        """Route by the configuration the last read found stored, if any."""
        if self.shared.config is not None:
            self.follow(self.shared.config)

    @property
    def state(self) -> SpeakQueueState:
        """The queue as last read or changed in this process."""
        return self.shared.state

    @contextlib.asynccontextmanager
    async def editing(self) -> AsyncGenerator[SpeakQueueState, None]:
        """Change the stored queue under the room lock, the room routed by the
        configuration stored with it (rule 16). What the block fires is
        announced once the queue is stored: a store that refuses the write
        changed nothing, so nothing is announced."""
        held: list[_Change] = []
        try:
            async with self.shared.editing() as state:
                self._follow_stored()
                self._holding = asyncio.current_task(), held
                try:
                    yield state
                except BaseException:
                    held.clear()  # no write follows a block that did not end
                    raise
                finally:
                    self._holding = None
        except Exception:
            held.clear()
            raise
        finally:
            # A stop cutting the write is no refusal: the turn it gave ends,
            # so it is told as given.
            for change in held:
                self.fire(*change)

    def _takes_nothing(self, state: SpeakQueueState) -> bool:
        """Whether the queue, read as *state* while editing, takes no turn:
        the discussion is over, this process left it, or the room no longer
        stores it (another process uninstalled it, rule 1)."""
        return state.over or self.closed or self.shared.config is None

    # -- Lifecycle --

    async def load(self) -> None:
        """Store the configuration and read the queue a previous install left."""
        room = await self.kit.store.get_room(self.room_id)
        self.organization_id = room.organization_id if room is not None else None
        await self.shared.store_config(self.config)
        async with self.editing() as state:
            # An instruction that names no process kept its text in one gone;
            # one another process holds is handed over when its turn comes.
            legacy = [e for e in state.entries if e.instruction is not None and e.holder is None]
            state.entries = [e for e in state.entries if e not in legacy]
            self.drop_instructions(legacy, keep=state)

    def hooks(self) -> list[HookRegistration]:
        """The room's hooks, after every other: an agent's silence blocked,
        then the names read in the text the event commits with (rules 4, 13)."""
        return [
            HookRegistration(
                trigger=HookTrigger.BEFORE_BROADCAST,
                execution=HookExecution.SYNC,
                fn=self._block_silence,
                priority=_LAST - 1,
                name=SILENT,
                event_types={EventType.MESSAGE},
            ),
            HookRegistration(
                trigger=HookTrigger.BEFORE_BROADCAST,
                execution=HookExecution.SYNC,
                fn=self._read_names,
                priority=_LAST,
                name=NAMES,
                event_types={EventType.MESSAGE},
            ),
        ]

    async def stop(self) -> None:
        """Stop taking part: the lease given up, the instruction turns only
        this process could give reported dropped; the queue stays (rule 16)."""
        self.closed = True
        if self.driver is not None:
            await self.driver.stop()
        try:
            await self.give_up()
        except Exception:
            logger.exception("The speak queue of room %s was not stored", self.room_id)
        await self._finish_announcing()

    async def give_up(self) -> None:
        """Give no turn from this process: the lease released, the instruction
        turns only this process could give reported dropped and the task
        results they were to post posted; the rest of the queue stays (rules
        16 and 19)."""
        async with self.editing() as state:
            self.shared.release(state)
            mine = [e for e in state.entries if e.holder == self.shared.me]
            state.entries = [e for e in state.entries if e not in mine]
            self.drop_instructions(mine)

    async def take_over(self, incoming: DiscussionRoom) -> None:
        """Hold *incoming*'s configuration and strategy in place of this
        room's, keeping the queue but for the agents it no longer holds
        (rule 1). The agents installs attached stay recorded but for those."""
        async with self.editing() as state:
            stored = self.shared.config
            gone = [a for a in stored.agents if a not in incoming.config.agents] if stored else []
            attached = [a for a in stored.attached if a not in gone] if stored else []
            config = incoming.config.model_copy(update={"attached": attached})
            await self.shared.store_config(config)
            self.drop_instructions(state.drop_agents(gone), keep=state)
            # The wait may have been for an agent that left.
            ended = state.end_wait_for_nobody()
        self.strategy, self.own = incoming.strategy, incoming.own
        self._configure(config)
        if ended:
            self.fire(SpeakQueueChange.WAITING, [])

    def drive(self) -> None:
        """Give the turns from this process: its driver started once, a room
        it only followed until now included."""
        if self.driver is None:
            self.driver = TurnDriver(self)
            self.driver.start()
        else:
            self.driver.wake()

    async def hold_attached(self, agents: Sequence[str]) -> None:
        """Store *agents*, which an install attached to the room, with the
        configuration, for any process uninstalling the discussion to detach
        them (RFC §19.7)."""
        async with self.editing():
            stored = self.shared.config
            if stored is None:
                return
            config = stored.model_copy(update={"attached": sorted({*stored.attached, *agents})})
            await self.shared.store_config(config)
        self.follow(config)

    async def forget(self) -> None:
        """Leave the discussion, for every process (rule 1): the turns
        stopped, the running one cut with its end recorded nowhere, then the
        queue (who listens, who asked whom, the turns given, whether it is
        over) and the configuration forgotten. The lease stays this process's
        until both are gone, so no other process gives a turn meanwhile. The
        task results the discussion took and has not committed, here or in
        another process, are posted before it returns, while their agents are
        still bound: an uninstall detaches them next (rule 19)."""
        self.forgotten = self.closed = True
        if self.driver is not None:
            await self.driver.stop()
        async with self.kit._lock_manager.locked(self.room_id):
            state = await self.shared.read()
            listed = await self.posts.take_listed()
            await self.shared.forget()
        self.drop_instructions([e for e in state.entries if e.holder == self.shared.me])
        await self.posts.post_all(listed)
        await self._finish_announcing()

    def handle(self, agent_id: str) -> str:
        """The name *agent_id* is addressed by: its handle, else its channel
        id (a stored turn of an agent the discussion no longer holds)."""
        return self.handles.get(agent_id, agent_id)

    def label(self, agent_id: str) -> str:
        """An agent as a model reads it named: ``@`` and its handle."""
        return f"@{self.handle(agent_id)}"

    def silence_for(self, channel_id: str) -> SilentToken | None:
        return self.silent if channel_id in self.agents else None

    def view(self) -> SpeakQueue:
        return self.state.view()

    # -- What the room commits --

    async def on_committed(self, event: RoomEvent, plan: DeliveryPlan | None) -> None:
        """Queue the agents a committed message asks for (rules 5 and 8): a
        person's at the front, any other sender's at the back. The rows of
        the turn this process runs are kept for its end."""
        if self.closed or event.type != EventType.MESSAGE or event.status == EventStatus.BLOCKED:
            return
        if plan is None:
            return
        # Who the event reaches is the router's delivery set (§10.1 step 12):
        # its memory hand-off and its queue never reach past it.
        reach = {t.channel_id for t in plan.targets}
        await self._hand_to_memories(event, plan.context, reach)
        speaker = event.source.channel_id
        if self.driver is not None and self.driver.speaking == speaker:
            self.turn_rows.append(event)
            return
        if speaker not in self.agents and is_person(event, plan.context):
            await self._route_person(event, plan.context, reach)
            # A person's message ends a wait even when it asks for no one.
            self.wake()
        elif await self._queue_named(event, reach, asker=speaker):
            self.wake()

    async def _hand_to_memories(
        self, event: RoomEvent, context: RoomContext, reach: set[str]
    ) -> None:
        """Hand *event* to the memory of every agent it reaches but its author,
        as a room with no discussion delivers it (rule 3): a memory that
        learns as messages arrive learns the whole conversation. It reaches
        each agent through the turn backend, where the agent runs (RFC
        §10.6.3), live or held by its declaration alone; an agent offline,
        which no executor serves, is handed nothing (RFC §10.6.7)."""
        handed_to = [
            agent_id
            for agent_id, agent in self.agents.items()
            if agent is not None and agent_id != event.source.channel_id and agent_id in reach
        ]
        offline = await self.kit._known_offline(self.room_id, handed_to)
        for agent_id in handed_to:
            if agent_id in offline:
                logger.debug("Agent %s is offline: its memory misses an event", agent_id)
                continue
            handed = CommittedEvent(channel_id=agent_id, room_id=self.room_id, event=event)
            try:
                await self.kit._turns.control(handed)
            except Exception:
                logger.warning(
                    "Agent %s's memory was not handed an event of room %s",
                    agent_id,
                    self.room_id,
                    exc_info=True,
                )

    async def queue_instruction(self, event: RoomEvent) -> None:
        """An instruction addressed to agents: a turn of its own each, at the
        front, taking the instruction as its input (rule 7). This process
        keeps its text; one that gives no turn (it only follows the
        discussion, or its host installed another) reports it dropped."""
        agents = [a for a in event.addressed_to or [] if a in self.agents]
        if not self.drives:
            self.fire(SpeakQueueChange.INSTRUCTION_DROPPED, agents, event.id)
            return
        async with self.editing() as state:
            if self._takes_nothing(state):
                self.fire(SpeakQueueChange.INSTRUCTION_DROPPED, agents, event.id)
                return
            if agents:
                self.instructions[event.id] = event
            for agent in agents:
                state.queue_instruction(agent, event.id, event.chain_depth, holder=self.shared.me)
        if agents:
            self.fire(SpeakQueueChange.QUEUED, agents, event.id)
            self.wake()

    async def queue_post(self, event: RoomEvent) -> bool:
        """A task's result to post as its agent's message (rule 19): a turn of
        its own, at the front, that posts it without a generation; whether
        this process took it. Over or stopped, the discussion posts it once
        no turn runs; a process that gives no turn (it only follows the
        discussion, or its host installed another) takes none, and the caller
        posts it at once."""
        agent = event.source.channel_id
        if not self.drives or agent not in self.agents:
            return False
        async with self.editing() as state:
            if self.shared.config is not None:
                # Listed until committed: any process uninstalling posts it.
                await self.posts.keep(event)
            if self._takes_nothing(state):
                self.posts.post_later(event)
                return True
            self.instructions[event.id] = event
            state.queue_instruction(
                agent, event.id, event.chain_depth, holder=self.shared.me, post=True
            )
        self.fire(SpeakQueueChange.QUEUED, [agent], event.id)
        self.wake()
        return True

    async def queue_regeneration(
        self, trigger: RoomEvent, agents: Sequence[str], *, depth: int | None = None
    ) -> list[str]:
        """A regenerated answer to *trigger*: a turn of its own, at the front,
        for each of *agents* not already queued for it (rule 7); the agents
        queued. *depth* is the chain the regeneration stands at when it is
        deeper than the trigger's own (a regeneration an ASYNC hook asked
        for, RFC §8.3): its turn answers at that depth, bounded by the
        discussion's limit (rules 9 and 11)."""
        async with self.editing() as state:
            if self._takes_nothing(state):
                return []
            ask = Ask(
                event_id=trigger.id,
                depth=max(trigger.chain_depth, depth or 0),
                asker="",
                index=trigger.index,
                person=True,
            )
            queued = [a for a in agents if a in self.agents and state.queue_regeneration(a, ask)]
        if queued:
            self.fire(SpeakQueueChange.QUEUED, queued, trigger.id)
            self.wake()
        return queued

    def asked_by(self, trigger: RoomEvent) -> list[str]:
        """The agents a person's message asks for by its address, else
        ``everyone`` (none with ``addressed_only``): who regenerates when no
        answer to it is left."""
        if trigger.addressed_to is not None:
            return [a for a in trigger.addressed_to if a in self.agents]
        if self.config.addressed_only:
            return []
        return self._everyone()

    # -- Listening only (rule 12) --

    async def listen_only(self, channel_ids: Sequence[str]) -> None:
        async with self.editing() as state:
            agents = [a for a in channel_ids if a in self.agents and a not in state.listening]
            state.listening.extend(agents)
            speaking = state.speaking if self.shared.holds(state) else None
        if agents:
            self.fire(SpeakQueueChange.LISTENING, agents)
        if speaking in agents:
            # Here, the turn runs in this process; elsewhere, its holder cuts it
            # once it reads the queue again.
            await self.cut(speaking)

    async def cut(self, agent: str, reason: str = "listen_only") -> None:
        """Cut *agent*'s running turn, as a ``Cancel`` does; a turn given but
        not started yet has no loop a Cancel reaches and is abandoned."""
        # An agent the discussion no longer holds still has its channel here.
        channel = self.agents.get(agent) or self.kit.channels.get(agent)
        # Its turn runs through the turn backend: the owner steers it, an
        # external agent's prompt cancelled as its own cancel does (RFC §6.4).
        reached = (
            await self.kit.steer(Cancel(reason=reason), agent, room_id=self.room_id)
            if channel is not None
            else 0
        )
        if not reached and self.driver is not None:
            await self.driver.cut(agent, reason)

    async def talk_again(self, channel_ids: Sequence[str]) -> None:
        async with self.editing() as state:
            agents = [a for a in channel_ids if a in state.listening]
            state.listening = [a for a in state.listening if a not in agents]
            # The wait may have been for an agent that only listened.
            ended = state.end_wait_for_nobody()
        if agents:
            self.fire(SpeakQueueChange.TALKING_AGAIN, agents)
        if ended:
            self.fire(SpeakQueueChange.WAITING, [])
        self.wake()

    def wake(self) -> None:
        if self.driver is not None:
            self.driver.wake()

    # -- Routing (rules 8 and 10) --

    async def _route_person(
        self, event: RoomEvent, context: RoomContext, reach: set[str]
    ) -> list[str]:
        """A person's message: the agents it addresses first; with no address
        and no name, the agents that asked that person, else ``everyone``, or
        those the dispatch policy picks; each only when the message reaches
        it."""
        who = person_label(event, context)
        async with self.editing() as state:
            if self._takes_nothing(state):
                return []
            overflow = None
            if self.config.dispatch and self._unaddressed(event, context):
                overflow = self._leave_for_dispatch(state, event, who, reach)
                agents: list[str] = []
            else:
                agents = [a for a in self._person_asks(state, event, context, who) if a in reach]
            self._ask_at_front(state, agents, event, who)
            # Routed first, then what was asked of them is answered.
            state.clear_asked(who, agents=agents if self.config.addressed_only else None)
            was_waiting = state.person_wrote()
        if overflow is not None and overflow[0]:
            self.fire(SpeakQueueChange.QUEUED, *overflow)
        if agents:
            self.fire(SpeakQueueChange.QUEUED, agents, event.id)
        if was_waiting:
            self.fire(SpeakQueueChange.WAITING, [])
        return agents

    @staticmethod
    def _ask_at_front(
        state: SpeakQueueState, agents: list[str], event: RoomEvent, who: str
    ) -> None:
        for agent in agents:
            ask = Ask(
                event_id=event.id,
                depth=event.chain_depth,
                asker=who,
                index=event.index,
                person=True,
                named=event.addressed_to is not None,
            )
            state.ask(agent, ask, front=True)

    def _person_asks(
        self, state: SpeakQueueState, event: RoomEvent, context: RoomContext, who: str
    ) -> list[str]:
        """The agents a person's message asks for with no dispatch policy."""
        if event.addressed_to is not None:
            return [a for a in event.addressed_to if a in self.agents]
        if not self._unaddressed(event, context):
            return []
        return state.asking(who) or self._everyone()

    def _unaddressed(self, event: RoomEvent, context: RoomContext) -> bool:
        """A message with no address and no name, which ``addressed_only``
        does not keep from every agent: rule 8 gives it to the agents that
        asked its author, else ``everyone``; a dispatch policy decides it."""
        if event.addressed_to is not None or self.config.addressed_only:
            return False
        return not self._names_people(event, context)

    def _leave_for_dispatch(
        self, state: SpeakQueueState, event: RoomEvent, who: str, reach: set[str]
    ) -> tuple[list[str], str] | None:
        """Leave an unaddressed message to the dispatch policy: the process
        holding the lease decides, off the lock, before its next turn (rule
        18). The candidates are the agents that asked its author, then
        ``everyone``. Once too many wait, the oldest asks what it asks with no
        policy; those agents and that message are returned."""
        asked = [a for a in state.asking(who) if a in reach]
        everyone = [a for a in self._everyone() if a in reach]
        candidates = list(dict.fromkeys([*asked, *everyone]))
        if not candidates:
            return None
        pending = Pending(
            event_id=event.id,
            depth=event.chain_depth,
            asker=who,
            index=event.index,
            candidates=candidates,
            asked=asked,
            seq=state.reserve_seq(),
        )
        oldest = state.wait_for_decision(pending)
        if oldest is None:
            return None
        logger.warning(
            "Room %s: too many messages wait for a dispatch decision; %s goes undecided",
            self.room_id,
            oldest.event_id,
        )
        agents = [a for a in oldest.undecided() if a not in state.listening]
        state.queue_picked(oldest, agents)
        return agents, oldest.event_id

    def _everyone(self) -> list[str]:
        everyone = self.config.everyone
        return list(self.agents if everyone is None else everyone)

    def _names_people(self, event: RoomEvent, context: RoomContext) -> bool:
        words = content_words(event.content)
        if not words:
            return False
        names = read_names(words, self.by_handle, people=self.people(context))
        return bool(names.people)

    async def _queue_named(self, event: RoomEvent, reach: set[str], *, asker: str) -> list[str]:
        """An event from an agent outside its turn, or from any other sender:
        the agents its address names and it reaches, at the back."""
        named = [
            a for a in event.addressed_to or [] if a in self.agents and a != asker and a in reach
        ]
        if not named:
            return []
        async with self.editing() as state:
            if self._takes_nothing(state):
                return []
            for agent in named:
                ask = Ask(
                    event_id=event.id, depth=event.chain_depth, asker=asker, index=event.index
                )
                state.ask(agent, ask)
        self.fire(SpeakQueueChange.QUEUED, named, event.id)
        return named

    def queue_turn_names(
        self, state: SpeakQueueState, agent: str, event: RoomEvent, context: RoomContext
    ) -> list[str]:
        """What one of a turn's delivered rows asks, once the turn has ended:
        the agents it names, at the back, and the people it names, as asked
        (rules 5 and 10). Call while editing *state*; the agents queued."""
        if event.type != EventType.MESSAGE or event.status == EventStatus.BLOCKED:
            return []
        if event.source.channel_id != agent:
            return []
        named = [
            a
            for a in event.addressed_to or []
            if a in self.agents and a != agent and sees(a, event, context)
        ]
        for other in named:
            ask = Ask(event_id=event.id, depth=event.chain_depth, asker=agent, index=event.index)
            state.ask(other, ask)
        if words := content_words(event.content):
            index = self.people_index(context)
            names = read_names(words, self.by_handle, people=names_of(index))
            for name in names.people:
                label = one_person(index, name)
                if label is not None:
                    state.record_asked(agent, label)
        return named

    # -- Who is who --

    def has_people(self, state: SpeakQueueState, context: RoomContext) -> bool:
        """Whether the room records a person, or one has written (rule 10)."""
        return state.people_spoke or records_people(context)

    def people(self, context: RoomContext) -> list[str]:
        """The names agents address people by, each once (§6.4)."""
        return names_of(self.people_index(context))

    def people_index(self, context: RoomContext) -> PeopleIndex:
        return people_index(context, self.handles.values(), self.config.people)

    # -- The hooks --

    async def _block_silence(self, event: RoomEvent, context: RoomContext) -> HookResult:
        said = event.content.body if isinstance(event.content, TextContent) else None
        silent = said is not None and self.silent.is_silent(said)
        if event.source.channel_id in self.agents and silent:
            return HookResult.block("silent")
        return HookResult.allow()

    async def _read_names(self, event: RoomEvent, context: RoomContext) -> HookResult:
        words = content_words(event.content)
        if not words or event.addressed_to == []:
            return HookResult.allow()
        speaker = event.source.channel_id
        names = read_names(
            words,
            self.by_handle,
            people=self.people(context),
            speaker=speaker if speaker in self.agents else None,
        )
        added = [a for a in names.agents if a not in (event.addressed_to or [])]
        if not added:
            return HookResult.allow()
        address = [*(event.addressed_to or []), *added]
        return HookResult.modify(event.model_copy(update={"addressed_to": address}))

    # -- Instructions and announcements --

    def drop_instructions(
        self, dropped: Sequence[Entry], *, keep: SpeakQueueState | None = None
    ) -> None:
        """Forget the text of instruction turns no longer queued, and report
        them. A task's result they were to post is posted at once: the task
        outlives the discussion (rule 19)."""
        for entry in dropped:
            if entry.instruction is None:
                continue
            posted = self.instructions.get(entry.instruction) if entry.post else None
            self.forget_instruction(entry.instruction, keep or self.state)
            if posted is not None:
                self.posts.post_later(posted)
                continue
            self.fire(SpeakQueueChange.INSTRUCTION_DROPPED, [entry.agent], entry.instruction)

    def _floor_free(self) -> bool:
        """Whether no turn runs in this process: a result to post waits for it."""
        return self.driver is None or self.driver.speaking is None

    def post_held(self) -> None:
        """Post every task result this process holds for a queued turn, the
        queue being gone: another process uninstalled the discussion (rule 19)."""
        self.posts.post_held(self.instructions, self.driver.posting if self.driver else None)

    def forget_instruction(self, instruction_id: str, state: SpeakQueueState) -> None:
        """Forget an instruction's text once no queued turn takes it (one
        instruction may be addressed to several agents)."""
        if not any(e.instruction == instruction_id for e in state.entries):
            self.instructions.pop(instruction_id, None)

    def fire(
        self, change: SpeakQueueChange, channel_ids: Sequence[str], event_id: str | None = None
    ) -> None:
        """Announce a change of the queue through ``ON_SPEAK_QUEUE``, in the
        order the changes happened: each announcement waits for the previous."""
        if not self.kit.hook_engine.has_hooks(HookTrigger.ON_SPEAK_QUEUE):
            return
        if self._holding is not None and self._holding[0] is asyncio.current_task():
            self._holding[1].append((change, channel_ids, event_id))
            return
        event = SpeakQueueEvent(
            room_id=self.room_id,
            queue=self.state.view(),
            change=change,
            channel_ids=tuple(channel_ids),
            event_id=event_id,
        )
        self.announce(lambda: self.kit._fire_speak_queue(event))

    def announce(self, run: Callable[[], Awaitable[None]]) -> None:
        """Run the hooks *run* calls once the announcements before it have run,
        without holding up the turns."""
        previous = self._last_fire

        async def announce() -> None:
            if previous is not None:
                await asyncio.wait([previous])
            await run()

        # Named in the hooks' held runs: a close from one spares the chain.
        task = start_named(announce(), context=contextvars.Context())
        self._last_fire = task
        self._announcing.add(task)
        task.add_done_callback(self._announcing.discard)

    async def _finish_announcing(self) -> None:
        """Let the queued announcements run, briefly: never from one of them
        (a hook's body closing the kit, one that joins the close included),
        which the rest wait for."""
        last = self._last_fire
        if last is None or spared_by_close() & self._announcing:
            return
        if _has_room_lock(self.room_id):
            # An announcement's hooks may take the room lock this caller holds.
            return
        await wait_for_unspared(
            lambda: () if spared_by_close() & self._announcing else (last,), timeout=_STOP_WAIT
        )
