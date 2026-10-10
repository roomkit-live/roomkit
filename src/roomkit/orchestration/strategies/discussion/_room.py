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
import contextvars
import logging
import sys
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any

from roomkit.core.hooks import HookRegistration
from roomkit.models.enums import EventStatus, EventType, HookExecution, HookTrigger
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.hook import HookResult
from roomkit.models.steering import Cancel

from ._config import DiscussionConfig
from ._driver import TurnDriver
from ._names import SilentToken, read_names
from ._people import (
    PeopleIndex,
    is_person,
    names_of,
    people_index,
    person_label,
    records_people,
    sees,
)
from ._queue import Ask, Entry, SpeakQueueState
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


class DiscussionRoom:
    """The discussion a room holds, as this process takes part in it."""

    def __init__(
        self, kit: Any, room_id: str, config: DiscussionConfig, strategy: Discussion | None
    ) -> None:
        self.kit = kit
        self.room_id = room_id
        self.config = config
        self.strategy = strategy
        own = {a.channel_id: a for a in strategy.agents()} if strategy is not None else {}
        self.agents: dict[str, Any] = {
            agent_id: own.get(agent_id) or kit.channels.get(agent_id) for agent_id in config.agents
        }
        self.silent = SilentToken(config.silent_token)
        self.max_depth = config.max_depth
        self.organization_id: str | None = None
        self.shared = SharedQueue(kit, room_id)
        self.instructions: dict[str, RoomEvent] = {}
        self.turn_rows: list[RoomEvent] = []
        """What the speaking agent committed during its turn, read at its end."""
        self.closed = False
        self.driver: TurnDriver | None = TurnDriver(self) if strategy is not None else None
        self._last_fire: asyncio.Task[None] | None = None
        self._announcing: set[asyncio.Task[None]] = set()

    @classmethod
    def installed(cls, kit: Any, room_id: str, strategy: Discussion) -> DiscussionRoom:
        """The discussion the host installs in this process: it may give turns."""
        return cls(kit, room_id, DiscussionConfig.of(strategy, kit._max_chain_depth), strategy)

    @classmethod
    def following(cls, kit: Any, room_id: str, config: DiscussionConfig) -> DiscussionRoom:
        """A discussion another process installed: this one routes and queues."""
        return cls(kit, room_id, config, None)

    @property
    def state(self) -> SpeakQueueState:
        """The queue as last read or changed in this process."""
        return self.shared.state

    def editing(self) -> AbstractAsyncContextManager[SpeakQueueState]:
        return self.shared.editing()

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
            async with self.editing() as state:
                self.shared.release(state)
                mine = [e for e in state.entries if e.holder == self.shared.me]
                state.entries = [e for e in state.entries if e not in mine]
                self.drop_instructions(mine)
        except Exception:
            logger.exception("The speak queue of room %s was not stored", self.room_id)
        await self._finish_announcing()

    async def reset(self) -> None:
        """Forget the discussion, for every process: the queue, who listens,
        who asked whom, the turns given and whether it is over (rule 1)."""
        async with self.kit._lock_manager.locked(self.room_id):
            state = await self.shared.read()
            mine = [e for e in state.entries if e.holder == self.shared.me]
            await self.shared.forget()
        self.drop_instructions(mine)
        await self._finish_announcing()

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
        learns as messages arrive learns the whole conversation."""
        for agent_id, agent in self.agents.items():
            if agent is None or agent_id == event.source.channel_id or agent_id not in reach:
                continue
            ingest = getattr(agent, "_ingest_event", None)
            if ingest is not None:
                await ingest(event, context)

    async def queue_instruction(self, event: RoomEvent) -> None:
        """An instruction addressed to agents: a turn of its own each, at the
        front, taking the instruction as its input (rule 7). This process
        keeps its text; one that only follows the discussion gives no turn,
        so it reports the instruction dropped."""
        agents = [a for a in event.addressed_to or [] if a in self.agents]
        if self.driver is None:
            self.fire(SpeakQueueChange.INSTRUCTION_DROPPED, agents, event.id)
            return
        async with self.editing() as state:
            if state.over or self.closed:
                self.fire(SpeakQueueChange.INSTRUCTION_DROPPED, agents, event.id)
                return
            if agents:
                self.instructions[event.id] = event
            for agent in agents:
                state.queue_instruction(agent, event.id, event.chain_depth, holder=self.shared.me)
        if agents:
            self.fire(SpeakQueueChange.QUEUED, agents, event.id)
            self.wake()

    async def queue_regeneration(self, trigger: RoomEvent, agents: Sequence[str]) -> list[str]:
        """A regenerated answer to *trigger*: a turn of its own, at the front,
        for each of *agents* not already queued for it (rule 7); the agents
        queued."""
        async with self.editing() as state:
            if state.over or self.closed:
                return []
            ask = Ask(event_id=trigger.id, depth=trigger.chain_depth, asker="", person=True)
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

    async def cut(self, agent: str) -> None:
        """Cut *agent*'s running turn, as a ``Cancel`` does; a turn given but
        not started yet has no loop a Cancel reaches and is abandoned."""
        channel = self.agents.get(agent)
        cancel = Cancel(reason="listen_only")
        reached = channel.steer(cancel, room_id=self.room_id) if channel is not None else 0
        if not reached and self.driver is not None:
            await self.driver.cut(agent, "listen_only")

    async def talk_again(self, channel_ids: Sequence[str]) -> None:
        async with self.editing() as state:
            agents = [a for a in channel_ids if a in state.listening]
            state.listening = [a for a in state.listening if a not in agents]
            # The wait may have been for an agent that only listened.
            ended = state.waiting and not state.owes_a_person(has_people=True)
            if ended:
                state.waiting = False
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
        and no name, the agents that asked that person, else ``everyone``;
        each only when the message reaches it."""
        who = person_label(event, context)
        named = event.addressed_to is not None
        async with self.editing() as state:
            if state.over or self.closed:
                return []
            agents = [a for a in self._person_asks(state, event, context, who) if a in reach]
            for agent in agents:
                ask = Ask(
                    event_id=event.id, depth=event.chain_depth, asker=who, person=True, named=named
                )
                state.ask(agent, ask, front=True)
            # Routed first, then what was asked of them is answered.
            state.clear_asked(who, agents=agents if self.config.addressed_only else None)
            was_waiting = state.person_wrote()
        if agents:
            self.fire(SpeakQueueChange.QUEUED, agents, event.id)
        if was_waiting:
            self.fire(SpeakQueueChange.WAITING, [])
        return agents

    def _person_asks(
        self, state: SpeakQueueState, event: RoomEvent, context: RoomContext, who: str
    ) -> list[str]:
        if event.addressed_to is not None:
            return [a for a in event.addressed_to if a in self.agents]
        if self.config.addressed_only or self._names_people(event, context):
            return []
        return state.asking(who) or self._everyone()

    def _everyone(self) -> list[str]:
        everyone = self.config.everyone
        return list(self.agents if everyone is None else everyone)

    def _names_people(self, event: RoomEvent, context: RoomContext) -> bool:
        if not isinstance(event.content, TextContent):
            return False
        names = read_names(event.content.body, self.agents, people=self.people(context))
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
            if state.over or self.closed:
                return []
            for agent in named:
                state.ask(agent, Ask(event_id=event.id, depth=event.chain_depth, asker=asker))
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
            state.ask(other, Ask(event_id=event.id, depth=event.chain_depth, asker=agent))
        if isinstance(event.content, TextContent):
            index = self.people_index(context)
            names = read_names(event.content.body, self.agents, people=names_of(index))
            for name in names.people:
                labels = index.get(name.casefold(), [])
                # A name two people answer to records no one: neither may
                # answer for the other.
                if len(labels) == 1:
                    state.record_asked(agent, labels[0])
        return named

    # -- Who is who --

    def has_people(self, state: SpeakQueueState, context: RoomContext) -> bool:
        """Whether the room records a person, or one has written (rule 10)."""
        return state.people_spoke or records_people(context)

    def people(self, context: RoomContext) -> list[str]:
        """The names agents address people by, each once (§6.4)."""
        return names_of(self.people_index(context))

    def people_index(self, context: RoomContext) -> PeopleIndex:
        return people_index(context, self.agents, self.config.people)

    # -- The hooks --

    async def _block_silence(self, event: RoomEvent, context: RoomContext) -> HookResult:
        said = event.content.body if isinstance(event.content, TextContent) else None
        silent = said is not None and self.silent.is_silent(said)
        if event.source.channel_id in self.agents and silent:
            return HookResult.block("silent")
        return HookResult.allow()

    async def _read_names(self, event: RoomEvent, context: RoomContext) -> HookResult:
        if not isinstance(event.content, TextContent) or event.addressed_to == []:
            return HookResult.allow()
        speaker = event.source.channel_id
        names = read_names(
            event.content.body,
            self.agents,
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
        """Forget the text of instruction turns no longer queued, and report them."""
        for entry in dropped:
            if entry.instruction is None:
                continue
            self.forget_instruction(entry.instruction, keep or self.state)
            self.fire(SpeakQueueChange.INSTRUCTION_DROPPED, [entry.agent], entry.instruction)

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
        event = SpeakQueueEvent(
            room_id=self.room_id,
            queue=self.state.view(),
            change=change,
            channel_ids=tuple(channel_ids),
            event_id=event_id,
        )
        previous = self._last_fire

        async def announce() -> None:
            if previous is not None:
                await asyncio.wait([previous])
            await self.kit._fire_speak_queue(event)

        task = asyncio.get_running_loop().create_task(announce(), context=contextvars.Context())
        self._last_fire = task
        self._announcing.add(task)
        task.add_done_callback(self._announcing.discard)

    async def _finish_announcing(self) -> None:
        """Let the queued announcements run, briefly: never from one of them,
        which the rest wait for."""
        last = self._last_fire
        if last is None or asyncio.current_task() in self._announcing:
            return
        await asyncio.wait([last], timeout=_STOP_WAIT)
