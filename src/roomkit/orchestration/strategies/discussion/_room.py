"""One room's discussion: its speak queue and what fills it (RFC §19.7.5).

The kit calls :meth:`DiscussionRoom.on_committed` for every event the room
commits and :meth:`DiscussionRoom.queue_instruction` for an instruction; the
room's :class:`~._driver.TurnDriver` gives the turns. State changes take
``lock``. The driver takes it under the room lock, the order every path keeps
(room lock, then state), and nothing that holds it commits an event, since a
commit calls back into :meth:`on_committed`.
"""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from roomkit.core.hooks import HookRegistration
from roomkit.core.visibility import visibility_allows
from roomkit.models.enums import (
    Access,
    ChannelCategory,
    EventStatus,
    EventType,
    HookExecution,
    HookTrigger,
    ParticipantRole,
)
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.hook import HookResult
from roomkit.models.steering import Cancel

from ._driver import TurnDriver
from ._names import SilentToken, name_key, read_names
from ._queue import Ask, Entry, SpeakQueueState
from .models import SpeakQueue, SpeakQueueChange, SpeakQueueEvent

if TYPE_CHECKING:
    from roomkit.channels.ai import AIChannel
    from roomkit.core.lanes import DeliveryPlan
    from roomkit.models.context import RoomContext
    from roomkit.models.participant import Participant

    from .strategy import Discussion

STATE_KEY = "_speak_queue"
"""The room metadata key the speak queue is stored under."""

SILENT = "discussion_silent"
"""The ``blocked_by`` of an agent's row that says nothing (rule 13)."""

NAMES = "discussion_names"

_NOT_PEOPLE = frozenset({ParticipantRole.AGENT, ParticipantRole.BOT})
_READS = frozenset({Access.READ_WRITE, Access.READ_ONLY})


class DiscussionRoom:
    """The discussion a room holds, alive in the process that gives its turns."""

    def __init__(self, kit: Any, room_id: str, strategy: Discussion) -> None:
        self.kit = kit
        self.room_id = room_id
        self.strategy = strategy
        self.agents: dict[str, AIChannel] = {a.channel_id: a for a in strategy.agents()}
        self.silent = SilentToken(strategy.silent_token)
        self.max_depth: int = strategy.max_depth or kit._max_chain_depth
        self.state = SpeakQueueState()
        self.lock = asyncio.Lock()
        self.instructions: dict[str, RoomEvent] = {}
        self.driver = TurnDriver(self)
        self._last_fire: asyncio.Task[None] | None = None

    # -- Lifecycle --

    async def load(self) -> None:
        """Read the queue a previous install stored. Its instruction turns are
        dropped: their text lived in that process only (rule 16)."""
        room = await self.kit.store.get_room(self.room_id)
        stored = (room.metadata or {}).get(STATE_KEY) if room is not None else None
        async with self.lock:
            if stored:
                self.state = SpeakQueueState.model_validate(stored)
            self.drop_instructions(self.state.drop_instructions())
            await self.save()

    def hooks(self) -> list[HookRegistration]:
        """The room's hooks, after every other: an agent's silence blocked,
        then the names read in the text the event commits with (rules 4, 13)."""
        return [
            HookRegistration(
                trigger=HookTrigger.BEFORE_BROADCAST,
                execution=HookExecution.SYNC,
                fn=self._block_silence,
                priority=10_000,
                name=SILENT,
                event_types={EventType.MESSAGE},
            ),
            HookRegistration(
                trigger=HookTrigger.BEFORE_BROADCAST,
                execution=HookExecution.SYNC,
                fn=self._read_names,
                priority=10_001,
                name=NAMES,
                event_types={EventType.MESSAGE},
            ),
        ]

    async def stop(self) -> None:
        """Stop giving turns and finish the announcements; the queue stays stored."""
        await self.driver.stop()
        if self._last_fire is not None:
            await asyncio.wait([self._last_fire])

    async def drop(self) -> None:
        """Empty the queue, reporting the instruction turns it held (rule 1)."""
        async with self.lock:
            self.drop_instructions(self.state.drop())
            await self.save()

    def silence_for(self, channel_id: str) -> SilentToken | None:
        return self.silent if channel_id in self.agents else None

    def view(self) -> SpeakQueue:
        return self.state.view()

    # -- What the room commits --

    async def on_committed(self, event: RoomEvent, plan: DeliveryPlan | None) -> None:
        """Queue the agents a committed message asks for (rules 5 and 8): a
        person's at the front, any other sender's at the back. The speaking
        agent's own rows are read when its turn ends."""
        if (
            event.type != EventType.MESSAGE
            or event.status == EventStatus.BLOCKED
            or plan is None
            or self.state.over
        ):
            return
        speaker = event.source.channel_id
        if speaker == self.state.speaking:
            return
        if speaker not in self.agents and self.is_person(event, plan.context):
            queued = await self._route_person(event, plan.context)
        else:
            queued = await self._queue_named(event, plan.context, asker=speaker)
        if queued:
            self.driver.wake()

    async def queue_instruction(self, event: RoomEvent) -> None:
        """An instruction addressed to an agent: a turn of its own, at the
        front, taking the instruction as its input (rule 7)."""
        agents = [a for a in event.addressed_to or [] if a in self.agents]
        async with self.lock:
            if self.state.over:
                self.fire(SpeakQueueChange.INSTRUCTION_DROPPED, agents, event.id)
                return
            if agents:
                self.instructions[event.id] = event
            for agent in agents:
                self.state.queue_instruction(agent, event.id, event.chain_depth)
            await self.save()
        if agents:
            self.fire(SpeakQueueChange.QUEUED, agents, event.id)
            self.driver.wake()

    async def queue_regeneration(self, trigger: RoomEvent, agents: Sequence[str]) -> list[str]:
        """A regenerated answer to *trigger*: a turn of its own, at the front,
        for each of *agents* (rule 7); the agents queued."""
        queued = [a for a in agents if a in self.agents]
        async with self.lock:
            if self.state.over or not queued:
                return []
            for agent in queued:
                ask = Ask(event_id=trigger.id, depth=trigger.chain_depth, asker="", person=True)
                self.state.queue_regeneration(agent, ask)
            await self.save()
        self.fire(SpeakQueueChange.QUEUED, queued, trigger.id)
        self.driver.wake()
        return queued

    # -- Listening only (rule 12) --

    async def listen_only(self, channel_ids: Sequence[str]) -> None:
        agents = [a for a in channel_ids if a in self.agents]
        async with self.lock:
            agents = [a for a in agents if a not in self.state.listening]
            self.state.listening.extend(agents)
            speaking = self.state.speaking
            await self.save()
        if speaking in agents:
            self.agents[speaking].steer(Cancel(reason="listen_only"), room_id=self.room_id)
        if agents:
            self.fire(SpeakQueueChange.LISTENING, agents)

    async def talk_again(self, channel_ids: Sequence[str]) -> None:
        async with self.lock:
            agents = [a for a in channel_ids if a in self.state.listening]
            self.state.listening = [a for a in self.state.listening if a not in agents]
            await self.save()
        if agents:
            self.fire(SpeakQueueChange.TALKING_AGAIN, agents)
            self.driver.wake()

    # -- Routing (rules 8 and 10) --

    async def _route_person(self, event: RoomEvent, context: RoomContext) -> list[str]:
        """A person's message: the agents it addresses first; with no address
        and no name, the agents that asked that person, else ``everyone``."""
        who = self._asked_key(event, context)
        async with self.lock:
            named = [
                a for a in self._person_asks(event, context, who) if self.sees(a, event, context)
            ]
            for agent in named:
                ask = Ask(event_id=event.id, depth=event.chain_depth, asker=who or "", person=True)
                self.state.ask(agent, ask, front=True)
            # Routed first, then what was asked of them is answered.
            self.state.clear_asked(who, agents=named if self.strategy.addressed_only else None)
            self.state.person_wrote()
            await self.save()
        if named:
            self.fire(SpeakQueueChange.QUEUED, named, event.id)
        return named

    def _person_asks(self, event: RoomEvent, context: RoomContext, who: str | None) -> list[str]:
        if event.addressed_to is not None:
            return [a for a in event.addressed_to if a in self.agents]
        if self.strategy.addressed_only or self._names_people(event, context):
            return []
        everyone = self.strategy.everyone
        return self.state.asking(who) or list(self.agents if everyone is None else everyone)

    def _names_people(self, event: RoomEvent, context: RoomContext) -> bool:
        if not isinstance(event.content, TextContent):
            return False
        names = read_names(event.content.body, self.agents, people=self.people(context))
        return bool(names.people)

    def _asked_key(self, event: RoomEvent, context: RoomContext) -> str | None:
        """The person of ``asked`` the sender is, or None when the discussion
        cannot tell the room's people apart by the sender's name."""
        key = name_key(person_name(event, context)).lower()
        known = {name_key(p).lower() for p in self.people(context)}
        return key if key and key in known else None

    async def _queue_named(
        self, event: RoomEvent, context: RoomContext, *, asker: str
    ) -> list[str]:
        """An event from an agent outside its turn, or from any other sender:
        the agents its address names, at the back."""
        named = [
            a
            for a in event.addressed_to or []
            if a in self.agents and a != asker and self.sees(a, event, context)
        ]
        if not named:
            return []
        async with self.lock:
            for agent in named:
                self.state.ask(agent, Ask(event_id=event.id, depth=event.chain_depth, asker=asker))
            await self.save()
        self.fire(SpeakQueueChange.QUEUED, named, event.id)
        return named

    def queue_turn_names(self, agent: str, event: RoomEvent, context: RoomContext) -> list[str]:
        """What one of a turn's delivered rows asks, once the turn has ended:
        the agents it names, at the back, and the people it names, as asked
        (rules 5 and 10). Call under ``lock``; the agents queued."""
        if event.type != EventType.MESSAGE or event.status == EventStatus.BLOCKED:
            return []
        if event.source.channel_id != agent:
            return []
        named = [
            a
            for a in event.addressed_to or []
            if a in self.agents and a != agent and self.sees(a, event, context)
        ]
        for other in named:
            self.state.ask(other, Ask(event_id=event.id, depth=event.chain_depth, asker=agent))
        if isinstance(event.content, TextContent):
            names = read_names(event.content.body, self.agents, people=self.people(context))
            for person in names.people:
                self.state.record_asked(agent, name_key(person).lower())
        return named

    def sees(self, agent: str, event: RoomEvent, context: RoomContext) -> bool:
        """Whether *agent* may read *event*: a name is no way around visibility."""
        binding = context.get_binding(agent)
        return (
            binding is not None
            and binding.access in _READS
            and visibility_allows(event.visibility, binding)
        )

    def is_person(self, event: RoomEvent, context: RoomContext) -> bool:
        """A transport's event from a participant that is neither an agent nor
        a bot, or with no participant record behind it (rule 2)."""
        binding = context.get_binding(event.source.channel_id)
        if binding is None or binding.category != ChannelCategory.TRANSPORT:
            return False
        participant = _participant(event, context)
        return participant is None or participant.role not in _NOT_PEOPLE

    def people(self, context: RoomContext) -> list[str]:
        """The names agents address people by: ``people``, else the room's,
        kept to an identifier's characters."""
        if self.strategy.people is not None:
            return [name for p in self.strategy.people if (name := name_key(p))]
        return [
            name
            for p in context.participants
            if p.role not in _NOT_PEOPLE and (name := name_key(p.display_name or p.id))
        ]

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

    # -- Storage and announcements --

    async def save(self) -> None:
        await self.kit.store.patch_room_metadata(
            self.room_id, {STATE_KEY: self.state.model_dump(mode="json")}
        )

    def drop_instructions(self, dropped: Sequence[Entry]) -> None:
        """Forget the text of instruction turns no longer queued, and report them."""
        for entry in dropped:
            if entry.instruction is None:
                continue
            self.instructions.pop(entry.instruction, None)
            self.fire(SpeakQueueChange.INSTRUCTION_DROPPED, [entry.agent], entry.instruction)

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

        self._last_fire = asyncio.get_running_loop().create_task(
            announce(), context=contextvars.Context()
        )


def _participant(event: RoomEvent, context: RoomContext) -> Participant | None:
    pid = event.source.participant_id
    return next((p for p in context.participants if p.id == pid), None) if pid else None


def person_name(event: RoomEvent, context: RoomContext) -> str:
    """The name a person's message is told apart by: their participant's name,
    the name their transport stamped, else their channel."""
    participant = _participant(event, context)
    if participant is not None:
        return participant.display_name or participant.id
    sender = (event.metadata or {}).get("sender_name")
    return sender if isinstance(sender, str) and sender else event.source.channel_id
