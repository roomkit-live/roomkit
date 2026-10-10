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
import logging
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from roomkit.channels._speaker import label_name, participant_name, turn_labels
from roomkit.core.hooks import HookRegistration
from roomkit.core.visibility import effective_visibility, visibility_allows
from roomkit.models.delivery import SYSTEM_SENDER_ID
from roomkit.models.enums import (
    Access,
    ChannelCategory,
    EventStatus,
    EventType,
    HookExecution,
    HookTrigger,
    ParticipantRole,
    ParticipantStatus,
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

logger = logging.getLogger("roomkit.orchestration.discussion")

STATE_KEY = "_speak_queue"
"""The room metadata key the speak queue is stored under."""

SILENT = "discussion_silent"
"""The ``blocked_by`` of an agent's row that says nothing (rule 13)."""

NAMES = "discussion_names"

_LAST = sys.maxsize
"""Priority of the names hook: after every other, so names are read in the
text the event commits with (rule 4); the silence check just before it."""

_STOP_WAIT = 5.0
"""How long stopping waits for the announcements still queued."""

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
        self.organization_id: str | None = None
        self.state = SpeakQueueState()
        self.lock = asyncio.Lock()
        self.instructions: dict[str, RoomEvent] = {}
        self.turn_rows: list[RoomEvent] = []
        """What the speaking agent committed during its turn, read at its end."""
        self.closed = False
        self.driver = TurnDriver(self)
        self._last_fire: asyncio.Task[None] | None = None
        self._announcing: set[asyncio.Task[None]] = set()

    # -- Lifecycle --

    async def load(self) -> None:
        """Read the queue a previous install stored. Its instruction turns are
        dropped: their text lived in that process only (rule 16)."""
        room = await self.kit.store.get_room(self.room_id)
        stored = (room.metadata or {}).get(STATE_KEY) if room is not None else None
        self.organization_id = room.organization_id if room is not None else None
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
        """Stop giving turns; the queue stays stored but for its instruction
        turns, which this process alone could give (rule 16)."""
        self.closed = True
        await self.driver.stop()
        try:
            async with self.lock:
                self.drop_instructions(self.state.drop_instructions())
                await self.save()
        except Exception:
            logger.exception("The speak queue of room %s was not stored", self.room_id)
        await self._finish_announcing()

    async def reset(self) -> None:
        """Forget the discussion: the queue, who listens, who asked whom, the
        turns given and whether it is over (rule 1)."""
        async with self.lock:
            dropped = self.state.drop()
            self.state = SpeakQueueState()
            self.drop_instructions(dropped)
            await self.kit.store.patch_room_metadata(self.room_id, {STATE_KEY: None})
        await self._finish_announcing()

    def silence_for(self, channel_id: str) -> SilentToken | None:
        return self.silent if channel_id in self.agents else None

    def view(self) -> SpeakQueue:
        return self.state.view()

    # -- What the room commits --

    async def on_committed(self, event: RoomEvent, plan: DeliveryPlan | None) -> None:
        """Queue the agents a committed message asks for (rules 5 and 8): a
        person's at the front, any other sender's at the back. The speaking
        agent's own rows are kept for its turn's end."""
        if (
            self.closed
            or event.type != EventType.MESSAGE
            or event.status == EventStatus.BLOCKED
            or plan is None
            or self.state.over
        ):
            return
        speaker = event.source.channel_id
        if speaker == self.state.speaking:
            self.turn_rows.append(event)
            return
        if speaker not in self.agents and self.is_person(event, plan.context):
            await self._route_person(event, plan.context)
            # A person's message ends a wait even when it asks for no one.
            self.driver.wake()
        elif await self._queue_named(event, plan.context, asker=speaker):
            self.driver.wake()

    async def queue_instruction(self, event: RoomEvent) -> None:
        """An instruction addressed to an agent: a turn of its own, at the
        front, taking the instruction as its input (rule 7)."""
        agents = [a for a in event.addressed_to or [] if a in self.agents]
        async with self.lock:
            if self.state.over or self.closed:
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
        for each of *agents* not already queued for it (rule 7); the agents
        queued."""
        async with self.lock:
            if self.state.over or self.closed:
                return []
            ask = Ask(event_id=trigger.id, depth=trigger.chain_depth, asker="", person=True)
            queued = [
                a for a in agents if a in self.agents and self.state.queue_regeneration(a, ask)
            ]
            if queued:
                await self.save()
        if queued:
            self.fire(SpeakQueueChange.QUEUED, queued, trigger.id)
            self.driver.wake()
        return queued

    def asked_by(self, trigger: RoomEvent) -> list[str]:
        """The agents a person's message asks for by its address, else
        ``everyone`` (none with ``addressed_only``): who regenerates when no
        answer to it is left."""
        if trigger.addressed_to is not None:
            return [a for a in trigger.addressed_to if a in self.agents]
        if self.strategy.addressed_only:
            return []
        everyone = self.strategy.everyone
        return list(self.agents if everyone is None else everyone)

    # -- Listening only (rule 12) --

    async def listen_only(self, channel_ids: Sequence[str]) -> None:
        async with self.lock:
            agents = [a for a in channel_ids if a in self.agents and a not in self.state.listening]
            self.state.listening.extend(agents)
            speaking = self.state.speaking
            await self.save()
        if agents:
            self.fire(SpeakQueueChange.LISTENING, agents)
        if speaking in agents:
            # A turn given but not started yet has no loop a Cancel reaches.
            cancel = Cancel(reason="listen_only")
            reached = self.agents[speaking].steer(cancel, room_id=self.room_id)
            if not reached:
                await self.driver.cut(speaking, "listen_only")

    async def talk_again(self, channel_ids: Sequence[str]) -> None:
        async with self.lock:
            agents = [a for a in channel_ids if a in self.state.listening]
            self.state.listening = [a for a in self.state.listening if a not in agents]
            # The wait may have been for an agent that only listened.
            ended = self.state.waiting and not self.state.owes_a_person(has_people=True)
            if ended:
                self.state.waiting = False
            await self.save()
        if agents:
            self.fire(SpeakQueueChange.TALKING_AGAIN, agents)
        if ended:
            self.fire(SpeakQueueChange.WAITING, [])
        self.driver.wake()

    # -- Routing (rules 8 and 10) --

    async def _route_person(self, event: RoomEvent, context: RoomContext) -> list[str]:
        """A person's message: the agents it addresses first; with no address
        and no name, the agents that asked that person, else ``everyone``."""
        who = self.person_label(event, context)
        named = event.addressed_to is not None
        async with self.lock:
            if self.state.over or self.closed:
                return []
            agents = [
                a for a in self._person_asks(event, context, who) if self.sees(a, event, context)
            ]
            for agent in agents:
                ask = Ask(
                    event_id=event.id, depth=event.chain_depth, asker=who, person=True, named=named
                )
                self.state.ask(agent, ask, front=True)
            # Routed first, then what was asked of them is answered.
            self.state.clear_asked(who, agents=agents if self.strategy.addressed_only else None)
            was_waiting = self.state.person_wrote()
            await self.save()
        if agents:
            self.fire(SpeakQueueChange.QUEUED, agents, event.id)
        if was_waiting:
            self.fire(SpeakQueueChange.WAITING, [])
        return agents

    def _person_asks(self, event: RoomEvent, context: RoomContext, who: str) -> list[str]:
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
            if self.state.over or self.closed:
                return []
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
                self.state.record_asked(agent, person)
        return named

    # -- Who is who --

    def sees(self, agent: str, event: RoomEvent, context: RoomContext) -> bool:
        """Whether *agent* may read *event*: a name is no way around visibility.
        The scope is the event's own, else its source binding's (§7.5)."""
        binding = context.get_binding(agent)
        if binding is None or binding.access not in _READS:
            return False
        scope = effective_visibility(event, context.get_binding(event.source.channel_id))
        return visibility_allows(scope, binding)

    def is_person(self, event: RoomEvent, context: RoomContext) -> bool:
        """A transport's event from a participant that is neither an agent nor
        a bot, or with no participant record behind it (rule 2); never the
        framework's own sender of a delivery (§22)."""
        if event.source.participant_id == SYSTEM_SENDER_ID:
            return False
        binding = context.get_binding(event.source.channel_id)
        if binding is None or binding.category != ChannelCategory.TRANSPORT:
            return False
        participant = _participant(event, context)
        return participant is None or participant.role not in _NOT_PEOPLE

    def has_people(self, context: RoomContext) -> bool:
        """Whether the room records a person, or one has written (rule 10)."""
        return self.state.people_spoke or any(_is_person(p) for p in context.participants)

    def person_label(self, event: RoomEvent, context: RoomContext) -> str:
        """The person a message is from, as the transcript labels them: their
        name, with the rank a look-alike of an earlier source carries
        (``Alice (2)``), so a sender who takes another's name is not them."""
        label = turn_labels([event], context).get(event.id)
        # A sender with no name reads as their channel (``@sms1``): the name
        # agents address them by drops the ``@``.
        return (label or event.source.channel_id).removeprefix("@")

    def people(self, context: RoomContext) -> list[str]:
        """The names agents address people by: ``people``, else the room's
        active people and the speakers of its recent messages (§6.4), kept to
        a name's characters, each once, never an agent's channel id."""
        if self.strategy.people is not None:
            names = [name_key(p) for p in self.strategy.people]
        else:
            names = [name_key(participant_name(p)) for p in context.participants if _is_person(p)]
            labels = turn_labels(
                [e for e in context.recent_events if self.is_person(e, context)], context
            )
            names += [name_key(label) for label in labels.values() if label and _plain(label)]
        taken = {a.casefold() for a in self.agents}
        kept: dict[str, str] = {}
        for name in names:
            if name and name.casefold() not in taken:
                kept.setdefault(name.casefold(), name)
        return list(kept.values())

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
            self.forget_instruction(entry.instruction)
            self.fire(SpeakQueueChange.INSTRUCTION_DROPPED, [entry.agent], entry.instruction)

    def forget_instruction(self, instruction_id: str) -> None:
        """Forget an instruction's text once no queued turn takes it (one
        instruction may be addressed to several agents)."""
        if not any(e.instruction == instruction_id for e in self.state.entries):
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


def _participant(event: RoomEvent, context: RoomContext) -> Participant | None:
    pid = event.source.participant_id
    return next((p for p in context.participants if p.id == pid), None) if pid else None


def _is_person(participant: Participant) -> bool:
    return participant.status == ParticipantStatus.ACTIVE and participant.role not in _NOT_PEOPLE


def _plain(label: str) -> bool:
    """Whether *label* is a name as written: not a look-alike's ranked one, nor
    a nameless sender's channel (``@sms1``)."""
    return label == label_name(label) and not label.startswith("@")
