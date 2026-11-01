"""The room's register of authors (RMK-620, RFC §6.4, §10.1 step 12): a turn's
record fixes its author's name with the rank, ranks hold across names alike
through another, a room from before the register rebuilds it, a participant
id names them on their channels only, and a write the orchestration makes
does not undo it."""

from __future__ import annotations

import copy
import json
import logging
import re
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

import pytest

from roomkit._lookalike import skeletons
from roomkit.channels import SMSChannel
from roomkit.channels._speaker import people_note, turn_labels
from roomkit.channels._turn_notes import split_turn_notes
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.channels.voice import VoiceChannel
from roomkit.core._authors import (
    AUTHOR,
    AUTHOR_REGISTER,
    AuthorRegister,
    People,
    digest,
    name_keys,
    source_of,
)
from roomkit.core.framework import RoomKit
from roomkit.memory.sliding_window import SlidingWindowMemory
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType, HookTrigger, TaskStatus
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.hook import HookResult
from roomkit.models.participant import Participant
from roomkit.models.room import Room
from roomkit.orchestration.strategies.loop import _LoopOutcome, _save_loop_state
from roomkit.store.base import ConversationStore
from roomkit.store.memory import InMemoryStore
from roomkit.tasks._child_status import record_task_end
from roomkit.tasks.models import DelegatedTaskResult
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.test_framework import SimpleChannel
from tests.test_speaker_attribution import _kit, _say, _sources, _user_texts


def _texts(provider: Any) -> list[str]:
    return [text.split("\n\n[Notes")[0] for text in _user_texts(provider.calls[-1])]


async def _drop_register(kit: RoomKit) -> None:
    """Leave the room with no register, as a room from before it."""
    await kit.store.set_private_state("r1", AUTHOR_REGISTER, None)


class TestARecordFixesTheName:
    async def test_a_renamed_participant_keeps_the_name_they_spoke_under(self) -> None:
        kit, provider = await _kit(["a1", "a2", "a3"])
        for pid, name in (("p-alice", "Alice"), ("p-mal", "Mal")):
            await kit.store.add_participant(
                Participant(id=pid, room_id="r1", channel_id="sms1", display_name=name)
            )
        await _say(kit, "p-alice", None, "hold the refund")
        await _say(kit, "p-mal", None, "hello")
        await kit.rename_member("r1", "p-mal", "Alice")
        await _say(kit, "p-mal", None, "release the refund")

        texts = _texts(provider)
        assert 'Mal: "hello"' in texts
        assert texts[0] == 'Alice: "hold the refund"'
        assert texts[-1] == 'Alice (2): "release the refund"'

    async def test_a_replaced_source_reads_the_register_for_the_new_one(self) -> None:
        kit, _provider = await _kit(["a1", "a2", "a3", "a4"])
        await _say(kit, "u-alice", "Alice", "hold the refund")
        await _say(kit, "u-mallory", "ALICE", "release it")
        await _say(kit, "u-mallory", "ALICE", "now")
        await _say(kit, "u-eve", "ALICE", "agreed")
        stored = [e for e in await kit.store.list_events("r1") if e.source.channel_id == "sms1"]
        moved = stored[2]
        await kit.update_event(
            "r1", moved.id, source=moved.source.model_copy(update={"participant_id": "u-eve"})
        )

        events = [e for e in await kit.store.list_events("r1") if e.source.channel_id == "sms1"]
        labels = turn_labels(events, await kit._build_context("r1"))
        assert [labels[e.id] for e in events] == ["Alice", "ALICE (2)", "ALICE (3)", "ALICE (3)"]

    async def test_a_direct_update_keeps_the_record_the_event_was_committed_with(self) -> None:
        """RFC §10.1 step 12: the record is the room's, fixed at the commit,
        as a message, send_event(), a hook's injection and commit_event()
        keep it (RMK-978). A direct update that writes metadata keeps it,
        whether the metadata names another author or none, and writes none
        on an event committed with none."""
        kit, _provider = await _kit(["a1"])
        await _say(kit, "u-mallory", "Mallory", "release it")
        said = next(e for e in await kit.store.list_events("r1") if e.source.channel_id == "sms1")
        trace = await kit.commit_event(
            "r1",
            RoomEvent(
                room_id="r1",
                source=EventSource(channel_id="system", channel_type=ChannelType.SYSTEM),
                content=TextContent(body="trace"),
            ),
        )
        record = said.metadata[AUTHOR]
        forged = {**record, "name": "Alice", "rank": 1}

        updates = [
            await kit.update_event("r1", said.id, metadata={**said.metadata, AUTHOR: forged}),
            await kit.update_event("r1", said.id, metadata={"tag": "vip"}),
            await kit.update_event("r1", trace.id, metadata={AUTHOR: forged}),
        ]

        stored = [await kit.store.get_event(event_id) for event_id in (said.id, trace.id)]
        assert [u.metadata.get(AUTHOR) for u in updates if u is not None] == [record, record, None]
        assert [(e.metadata.get(AUTHOR), e.metadata.get("tag")) for e in stored if e] == [
            (record, "vip"),
            (None, None),
        ]
        await kit.close()


class TestTheRecordIsMadeFromWhatTheLockRead:
    """RMK-1032: an inbound commit makes its author's record from the room's
    context read under the lock, after the hooks that ran there (RFC §10.1
    steps 6 and 9a), and reads neither the room, its participants nor its
    register again."""

    async def test_the_commit_reads_neither_the_room_nor_its_people_again(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        kit, _provider = await _kit(["a1"])
        committing: ContextVar[bool] = ContextVar("committing", default=False)
        reads: list[str] = []
        for name in ("get_room", "list_participants", "get_private_state"):
            noted = _noted(getattr(kit.store, name), committing, reads)
            monkeypatch.setattr(kit.store, name, noted)
        run_commit = kit._run_commit

        async def committed(*args: Any, **kwargs: Any) -> Any:
            token = committing.set(True)
            try:
                return await run_commit(*args, **kwargs)
            finally:
                committing.reset(token)

        monkeypatch.setattr(kit, "_run_commit", committed)
        await _say(kit, "u-alice", "Alice", "hold the refund")

        said = await _said(kit)
        assert (reads, said.metadata[AUTHOR]["name"]) == ([], "Alice")
        await kit.close()

    async def test_a_name_a_locked_hook_gave_is_the_one_recorded(self) -> None:
        kit, _provider = await _kit(["a1"])
        mal = Participant(id="p-mal", room_id="r1", channel_id="sms1", display_name="Mal")
        await kit.store.add_participant(mal)

        @kit.hook(HookTrigger.BEFORE_BROADCAST)
        async def rename(event: RoomEvent, ctx: RoomContext) -> HookResult:
            await kit.store.update_participant(mal.model_copy(update={"display_name": "Malcolm"}))
            return HookResult.allow()

        await _say(kit, "p-mal", None, "hello")

        said = await _said(kit)
        assert said.metadata[AUTHOR]["name"] == "Malcolm"
        await kit.close()


async def _said(kit: RoomKit) -> RoomEvent:
    """The one event the room's SMS channel stored."""
    (said,) = [e for e in await kit.store.list_events("r1") if e.source.channel_id == "sms1"]
    return said


def _noted(read: Any, committing: ContextVar[bool], reads: list[str]) -> Any:
    """*read*, noting its name in *reads* when called inside a commit."""

    async def call(*args: Any, **kwargs: Any) -> Any:
        if committing.get():
            reads.append(read.__name__)
        return await read(*args, **kwargs)

    return call


class TestRanksHoldAcrossNames:
    async def test_names_alike_through_another_never_share_a_rank(self) -> None:
        kit, provider = await _kit(["a1", "a2", "a3"])
        await _say(kit, "u-1", "Lan", "first")
        await _say(kit, "u-2", "Ian", "second")
        await _say(kit, "u-3", "ian", "third")

        assert _texts(provider) == ['Lan: "first"', 'Ian (2): "second"', 'ian (3): "third"']

    def test_a_name_alike_two_others_ranks_after_both_and_renumbers_neither(self) -> None:
        register = AuthorRegister()
        lan, ian, both = {"lan"}, {"ian"}, {"lan", "ian"}
        a, b, c = "a" * 16, "b" * 16, "c" * 16

        assert register.seat(a, lan) == 1
        assert register.seat(b, ian) == 1
        assert register.rank(c, both) == 2
        assert register.seat(b, ian) == 1
        assert register.seat(a, lan) == 1

    async def test_a_registered_participant_keeps_the_name_a_bridging_sender_reads_like(
        self,
    ) -> None:
        kit, provider = await _kit(["a1", "a2", "a3"])
        for pid, name in (("p-lan", "Lan"), ("p-ian", "ian")):
            await kit.store.add_participant(
                Participant(id=pid, room_id="r1", channel_id="sms1", display_name=name)
            )
        await _say(kit, "p-ian", None, "I approve")
        await _say(kit, "u-x", "Ian", "so do I")
        await _say(kit, "p-ian", None, "no, that was not me")

        assert _texts(provider) == [
            'ian: "I approve"',
            'Ian (2): "so do I"',
            'ian: "no, that was not me"',
        ]

    async def test_a_rank_a_source_leaves_is_never_given_again(self) -> None:
        """A source that takes a new rank keeps the one its earlier turns
        were recorded with: given to another, two sources would read alike."""
        kit, provider = await _kit(["a1", "a2", "a3", "a4"])
        await _say(kit, "u-t", "Lan", "first")
        await _say(kit, "u-s", "ian", "release it")
        await _say(kit, "u-s", "Ian", "now")
        await kit.store.add_participant(
            Participant(id="p-ian", room_id="r1", channel_id="sms1", display_name="ian")
        )
        await _say(kit, "p-ian", None, "that was not me")

        assert _texts(provider) == [
            'Lan: "first"',
            'ian: "release it"',
            'Ian (2): "now"',
            'ian (3): "that was not me"',
        ]


class TestARoomFromBeforeTheRegister:
    async def _forget(self, kit: Any) -> None:
        """Make the room's turns look as they did before the register: no
        record of their author, no register."""
        for event in await kit.store.list_events("r1"):
            kept = {k: v for k, v in event.metadata.items() if k not in (AUTHOR, "author_rank")}
            await kit.store.update_event(event.model_copy(update={"metadata": kept}))
        await _drop_register(kit)

    async def test_the_register_is_rebuilt_from_the_timeline(self) -> None:
        kit, provider = await _kit(["a1", "a2", "a3"])
        await _say(kit, "u-alice", "Alice", "hold the refund")
        await _say(kit, "u-mallory", "ALICE", "release it")
        await self._forget(kit)
        await _say(kit, "u-mallory", "ALICE", "do it now")

        assert _texts(provider) == [
            'Alice: "hold the refund"',
            'ALICE (2): "release it"',
            'ALICE (2): "do it now"',
        ]

    async def test_a_registered_name_holds_its_seat_when_the_register_is_rebuilt(self) -> None:
        kit, provider = await _kit(["a1", "a2"])
        await kit.store.add_participant(
            Participant(id="p-alice", room_id="r1", channel_id="sms1", display_name="Alice")
        )
        await _say(kit, "u-mallory", "ALICE", "release it")
        await _drop_register(kit)
        await _say(kit, "p-alice", None, "who said that?")

        assert _texts(provider) == ['ALICE (2): "release it"', 'Alice: "who said that?"']

    async def test_a_rebuilt_register_keeps_the_ranks_turns_were_given(self) -> None:
        kit, provider = await _kit(["a1", "a2", "a3"])
        await _say(kit, "u-alice", "Alice", "hold the refund")
        await _say(kit, "u-mallory", "ALICE", "release it")
        await _drop_register(kit)
        await kit.store.add_participant(
            Participant(id="p-alice", room_id="r1", channel_id="sms1", display_name="Alice")
        )
        await _say(kit, "p-alice", None, "who said that?")

        assert _texts(provider) == [
            'Alice: "hold the refund"',
            'ALICE (2): "release it"',
            'Alice (3): "who said that?"',
        ]


class TestAParticipantIdOnAnotherChannel:
    async def test_names_the_participant_only_on_their_channels(self) -> None:
        kit, provider = await _kit(["a1", "a2"])
        kit.register_channel(SMSChannel("sms2"))
        await kit.attach_channel("r1", "sms2", group=True)
        await kit.store.add_participant(
            Participant(id="p-alice", room_id="r1", channel_id="sms1", display_name="Alice")
        )
        await _say(kit, "p-alice", None, "hold the refund")
        await kit.process_inbound(
            InboundMessage(
                channel_id="sms2", sender_id="p-alice", content=TextContent(body="release it")
            )
        )

        assert _texts(provider) == ['Alice: "hold the refund"', '@sms2: "release it"']

    async def test_a_voice_join_reaches_them_on_the_voice_channel(self) -> None:
        kit, provider = await _kit(["a1", "a2"])
        kit.register_channel(
            VoiceChannel("voice", backend=MockVoiceBackend(), pipeline=AudioPipelineConfig())
        )
        await kit.attach_channel("r1", "voice")
        await kit.ensure_participant("r1", "sms1", "p-alice", display_name="Alice")
        session = await kit.join("r1", "voice", participant_id="p-alice")
        await _say(kit, "u-bob", "Bob", "hi all")
        await kit.process_inbound(
            InboundMessage(
                channel_id="voice",
                sender_id=session.participant_id,
                content=TextContent(body="refund the order"),
            ),
            room_id="r1",
        )

        assert _texts(provider) == ['Bob: "hi all"', 'Alice: "refund the order"']
        await kit.close()

    async def test_a_realtime_session_reaches_them_on_its_channel(self) -> None:
        kit, _provider = await _kit(["a1"])
        realtime = RealtimeVoiceChannel(
            "rt", provider=MockRealtimeProvider(), transport=MockRealtimeTransport()
        )
        kit.register_channel(realtime)
        await kit.attach_channel("r1", "rt")
        await kit.ensure_participant("r1", "sms1", "p-alice", display_name="Alice")
        await realtime.start_session("r1", "p-alice", connection=None)

        alice = await kit.store.get_participant("r1", "p-alice")
        assert alice is not None and alice.connected_via == ["sms1", "rt"]
        await kit.close()

    def test_an_identity_names_them_on_a_channel_it_was_resolved_on(self) -> None:
        """The identity pipeline records the channel it resolved them on (RFC
        §5.5); the identity's id on any other channel names nobody."""
        person = Participant(id="p1", room_id="r", channel_id="sms1", identity_id="i1")
        resolved = Participant(id="i1", room_id="r", channel_id="email1", identity_id="i1")
        people = People([person, resolved])

        def event(channel: str, pid: str) -> RoomEvent:
            source = EventSource(
                channel_id=channel, channel_type=ChannelType.SMS, participant_id=pid
            )
            return RoomEvent(room_id="r", source=source, content=TextContent(body="x"))

        assert people.of(event("sms1", "p1")) is person
        assert people.of(event("email1", "i1")) is person
        assert people.of(event("email1", "p1")) is None
        assert people.of(event("web1", "i1")) is None


class TestAnIdentityIsOnePerson:
    """Two records of one identity (a member the host added under it, and the
    record the identity pipeline keeps for it) are one source, one rank."""

    async def test_a_member_added_under_their_identity_keeps_their_name(self) -> None:
        kit, provider = await _kit(["a1", "a2"])
        await kit.store.add_participant(
            Participant(
                id="member:u-alice",
                room_id="r1",
                channel_id="member:u-alice",
                display_name="Alice",
                identity_id="u-alice",
            )
        )
        await kit.store.add_participant(
            Participant(
                id="u-alice",
                room_id="r1",
                channel_id="sms1",
                display_name="Alice",
                identity_id="u-alice",
            )
        )
        await _say(kit, "u-bob", "Bob", "hi all")
        await _say(kit, "u-alice", "Alice", "hold the refund")

        assert _texts(provider) == ['Bob: "hi all"', 'Alice: "hold the refund"']

    @staticmethod
    def _records() -> tuple[Participant, Participant]:
        """The member the host added under identity ``i1``, then the record
        the identity pipeline keeps for it."""
        first, later = (
            Participant(
                id=pid,
                room_id="r",
                channel_id=channel,
                display_name="Alice",
                identity_id="i1",
                joined_at=datetime(2026, 1, day, tzinfo=UTC),
            )
            for pid, channel, day in (("member:i1", "member:i1", 1), ("i1", "sms1", 2))
        )
        return first, later

    @staticmethod
    def _turn(metadata: dict[str, Any] | None = None) -> RoomEvent:
        source = EventSource(channel_id="sms1", channel_type=ChannelType.SMS, participant_id="i1")
        return RoomEvent(
            room_id="r", source=source, content=TextContent(body="x"), metadata=metadata or {}
        )

    def test_the_first_record_stands_for_the_identity_whatever_the_order(self) -> None:
        first, later = self._records()
        event = self._turn()

        for records in ([first, later], [later, first]):
            people = People(records)
            assert people.of(event) is first
            assert source_of(event, people) == ["participant", "member:i1"]

    def test_a_turn_recorded_under_the_other_record_reads_as_its_person(self) -> None:
        """The register 0.96.0 kept: the member seated, the pipeline's record
        ranked after it, and a turn recorded with that rank."""
        first, later = self._records()
        names = name_keys("r", "Alice")
        register = AuthorRegister()
        register.seat(digest("r", ["participant", "member:i1"]), names)
        other = digest("r", ["participant", "i1"])
        recorded = {"name": "Alice", "rank": register.rank(other, names), "source": other}
        event = self._turn({"sender_name": "Alice", AUTHOR: recorded})
        context = RoomContext(
            room=Room(id="r"), participants=[first, later], author_register=register.data
        )

        labels = turn_labels([event], context)
        assert recorded["rank"] == 2
        assert labels[event.id] == "Alice"


class TestAWriteMadeMeanwhileDoesNotUndoIt:
    """A room read, then a commit, then a write: the write keeps the register."""

    async def _commit_while_reading(self, kit: Any) -> None:
        read = kit.get_room

        async def get_room(room_id: str, **kwargs: Any) -> Any:
            room = await read(room_id, **kwargs)
            await _say(kit, "u-alice", "Alice", "hold the refund")
            return room

        kit.get_room = get_room

    async def _register_size(self, kit: Any) -> int:
        return len(
            _sources(await kit.store.get_private_state("r1", AUTHOR_REGISTER) or {"names": {}})
        )

    async def test_the_loop_state_keeps_the_register(self) -> None:
        kit, _provider = await _kit(["a1"])
        await self._commit_while_reading(kit)
        outcome = _LoopOutcome(approved=True, iteration=1, stopped="approved")
        await _save_loop_state(kit, "r1", outcome)

        assert await self._register_size(kit) == 1

    async def test_a_task_s_end_keeps_the_register(self) -> None:
        kit, _provider = await _kit(["a1"])
        await self._commit_while_reading(kit)
        result = DelegatedTaskResult(
            task_id="t1",
            child_room_id="r1",
            parent_room_id="r0",
            agent_id="ai1",
            status=TaskStatus.COMPLETED,
            output="done",
        )
        await record_task_end(kit, result)

        room = await kit.store.get_room("r1")
        assert room.metadata["task_status"] == TaskStatus.COMPLETED
        assert await self._register_size(kit) == 1


def test_a_context_with_no_register_ranks_by_the_window() -> None:
    def said(pid: str, name: str) -> RoomEvent:
        source = EventSource(channel_id="sms1", channel_type=ChannelType.SMS, participant_id=pid)
        return RoomEvent(
            room_id="r",
            source=source,
            content=TextContent(body="x"),
            metadata={"sender_name": name},
        )

    first, second = said("u1", "Alice"), said("u2", "ALICE")
    labels = turn_labels([first, second], RoomContext(room=Room(id="r")))

    assert labels == {first.id: "Alice", second.id: "ALICE (2)"}


_DIGEST = re.compile(r"[0-9a-f]{16}")


class _StoreFromBefore(InMemoryStore):
    """A store written before the register was kept apart: the base class's
    methods, which keep it in the room's metadata."""

    get_private_state = ConversationStore.get_private_state
    set_private_state = ConversationStore.set_private_state


def _upgraded(store: InMemoryStore, upgraded: bool = True) -> None:
    """*store*'s data read and written by a process whose store keeps the
    register apart, or by a process from before (``upgraded=False``): an
    upgrade, or a rollback, on a shared store."""
    store.__class__ = InMemoryStore if upgraded else _StoreFromBefore


class TestTheStoreKeepsItApart:
    """RMK-631: the register's digests hash the room's id, which is not
    secret, with a sender's id or name, a small space: whoever reads them
    finds the sender by trying them all. The store keeps the register apart
    from the room's metadata, which clients may read (RFC §10.1 step 12)."""

    ALICE, BOB = "+15145550123", "+15145559876"

    def _found(self, text: str) -> tuple[list[tuple[str, str]], list[str]]:
        """The senders and names whose digests *text* holds: every number of
        Alice's and Bob's exchange on either channel, and a few names."""
        held = set(_DIGEST.findall(text))
        numbers = [
            (channel, number)
            for channel in ("sms1", "sms2")
            for number in (f"+1514555{n:04d}" for n in range(10_000))
            if digest("r1", ["sender", channel, number]) in held
        ]
        names = [
            name
            for name in ("Alice", "Bob", "Carol", "Mallory")
            if any(digest("r1", [key]) in held for key in skeletons(name))
        ]
        return numbers, names

    async def test_a_side_cannot_find_the_sender_of_a_turn_it_does_not_see(self) -> None:
        kit, _provider = await _kit(["a1", "a2"])
        kit.register_channel(SMSChannel("sms2"))
        await kit.attach_channel("r1", "sms2", visibility="ai1")
        await _say(kit, self.ALICE, "Alice", "hello")
        await kit.process_inbound(
            InboundMessage(
                channel_id="sms2",
                sender_id=self.BOB,
                content=TextContent(body="hello"),
                metadata={"sender_name": "Bob"},
            )
        )

        room = await kit.get_room("r1")
        timeline = await kit.get_timeline("r1")
        seen = [e for e in timeline if e.visibility == "all"]
        read = room.model_dump_json() + "".join(e.model_dump_json() for e in seen)
        assert [e.visibility for e in timeline if e.source.channel_id == "sms2"] == ["ai1"]
        assert self._found(read) == ([("sms1", self.ALICE)], [])
        assert AUTHOR_REGISTER not in room.metadata
        # What the store keeps would name Bob: the attack is the real one.
        register = await kit.store.get_private_state("r1", AUTHOR_REGISTER)
        assert ("sms2", self.BOB) in self._found(json.dumps(register))[0]
        await kit.close()

    async def test_a_hook_reads_it_from_the_context_not_from_the_room(self) -> None:
        kit, _provider = await _kit(["a1", "a2"])
        contexts: list[RoomContext] = []

        @kit.hook(HookTrigger.BEFORE_BROADCAST)
        async def keep(event: RoomEvent, context: RoomContext) -> HookResult:
            contexts.append(context)
            return HookResult.allow()

        await _say(kit, "u-alice", "Alice", "hello")
        await _say(kit, "u-bob", "Bob", "hi")

        assert all(AUTHOR_REGISTER not in c.room.metadata for c in contexts)
        alice = digest("r1", ["sender", "sms1", "u-alice"])
        assert alice in _sources(contexts[-1].author_register or {"names": {}})
        await kit.close()

    @pytest.mark.parametrize("window", [None, 0], ids=["history", "no-history"])
    async def test_the_people_note_reads_it_across_the_turn_wire(self, window: int | None) -> None:
        """A turn's request crosses as JSON, in this process too: a seat the
        register gave reaches the agent's turn notes with its context, an
        agent whose memory reads no history included (its context loads no
        event, and still the register): read without it, the note would name
        the seated Alice as the sender who took the name first."""
        memory = SlidingWindowMemory(window) if window is not None else None
        kit, provider = await _kit(["a1", "a2"], memory=memory)
        await _say(kit, "u-x", "Alice", "first")
        for pid, name in (("p-alice", "Alice"), ("p-bob", "Bob")):
            await kit.store.add_participant(
                Participant(id=pid, room_id="r1", channel_id="sms1", display_name=name)
            )
        await _say(kit, "p-bob", None, "second")

        _said, notes = split_turn_notes(str(provider.calls[-1].messages[-1].content))
        assert people_note(("Alice (2)", "Bob")) in notes
        await kit.close()

    async def test_a_room_from_before_keeps_its_ranks_once_its_register_moved(self) -> None:
        """The register moves as it is, its digests unchanged: a rank its
        timeline no longer shows (the turn deleted) stays given, so the
        impostor who speaks first after the move takes the next one, where a
        register rebuilt from the timeline would give it again. Another kit
        on the store reads the same labels. The room a reader held before
        the move is the store's snapshot and stays as it was (RMK-984)."""
        store = _StoreFromBefore()
        kit, _provider = await _kit(["a1", "a2", "a3", "a4"], store=store)
        await _say(kit, "u-alice", "Alice", "hold the refund")
        await _say(kit, "u-mallory", "ALICE", "release it")
        released = next(e for e in await store.list_events("r1") if e.content.body == "release it")
        await store.delete_event("r1", released.id)

        _upgraded(store)
        held = await store.get_room("r1")
        held_register = copy.deepcopy(held.metadata[AUTHOR_REGISTER])
        await _say(kit, "u-eve", "ALICE", "release it now")
        await _say(kit, "u-alice", "Alice", "no, keep it")

        events = [e for e in await store.list_events("r1") if e.source.channel_id == "sms1"]
        other = RoomKit(store=store)
        labels = [turn_labels(events, await k._build_context("r1")) for k in (kit, other)]
        assert [e.metadata[AUTHOR]["rank"] for e in events] == [1, 3, 1]
        assert AUTHOR_REGISTER not in (await store.get_room("r1")).metadata
        assert held.metadata[AUTHOR_REGISTER] == held_register
        for read in labels:
            assert [read[e.id] for e in events] == ["Alice", "ALICE (3)", "Alice"]
        await kit.close()
        await other.close()

    async def test_a_rank_a_process_from_before_gave_is_not_given_again(self) -> None:
        """A process from before, on the store an upgraded one moved the
        register out of (a deployment upgrading, or rolled back), ranks from
        the metadata and writes there: the store's register takes its ranks
        at the next turn, so none is given to two sources."""
        store = InMemoryStore()
        kit, _provider = await _kit(["a1", "a2", "a3"], store=store)
        await _say(kit, "u-alice", "Alice", "hold the refund")
        _upgraded(store, False)
        await _say(kit, "u-mallory", "ALICE", "release it")
        _upgraded(store)
        await _say(kit, "u-eve", "Alice", "release it, I'm Alice")

        events = [e for e in await store.list_events("r1") if e.source.channel_id == "sms1"]
        ranks = {e.source.participant_id: e.metadata[AUTHOR]["rank"] for e in events}
        assert ranks == {"u-alice": 1, "u-mallory": 2, "u-eve": 3}
        assert AUTHOR_REGISTER not in (await store.get_room("r1")).metadata
        await kit.close()

    async def test_a_store_without_its_methods_keeps_it_in_the_metadata(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The base class keeps a store written before working, the register
        where it was, and says once that it is not private there."""
        store = _StoreFromBefore()
        kit, provider = await _kit(["a1", "a2"], store=store)
        with caplog.at_level(logging.WARNING, logger="roomkit.store"):
            await _say(kit, "u-alice", "Alice", "hello")
            await _say(kit, "u-mallory", "ALICE", "approve")

        warned = [r for r in caplog.records if "get_private_state" in r.getMessage()]
        assert len(warned) == 1
        assert len(_sources((await store.get_room("r1")).metadata[AUTHOR_REGISTER])) == 2
        assert _texts(provider)[-1] == 'ALICE (2): "approve"'
        await kit.close()


def _forged_register(sender_id: str, name: str) -> dict[str, Any]:
    """A register giving the source of *sender_id* on ``sms1`` rank 1 under
    *name* in room ``r1``: what a writer of the room's metadata would plant
    to take the name another sender speaks under (RMK-978)."""
    source = digest("r1", ["sender", "sms1", sender_id])
    return {"version": 3, "names": {key: f"{source}:1" for key in name_keys("r1", name)}}


class _Updating(SimpleChannel):
    """A transport whose first delivery returns room metadata updates (RFC
    §10.1 step 15)."""

    def __init__(self, channel_id: str, updates: dict[str, Any]) -> None:
        super().__init__(channel_id)
        self._updates = updates

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        updates, self._updates = self._updates, {}
        return ChannelOutput(metadata_updates=updates)


async def _plant(kit: RoomKit, writer: str, metadata: dict[str, Any]) -> str:
    """Write *metadata* on room ``r1`` the way *writer* does; what came of it."""
    if writer == "a channel's metadata_updates":
        kit.register_channel(_Updating("crm", metadata))
        await kit.attach_channel("r1", "crm")
        await _say(kit, "u-bob", "Bob", "first")
        planted = set(metadata) & set((await kit.get_room("r1")).metadata)
        return "written" if planted else "refused"
    if writer == "update_room_metadata":
        await kit.update_room_metadata("r1", metadata)
        planted = set(metadata) & set((await kit.get_room("r1")).metadata)
        return "written" if planted else "refused"
    outcome: list[str] = []

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="planter")
    async def planter(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if not outcome:
            outcome.append(await _plant(kit, "update_room_metadata", metadata))
        return HookResult.allow()

    await _say(kit, "u-bob", "Bob", "first")
    return outcome[0]


class TestOnlyTheRuntimeWritesTheRegister:
    """RMK-978: a register the room's metadata holds is merged into the
    store's at the next named turn, for a room stored before and a process
    from before. The key is the runtime's alone: a writer of the room's
    metadata who planted a register there would take another sender's name
    at rank 1, and the real sender would read ``Alice (2)``."""

    @pytest.mark.parametrize(
        "writer", ["update_room_metadata", "a hook", "a channel's metadata_updates"]
    )
    async def test_a_forged_register_takes_nobody_s_rank(self, writer: str) -> None:
        kit, _provider = await _kit(["a1", "a2", "a3"])
        forged = {AUTHOR_REGISTER: _forged_register("u-mallory", "Alice")}

        outcome = await _plant(kit, writer, forged)
        await _say(kit, "u-alice", "Alice", "hold the refund")
        await _say(kit, "u-mallory", "Alice", "release it")

        events = [e for e in await kit.store.list_events("r1") if e.source.channel_id == "sms1"]
        ranks = [
            (e.source.participant_id, e.metadata[AUTHOR]["rank"])
            for e in events
            if e.source.participant_id != "u-bob"
        ]
        assert (ranks, outcome) == ([("u-alice", 1), ("u-mallory", 2)], "refused")
        assert AUTHOR_REGISTER not in (await kit.get_room("r1")).metadata
        await kit.close()
