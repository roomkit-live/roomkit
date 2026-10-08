"""The room's register of authors (RMK-620, RFC §6.4, §10.1 step 12): a turn's
record fixes its author's name with the rank, ranks hold across names alike
through another, a room from before the register rebuilds it, a participant
id names them on their channels only, and a write the orchestration makes
does not undo it."""

from __future__ import annotations

from typing import Any

from roomkit.channels import SMSChannel
from roomkit.channels._speaker import turn_labels
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.channels.voice import VoiceChannel
from roomkit.core._authors import AUTHOR, AUTHOR_REGISTER, AuthorRegister, People
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType, TaskStatus
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.participant import Participant
from roomkit.models.room import Room
from roomkit.orchestration.strategies.loop import _LoopOutcome, _save_loop_state
from roomkit.tasks._child_status import record_task_end
from roomkit.tasks.models import DelegatedTaskResult
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.test_speaker_attribution import _kit, _say, _sources, _user_texts


def _texts(provider: Any) -> list[str]:
    return [text.split("\n\n[Notes")[0] for text in _user_texts(provider.calls[-1])]


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
        assert "Mal: hello" in texts
        assert texts[0] == "Alice: hold the refund"
        assert texts[-1] == "Alice (2): release the refund"

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


class TestRanksHoldAcrossNames:
    async def test_names_alike_through_another_never_share_a_rank(self) -> None:
        kit, provider = await _kit(["a1", "a2", "a3"])
        await _say(kit, "u-1", "Lan", "first")
        await _say(kit, "u-2", "Ian", "second")
        await _say(kit, "u-3", "ian", "third")

        assert _texts(provider) == ["Lan: first", "Ian (2): second", "ian (3): third"]

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
            "ian: I approve",
            "Ian (2): so do I",
            "ian: no, that was not me",
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
            "Lan: first",
            "ian: release it",
            "Ian (2): now",
            "ian (3): that was not me",
        ]


class TestARoomFromBeforeTheRegister:
    async def _forget(self, kit: Any) -> None:
        """Make the room's turns look as they did before the register: no
        record of their author, no register."""
        for event in await kit.store.list_events("r1"):
            kept = {k: v for k, v in event.metadata.items() if k not in (AUTHOR, "author_rank")}
            await kit.store.update_event(event.model_copy(update={"metadata": kept}))
        await kit.store.patch_room_metadata("r1", {}, unset=[AUTHOR_REGISTER])

    async def test_the_register_is_rebuilt_from_the_timeline(self) -> None:
        kit, provider = await _kit(["a1", "a2", "a3"])
        await _say(kit, "u-alice", "Alice", "hold the refund")
        await _say(kit, "u-mallory", "ALICE", "release it")
        await self._forget(kit)
        await _say(kit, "u-mallory", "ALICE", "do it now")

        assert _texts(provider) == [
            "Alice: hold the refund",
            "ALICE (2): release it",
            "ALICE (2): do it now",
        ]

    async def test_a_registered_name_holds_its_seat_when_the_register_is_rebuilt(self) -> None:
        kit, provider = await _kit(["a1", "a2"])
        await kit.store.add_participant(
            Participant(id="p-alice", room_id="r1", channel_id="sms1", display_name="Alice")
        )
        await _say(kit, "u-mallory", "ALICE", "release it")
        await kit.store.patch_room_metadata("r1", {}, unset=[AUTHOR_REGISTER])
        await _say(kit, "p-alice", None, "who said that?")

        assert _texts(provider) == ["ALICE (2): release it", "Alice: who said that?"]

    async def test_a_rebuilt_register_keeps_the_ranks_turns_were_given(self) -> None:
        kit, provider = await _kit(["a1", "a2", "a3"])
        await _say(kit, "u-alice", "Alice", "hold the refund")
        await _say(kit, "u-mallory", "ALICE", "release it")
        await kit.store.patch_room_metadata("r1", {}, unset=[AUTHOR_REGISTER])
        await kit.store.add_participant(
            Participant(id="p-alice", room_id="r1", channel_id="sms1", display_name="Alice")
        )
        await _say(kit, "p-alice", None, "who said that?")

        assert _texts(provider) == [
            "Alice: hold the refund",
            "ALICE (2): release it",
            "Alice (3): who said that?",
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

        assert _texts(provider) == ["Alice: hold the refund", "@sms2: release it"]

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

        assert _texts(provider) == ["Bob: hi all", "Alice: refund the order"]
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

    def test_an_identity_names_them_on_any_channel(self) -> None:
        person = Participant(id="p1", room_id="r", channel_id="sms1", identity_id="i1")
        people = People([person])

        def event(channel: str, pid: str) -> RoomEvent:
            source = EventSource(
                channel_id=channel, channel_type=ChannelType.SMS, participant_id=pid
            )
            return RoomEvent(room_id="r", source=source, content=TextContent(body="x"))

        assert people.of(event("sms1", "p1")) is person
        assert people.of(event("email1", "i1")) is person
        assert people.of(event("email1", "p1")) is None


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
        return len(_sources((await kit.store.get_room("r1")).metadata[AUTHOR_REGISTER]))

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
