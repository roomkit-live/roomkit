"""Pure transport mode: a conference channel with nothing to consume or say.

RFC §12.10.4 step 1 (RMK-75): the join exists for the intelligence. A channel
configured without stt, tts or recording has no first need — the mint, the
arrival and the attach's occupancy probe start no join, and the probe is not
made at all, the join being the only consequence it can have. The channel
remains the room's admission gate and roster; what such a deployment gives up
is the event bridge the bot's connection would have been (RFC §12.10.3).

The mode is a contract, and this file is its name: RoomKit can run a purely
human meeting — credentials minted, arrivals recorded, `info()` answering —
without ever putting a participant of its own in it.
"""

from __future__ import annotations

import asyncio

import pytest

from roomkit import (
    ConferenceParticipant,
    ConferenceRecordingConfig,
    MockConferenceBackend,
    RoomKit,
)
from roomkit.channels.conference import ConferenceChannel
from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.recorder.mock import MockMediaRecorder
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

ROOM = "room-1"


async def _pure_transport() -> tuple[RoomKit, ConferenceChannel, MockConferenceBackend]:
    """A conference channel with no stt, no tts, no recording — transport only."""
    backend = MockConferenceBackend()
    channel = ConferenceChannel("conf", backend=backend)
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")
    return kit, channel, backend


async def _room_settled(channel: ConferenceChannel) -> None:
    """Wait out the room's background work — any spawned trigger among it."""
    room = channel._room(ROOM)
    while room.tasks:
        await asyncio.wait(list(room.tasks), timeout=5.0)


class TestNoJoinWithoutANeed:
    async def test_a_mint_starts_no_join(self) -> None:
        """The credential is the participant's; the session the join would
        open has nothing to consume and nothing to say.
        """
        kit, channel, backend = await _pure_transport()
        await kit.ensure_participant(ROOM, "conf", "p-alice")

        await channel.mint_access(ROOM, "p-alice")
        await _room_settled(channel)

        assert backend.bots == []
        assert not [c for c in backend.calls if c.method == "join_as_bot"]

    async def test_an_arrival_is_recorded_without_a_join(self) -> None:
        """Recording the arrival is the unconditional MUST (RFC §12.10.4
        step 2); a join would serve nothing here, so none is started.
        """
        kit, channel, backend = await _pure_transport()
        joined: list[str] = []

        @kit.hook(HookTrigger.ON_CONFERENCE_PARTICIPANT_JOINED, execution=HookExecution.ASYNC)
        async def _saw(event, ctx) -> None:  # type: ignore[no-untyped-def]
            joined.append(event.content.data["participant_id"])

        await backend.simulate_participant_joined(ROOM, "p-alice")
        await _room_settled(channel)

        participants = await kit.store.list_participants(ROOM)
        assert [p.id for p in participants] == ["p-alice"]
        assert joined == ["p-alice"]
        assert backend.bots == []
        assert not [c for c in backend.calls if c.method == "join_as_bot"]

    async def test_an_attach_over_a_live_conference_makes_no_probe(self) -> None:
        """The probe's only consequence is the join, so there is nothing to
        ask the control plane — an occupied conference stays unjoined and
        uncounted (the RFC §17.7 window is the mode's stated price).
        """
        backend = MockConferenceBackend()
        backend.participants[ROOM] = {"p-alice": ConferenceParticipant(participant_id="p-alice")}
        channel = ConferenceChannel("conf", backend=backend)
        kit = RoomKit()
        kit.register_channel(channel)
        await kit.create_room(ROOM)

        await kit.attach_channel(ROOM, "conf")
        await _room_settled(channel)

        assert not [c for c in backend.calls if c.method == "list_participants"]
        assert backend.bots == []

    async def test_no_conference_is_ever_announced(self) -> None:
        """No session, no ``conference_started``, no ON_SESSION_STARTED: a
        conference that was never joined must not be narrated as one.
        """
        kit, channel, backend = await _pure_transport()
        announced: list[str] = []

        @kit.on("conference_started")
        async def _started(event: object) -> None:
            announced.append("started")

        @kit.hook(HookTrigger.ON_SESSION_STARTED, execution=HookExecution.ASYNC)
        async def _session(event, ctx) -> None:  # type: ignore[no-untyped-def]
            announced.append("session")

        await kit.ensure_participant(ROOM, "conf", "p-alice")
        await channel.mint_access(ROOM, "p-alice")
        await backend.simulate_participant_joined(ROOM, "p-alice")
        await _room_settled(channel)

        assert announced == []

    async def test_info_reports_an_unattended_conference(self) -> None:
        """The §17.7 disclosure surface answers plainly: nothing is in the
        meeting and nothing is being collected.
        """
        kit, channel, backend = await _pure_transport()
        await kit.ensure_participant(ROOM, "conf", "p-alice")
        await channel.mint_access(ROOM, "p-alice")
        await backend.simulate_participant_joined(ROOM, "p-alice")
        await _room_settled(channel)

        info = channel.info()
        assert info["stt_configured"] is False
        assert info["recording_configured"] is False
        room = info["rooms"][ROOM]
        assert room["bot_present"] is False
        assert room["bot_session_id"] is None
        assert room["stt_active"] is False
        assert room["recording_active"] is False
        assert room["active_lanes"] == 0

    async def test_detach_and_close_owe_the_conference_nothing(self) -> None:
        """No session was ever opened, so nothing leaves and nothing fails."""
        kit, channel, backend = await _pure_transport()
        await kit.ensure_participant(ROOM, "conf", "p-alice")
        await channel.mint_access(ROOM, "p-alice")
        await backend.simulate_participant_joined(ROOM, "p-alice")
        await _room_settled(channel)

        await kit.detach_channel(ROOM, "conf")
        await kit.close()

        assert not [c for c in backend.calls if c.method == "leave"]
        assert backend.bots == []


class TestAnyNeedRestoresTheTriggers:
    """One consumer or one voice is a need, and the lazy join is unchanged."""

    @staticmethod
    def _needs() -> dict[str, dict[str, object]]:
        return {
            "stt": {"stt": MockSTTProvider()},
            "tts": {"tts": MockTTSProvider()},
            "recording": {
                "recorder": MockMediaRecorder(),
                "recording": ConferenceRecordingConfig(),
            },
        }

    @pytest.mark.parametrize("need", ["stt", "tts", "recording"])
    async def test_a_mint_brings_the_bot_in(self, need: str) -> None:
        backend = MockConferenceBackend()
        channel = ConferenceChannel("conf", backend=backend, **self._needs()[need])  # type: ignore[arg-type]
        kit = RoomKit()
        kit.register_channel(channel)
        await kit.create_room(ROOM)
        await kit.attach_channel(ROOM, "conf")
        await kit.ensure_participant(ROOM, "conf", "p-alice")

        await channel.mint_access(ROOM, "p-alice")
        await _room_settled(channel)

        assert len(backend.bots) == 1

    @pytest.mark.parametrize("need", ["stt", "tts", "recording"])
    async def test_the_attach_probe_is_made(self, need: str) -> None:
        """A channel that could listen or speak still asks who is already
        there (RFC §12.10.4 step 1): the restart-over-a-live-meeting trigger
        is a need-holder's trigger too.
        """
        backend = MockConferenceBackend()
        backend.participants[ROOM] = {"p-alice": ConferenceParticipant(participant_id="p-alice")}
        channel = ConferenceChannel("conf", backend=backend, **self._needs()[need])  # type: ignore[arg-type]
        kit = RoomKit()
        kit.register_channel(channel)
        await kit.create_room(ROOM)

        await kit.attach_channel(ROOM, "conf")
        await _room_settled(channel)

        assert [c.method for c in backend.calls if c.method == "list_participants"]
        assert len(backend.bots) == 1
