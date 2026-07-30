"""An attachment the SFU refuses — RFC §12.10.4 step 1.

The step is a MUST: attaching the channel to a room MUST call ``ensure_room()``.
It always did, but from an ``ON_CHANNEL_ATTACHED`` hook, and a lifecycle hook is
observation — its errors are logged and never raised, and the binding is already
written by the time it runs. An unreachable SFU therefore produced a room that
believed it was conferenced, participants holding credentials for a conference
that did not exist, and no sign of any of it until the first ``join_as_bot()``.

``ensure_room()`` is now the channel contract (``on_room_attached``), awaited by
the attach before anything has observed it. These tests are what that buys.
"""

from __future__ import annotations

import pytest

from roomkit import MockConferenceBackend, RoomKit
from roomkit.channels.conference import ConferenceChannel
from roomkit.models.context import RoomContext
from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.models.event import RoomEvent

ROOM = "room-1"


class SFUUnreachableError(RuntimeError):
    """What an SFU that cannot be reached raises through the backend.

    A timeout, a refused E2EE configuration and an SFU that is simply down all
    look the same from the channel: ``ensure_room()`` raises.
    """


async def _kit_with_unreachable_sfu(
    *, times: int | None = None
) -> tuple[RoomKit, ConferenceChannel, MockConferenceBackend]:
    backend = MockConferenceBackend()
    backend.fail("ensure_room", SFUUnreachableError("cannot reach the SFU"), times=times)
    channel = ConferenceChannel("conf", backend=backend)
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(ROOM)
    return kit, channel, backend


class TestRefusedAttach:
    async def test_the_attach_raises(self) -> None:
        kit, _, _ = await _kit_with_unreachable_sfu()

        with pytest.raises(SFUUnreachableError):
            await kit.attach_channel(ROOM, "conf")

    async def test_no_binding_survives(self) -> None:
        """The state this exists to prevent: a room bound to a conference that
        was never created."""
        kit, _, _ = await _kit_with_unreachable_sfu()

        with pytest.raises(SFUUnreachableError):
            await kit.attach_channel(ROOM, "conf")

        assert await kit.list_bindings(ROOM) == []

    async def test_the_channel_reports_no_conference(self) -> None:
        """RFC §17.7 asks that an integrator be able to ask at any time; the
        answer for a refused attach is that there is no meeting here."""
        kit, channel, _ = await _kit_with_unreachable_sfu()

        with pytest.raises(SFUUnreachableError):
            await kit.attach_channel(ROOM, "conf")

        assert channel.info()["rooms"] == {}

    async def test_no_one_was_told_it_attached(self) -> None:
        kit, _, _ = await _kit_with_unreachable_sfu()
        observed: list[str] = []

        @kit.hook(
            HookTrigger.ON_CHANNEL_ATTACHED,
            execution=HookExecution.ASYNC,
            name="observer",
        )
        async def _observe(event: RoomEvent, context: RoomContext) -> None:
            observed.append(event.room_id)

        with pytest.raises(SFUUnreachableError):
            await kit.attach_channel(ROOM, "conf")

        assert observed == []

    async def test_the_room_attaches_once_the_sfu_answers(self) -> None:
        """The refusal is the first attempt only; the second finds the SFU up."""
        kit, channel, backend = await _kit_with_unreachable_sfu(times=1)

        with pytest.raises(SFUUnreachableError):
            await kit.attach_channel(ROOM, "conf")

        await kit.attach_channel(ROOM, "conf")

        assert ROOM in backend.rooms
        assert ROOM in channel.info()["rooms"]


class TestRefusedReattach:
    """An attach over a conference that is already running, and refused.

    The channel is still attached when it refuses — the generation is bumped
    only once ``ensure_room()`` has returned — so the conference it holds is
    untouched: the bot is in the meeting, the collection is running. What the
    rollback used to take away was the room's binding, which is the only handle
    ``detach_channel()`` has on any of it.
    """

    async def _running_conference(
        self,
    ) -> tuple[RoomKit, ConferenceChannel, MockConferenceBackend]:
        backend = MockConferenceBackend()
        channel = ConferenceChannel("conf", backend=backend)
        kit = RoomKit()
        kit.register_channel(channel)
        await kit.create_room(ROOM)
        await kit.attach_channel(ROOM, "conf")
        await backend.simulate_participant_joined(ROOM, "p-alice")
        assert backend.bots != [], "the bot never joined, so there is nothing to strand"
        return kit, channel, backend

    async def test_the_conference_it_could_not_replace_is_still_reachable(self) -> None:
        """The defect: a bot in a meeting, a channel collecting from it, and no
        binding left for the detach that would have taken either out.
        """
        kit, channel, backend = await self._running_conference()
        backend.fail("ensure_room", SFUUnreachableError("cannot reach the SFU"))

        with pytest.raises(SFUUnreachableError):
            await kit.attach_channel(ROOM, "conf")

        assert [b.channel_id for b in await kit.list_bindings(ROOM)] == ["conf"]
        assert channel.info()["rooms"][ROOM]["bot_present"] is True

        assert await kit.detach_channel(ROOM, "conf") is True
        assert backend.bots == []
        assert channel.info()["rooms"] == {}

    async def test_the_running_conference_is_not_disturbed(self) -> None:
        """A refused attach is a no-op on the attachment it failed to replace,
        not a half-applied one: the generation the lanes and the credentials are
        written against has not moved.
        """
        kit, channel, backend = await self._running_conference()
        generation = channel._room(ROOM).generation
        backend.fail("ensure_room", SFUUnreachableError("cannot reach the SFU"))

        with pytest.raises(SFUUnreachableError):
            await kit.attach_channel(ROOM, "conf")

        assert channel._room(ROOM).generation == generation
        assert channel._room(ROOM).attached is True


class TestAttachOrder:
    async def test_the_conference_exists_before_the_attach_is_announced(self) -> None:
        """An ``ON_CHANNEL_ATTACHED`` handler that mints admission finds a
        conference to admit someone to.

        Async hooks of one trigger run concurrently, so for as long as
        ``ensure_room()`` was itself a hook there was no order between it and
        the integrator's — a handler calling ``mint_access()`` could address a
        conference that had not been created yet.
        """
        backend = MockConferenceBackend()
        channel = ConferenceChannel("conf", backend=backend)
        kit = RoomKit()
        kit.register_channel(channel)
        await kit.create_room(ROOM)
        rooms_when_observed: list[list[str]] = []

        @kit.hook(
            HookTrigger.ON_CHANNEL_ATTACHED,
            execution=HookExecution.ASYNC,
            name="observer",
        )
        async def _observe(event: RoomEvent, context: RoomContext) -> None:
            rooms_when_observed.append(sorted(backend.rooms))

        await kit.attach_channel(ROOM, "conf")

        assert rooms_when_observed == [[ROOM]]

    async def test_the_conference_is_over_before_the_detach_is_announced(self) -> None:
        """The symmetric guarantee: the channel has let go before a handler
        runs, rather than alongside it."""
        backend = MockConferenceBackend()
        channel = ConferenceChannel("conf", backend=backend, close_room_on_detach=True)
        kit = RoomKit()
        kit.register_channel(channel)
        await kit.create_room(ROOM)
        await kit.attach_channel(ROOM, "conf")
        await backend.simulate_participant_joined(ROOM, "p-alice")
        rooms_when_observed: list[list[str]] = []

        @kit.hook(
            HookTrigger.ON_CHANNEL_DETACHED,
            execution=HookExecution.ASYNC,
            name="observer",
        )
        async def _observe(event: RoomEvent, context: RoomContext) -> None:
            rooms_when_observed.append(sorted(backend.rooms))

        await kit.detach_channel(ROOM, "conf")

        assert rooms_when_observed == [[]]
