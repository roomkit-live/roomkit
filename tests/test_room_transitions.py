"""An integrator's room transition (RFC §5.1).

One on a room already in the status it leads to does nothing: no write, no
hook, no framework event. One on a room in a status it does not leave from
fails with nothing written. No call takes a room out of ARCHIVED.
"""

from __future__ import annotations

import pytest

from roomkit import (
    FrameworkEvent,
    HookExecution,
    HookTrigger,
    RoomContext,
    RoomEvent,
    RoomKit,
    RoomStatus,
    TextContent,
    TransitionRefusedError,
)
from tests.test_framework import SimpleChannel


async def _room_in(status: RoomStatus) -> tuple[RoomKit, list[str]]:
    """A kit holding room ``r1`` in *status*, and what its transitions report
    from then on: their hooks and their framework events."""
    kit = RoomKit()
    await kit.create_room(room_id="r1")
    if status is RoomStatus.PAUSED:
        await kit.pause_room("r1")
    elif status is RoomStatus.CLOSED:
        await kit.close_room("r1")
    elif status is RoomStatus.ARCHIVED:
        await kit.archive_room("r1")
    await kit.hook_engine.drain_observers()
    heard: list[str] = []

    for trigger in (HookTrigger.ON_ROOM_PAUSED, HookTrigger.ON_ROOM_CLOSED):

        @kit.hook(trigger, execution=HookExecution.ASYNC)
        async def on_change(event: RoomEvent, ctx: RoomContext, trigger: str = trigger) -> None:
            heard.append(trigger)

    for name in ("room_paused", "room_closed", "room_archived"):

        @kit.on(name)
        async def on_event(event: FrameworkEvent) -> None:
            heard.append(event.type)

    return kit, heard


@pytest.mark.parametrize(
    ("operation", "status"),
    [
        ("close_room", RoomStatus.CLOSED),
        ("archive_room", RoomStatus.ARCHIVED),
        ("pause_room", RoomStatus.PAUSED),
        ("resume_room", RoomStatus.ACTIVE),
    ],
)
async def test_a_transition_already_done_does_nothing(operation: str, status: RoomStatus) -> None:
    kit, heard = await _room_in(status)
    before = await kit.store.get_room("r1")

    room = await getattr(kit, operation)("r1")
    await kit.hook_engine.drain_observers()

    assert room == before
    assert await kit.store.get_room("r1") == before
    assert heard == []
    await kit.close()


@pytest.mark.parametrize(
    ("operation", "status"),
    [
        ("close_room", RoomStatus.ARCHIVED),
        ("pause_room", RoomStatus.CLOSED),
        ("pause_room", RoomStatus.ARCHIVED),
        ("resume_room", RoomStatus.CLOSED),
        ("resume_room", RoomStatus.ARCHIVED),
        ("reopen_room", RoomStatus.ACTIVE),
        ("reopen_room", RoomStatus.PAUSED),
        ("reopen_room", RoomStatus.ARCHIVED),
    ],
)
async def test_a_transition_from_a_status_it_does_not_leave_fails(
    operation: str, status: RoomStatus
) -> None:
    kit, heard = await _room_in(status)
    before = await kit.store.get_room("r1")

    with pytest.raises(TransitionRefusedError) as refused:
        await getattr(kit, operation)("r1")

    assert (refused.value.room_id, refused.value.status) == ("r1", status)
    assert await kit.store.get_room("r1") == before
    assert heard == []
    await kit.close()


@pytest.mark.parametrize(
    ("operation", "status", "after", "reported"),
    [
        ("pause_room", RoomStatus.ACTIVE, RoomStatus.PAUSED, [HookTrigger.ON_ROOM_PAUSED]),
        ("resume_room", RoomStatus.PAUSED, RoomStatus.ACTIVE, []),
        ("reopen_room", RoomStatus.CLOSED, RoomStatus.ACTIVE, []),
    ],
)
async def test_a_pause_a_resume_and_a_reopen_move_the_room_as_roomkit_rs_does(
    operation: str, status: RoomStatus, after: RoomStatus, reported: list[str]
) -> None:
    """RFC §5.1, RMK-1021 item 22: the integrator's pause fires
    ``ON_ROOM_PAUSED`` and emits no ``room_paused``, which reports the
    timer's (§8.2); a resume and a reopen report nothing and restart the
    inactivity timer; the room accepts events after each. RoomKit had none
    of the three calls."""
    kit, heard = await _room_in(status)
    before = await kit.store.get_room("r1")
    assert before is not None

    room = await getattr(kit, operation)("r1")
    await kit.hook_engine.drain_observers()

    assert (room.status, heard) == (after, reported)
    assert (room.paused_at is not None) is (after is RoomStatus.PAUSED)
    if operation != "pause_room":
        assert room.timers.last_activity_at is not None
        assert room.timers.last_activity_at > before.updated_at
        assert room.closed_at is None
    kit.register_channel(SimpleChannel("sms"))
    await kit.attach_channel("r1", "sms")
    said = await kit.send_event("r1", "sms", TextContent(body="Back."))
    assert said.blocked_by is None
    await kit.close()
