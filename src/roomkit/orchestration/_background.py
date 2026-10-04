"""A strategy's background run: work a tool call started, its outcome handed
back to whoever made the call (RFC §19.7.3, §19.7.4, §23.3 step 8).

One sequence for a supervisor's background workers and an asynchronous Loop:
run the work, free the room, hand the outcome back to the channel whose call
started it (in the session that made the call, on a realtime voice channel),
then post the run's one terminal entry on the status bus. The kit holds every
run until it ends: ``close()`` cancels it, and a run cancelled frees its room
and posts its terminal entry, failed, with nothing handed back.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from roomkit.channels._realtime_context import get_current_voice_session
from roomkit.core._fallback import FALLBACK_FAILED
from roomkit.core.exceptions import RoomKitError
from roomkit.core.task_utils import check_open, log_task_exception
from roomkit.orchestration.status_bus import StatusLevel
from roomkit.tasks.handback import hand_back, not_handed_back
from roomkit.tools.context import _current_turn_chain_depth, current_tool_call

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.orchestration.background")


def calling_channel_id() -> str:
    """The channel whose tool call is running: who a background run it starts
    tells (the voice channel whose session made the call)."""
    call = current_tool_call()
    return call.channel_id if call is not None else ""


def background_failure_text(what: str) -> str:
    """What a run's caller reads when the run raised: the framework's words,
    never the error's message (RFC §9.3), and nothing to fence."""
    return f"[Your background {what} failed: {FALLBACK_FAILED} Tell the user.]"


@dataclass(frozen=True)
class BackgroundRun[T]:
    """One background run of a strategy, and how its outcome reads.

    Attributes:
        room_id: The room of the call that started the run.
        notify: The channel whose call started it: who is told.
        work: The run itself; it raises when it fails.
        told: What *notify* reads of the outcome: the work's result, or
            ``None`` when it raised (never the error's message, RFC §9.3).
        ended: The terminal entry of work that returned, once handed back.
        post: Posts the run's terminal entry on the status bus.
        release: Frees the room, told whether the work returned: the model's
            turn on the outcome may start a new run.
    """

    room_id: str
    notify: str
    work: Callable[[], Awaitable[T]]
    told: Callable[[T | None], str]
    ended: Callable[[T], tuple[StatusLevel, str]]
    post: Callable[[StatusLevel, str], None]
    release: Callable[[bool], None]


def start_background_run(kit: RoomKit, run: Coroutine[Any, Any, None]) -> None:
    """Start a strategy's background *run* as a task *kit* holds until it
    ends, so ``close()`` cancels it (RFC §19.7.3, §19.7.4).

    Raises:
        RoomKitError: *kit* is closing (:func:`check_open`).
    """
    try:
        check_open(kit)
    except RoomKitError:
        run.close()
        raise
    task = asyncio.create_task(run)
    runs = kit._background_runs
    runs.add(task)
    task.add_done_callback(runs.discard)
    task.add_done_callback(log_task_exception)


async def run_in_background[T](kit: RoomKit, run: BackgroundRun[T]) -> None:
    """Run *run*, free its room, hand its outcome back, post its terminal entry.

    Started as a task by the tool call that asked for the run, so the context
    it copied is that call's (RFC §21.4): the outcome continues the chain of
    the turn that made it (§23.3), and a realtime voice channel is told in
    the session that made the call.
    """
    chain_depth = _current_turn_chain_depth()
    session = get_current_voice_session()
    outcome = await _work(run)
    text = run.told(outcome)
    try:
        delivered = await hand_back(
            kit,
            run.room_id,
            run.notify,
            text,
            chain_depth,
            session_id=session.id if session is not None else None,
        )
    except asyncio.CancelledError:
        _post_unless_failed(run, outcome, "cancelled")
        raise
    except Exception as exc:
        logger.error("Handing back a background run in room %s failed", run.room_id, exc_info=exc)
        _post_unless_failed(run, outcome, str(exc))
        return
    missed = not_handed_back(delivered)
    if missed is not None:
        _post_unless_failed(run, outcome, missed)
    elif outcome is not None:
        run.post(*run.ended(outcome))


async def _work[T](run: BackgroundRun[T]) -> T | None:
    """The run's result, or ``None`` when it raised, which is logged and
    posted with its message (for the logs and the status bus, never a model);
    the room is freed either way, a cancellation included."""
    outcome: T | None = None
    try:
        outcome = await run.work()
    except asyncio.CancelledError:
        run.post(StatusLevel.FAILED, "cancelled")
        raise
    except Exception as exc:
        logger.exception("Background run in room %s failed", run.room_id)
        run.post(StatusLevel.FAILED, str(exc))
    finally:
        run.release(outcome is not None)
    return outcome


def _post_unless_failed[T](run: BackgroundRun[T], outcome: T | None, detail: str) -> None:
    """Post a run whose outcome was not handed back as failed, unless its
    work already failed and was posted so."""
    if outcome is not None:
        run.post(StatusLevel.FAILED, detail)
