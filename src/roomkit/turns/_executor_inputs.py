"""One turn's human-input requests on its executor (RFC §10.6.5).

A request the channel's handler opens is sent to the owner as an
``input_request`` before ``ON_USER_INPUT_REQUIRED`` fires; the owner's
``input_answer`` settles it here while it is still open, and ``input_taken``
says whether it did. A request settled here (an answer the channel's handler
received, a hook's block, its bound, a release) is sent as an
``input_closed``, and the owner answers it no more. When the turn ends, its
requests still open are withdrawn, but a detached one, its creator's to
release, which the turn only lets go of; nothing more is sent.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING

from roomkit.core.exceptions import TurnLinkError
from roomkit.turns.wire import InputClosed, InputRequest, InputTaken

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from roomkit.models.pending_input import PendingInput
    from roomkit.tools._turn_inputs import InputSettler
    from roomkit.turns.wire import Frame, InputAnswer

logger = logging.getLogger("roomkit.turns")


class ExecutorInputs:
    """The open human-input requests of one turn, and their way to its owner."""

    def __init__(self, send: Callable[[Frame], Awaitable[int]]) -> None:
        self._send = send
        self._loop = asyncio.get_running_loop()
        self._open: dict[str, InputSettler] = {}
        self._running = True
        self._sending: set[asyncio.Task[None]] = set()

    @property
    def open(self) -> bool:
        return self._running

    def close(self) -> None:
        """The turn ended: its requests still open are withdrawn, on both
        sides (RFC §10.6.5), but a detached one, which its creator releases,
        and later ones are the handler's alone."""
        self._running = False
        still_open, self._open = self._open, {}
        for pending_id, settle in still_open.items():
            settle.withdraw(pending_id)

    async def opened(self, pending: PendingInput, settle: InputSettler) -> None:
        if not self._running:
            return
        self._open[pending.pending_id] = settle
        await self._send(
            InputRequest(
                request_id=pending.pending_id,
                tool_name=pending.tool_name,
                arguments=pending.arguments,
                room_id=pending.room_id,
                channel_id=pending.channel_id,
                tool_call_id=pending.tool_call_id,
                actor_id=pending.actor_id,
                created_at=pending.created_at,
            )
        )

    def closed(self, pending_id: str, reason: str) -> None:
        if not self._running:
            return
        # Sent from the turn's own loop, whatever loop or thread settled it;
        # a closed loop has no owner left to tell.
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._send_closed, pending_id, reason)

    async def answer(self, message: InputAnswer) -> None:
        """The owner's outcome for a request: it settles here if still open,
        and the owner learns whether it did."""
        settle = self._open.pop(message.request_id, None)
        taken = False
        if settle is not None:
            if message.outcome == "answered":
                taken = settle.settle(message.request_id, value=message.value or "", reason=None)
            else:
                taken = settle.settle(message.request_id, value=None, reason=message.reason or "")
        await self._send(InputTaken(request_id=message.request_id, taken=taken))

    def _send_closed(self, pending_id: str, reason: str) -> None:
        if not self._running or self._open.pop(pending_id, None) is None:
            return
        task = self._loop.create_task(
            self._send_quietly(InputClosed(request_id=pending_id, reason=reason))
        )
        self._sending.add(task)
        task.add_done_callback(self._sending.discard)

    async def _send_quietly(self, frame: Frame) -> None:
        try:
            await self._send(frame)
        except TurnLinkError:
            pass
        except Exception:
            logger.warning("A human-input close could not reach the room owner", exc_info=True)
