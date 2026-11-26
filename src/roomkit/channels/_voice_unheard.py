"""The routed voice turn whose response the user has not heard yet (RFC §12.3.12).

A turn is routed, and until the first audio of its response reaches the
transport, nobody has heard it. A user who resumes speaking in that window is
still on their sentence: the response is held while they speak, then either
superseded by what they added or released to play as it would have.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field

from roomkit import _clock
from roomkit.models.delivery import DeliveryHandle


@dataclass
class _UnheardTurn:
    handle: DeliveryHandle
    held_since: float | None = None
    """Monotonic onset of the speech holding the response, None when not held."""
    resumed_ms: float = 0.0
    """How long the speech that held the response lasted, once it ended."""
    heard: bool = False
    superseded: bool = False
    released: asyncio.Event = field(default_factory=asyncio.Event)


class UnheardTurns:
    """Per session, the routed turn not yet heard, held while the user resumes.

    ``hold`` and ``note_speech_end`` are called from the VAD, which may run on
    the audio thread; everything else runs on the event loop, which is the only
    place an ``asyncio.Event`` is set.
    """

    def __init__(self) -> None:
        self._turns: dict[str, _UnheardTurn] = {}
        self._lock = threading.Lock()

    def register(
        self, session_id: str, handle: DeliveryHandle, *, speaking_since: float | None
    ) -> None:
        """Track a turn just routed; speech already under way (its monotonic onset) holds it."""
        turn = _UnheardTurn(handle=handle)
        if speaking_since is not None:
            turn.held_since = speaking_since
        else:
            turn.released.set()
        with self._lock:
            self._turns[session_id] = turn

    def discard(self, session_id: str, handle: DeliveryHandle) -> None:
        """Forget the session's turn once its delivery ended, if it is still *handle*'s."""
        with self._lock:
            turn = self._turns.get(session_id)
            if turn is not None and turn.handle is handle:
                del self._turns[session_id]
        if turn is not None and turn.handle is handle:
            turn.released.set()

    def hold(self, session_id: str) -> bool:
        """Speech started: hold the session's unheard response. False when none is unheard."""
        with self._lock:
            turn = self._turns.get(session_id)
            if turn is None or turn.heard or turn.superseded:
                return False
            if turn.held_since is None:
                # A fresh event: clearing one from this thread is not safe.
                turn.released = asyncio.Event()
            # Each segment is measured on its own: short sounds and the pauses
            # between them never add up to a continuation.
            turn.held_since = _clock.monotonic()
            turn.resumed_ms = 0.0
            return True

    def note_speech_end(self, session_id: str) -> None:
        """Speech ended: record how long the speech holding the response lasted."""
        with self._lock:
            turn = self._turns.get(session_id)
            if turn is not None and turn.held_since is not None:
                turn.resumed_ms = (_clock.monotonic() - turn.held_since) * 1000

    async def wait_to_play(self, session_id: str) -> bool:
        """Before the response's first audio: wait while it is held, then mark it heard.

        False when the turn was superseded meanwhile: its audio must not go out.
        """
        while True:
            with self._lock:
                turn = self._turns.get(session_id)
                if turn is None:
                    return True
                if turn.superseded:
                    return False
                if turn.held_since is None:
                    turn.heard = True
                    return True
                released = turn.released
            await released.wait()

    def take_superseded(self, session_id: str, min_speech_ms: float) -> DeliveryHandle | None:
        """The held turn the ended speech supersedes, removed from tracking; else None."""
        with self._lock:
            turn = self._turns.get(session_id)
            if turn is None or turn.held_since is None or turn.resumed_ms < min_speech_ms:
                return None
            # Kept until its delivery ends, so its gate reads it superseded.
            turn.superseded = True
        turn.released.set()
        return turn.handle

    def release(self, session_id: str) -> None:
        """The speech did not supersede the turn: its held response plays."""
        with self._lock:
            turn = self._turns.get(session_id)
            if turn is None or turn.held_since is None or turn.superseded:
                return
            turn.held_since = None
            released = turn.released
        released.set()
