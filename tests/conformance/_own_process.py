"""An executor run as in a process of its own (roomkit-specs
``conformance/README.md``, an executor's ``own_process``).

It runs on a thread and an event loop of its own, over a node of its own of
the Redis turn backend, so that a turn busy with its CPU there stalls nothing
beside it: the executor's beat comes from the backend's thread, which the
turn's work cannot stall (RFC §10.6.8). What the run's gates and records share
stays on the run's own loop: the channels it hosts are scripted channels whose
reactions hold no gate and make no action other than a host verb.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Callable, Coroutine, Iterable, Mapping
from typing import Any

from roomkit.turns import TurnExecutor, TurnTimeouts
from roomkit.turns.redis import RedisTurnBackend

# What a reaction of a channel such an executor hosts may hold: nothing that
# waits on the run's loop.
_REACTION = {"when", "repeat", "delay_ms", "holds_cpu_ms", "do", "respond", "output", "fail"}


def apart_problems(spec: Mapping[str, Any], channels: Iterable[Mapping[str, Any]]) -> list[str]:
    """Why the executor *spec* cannot run in a process of its own: each
    channel it hosts that is no scripted channel, or whose reactions hold a
    gate or make an action other than a host verb (README, ``own_process``)."""
    problems = []
    for channel in channels:
        if not channel.get("declared") or not _hosts(spec, channel["id"]):
            continue
        where = f"executor {spec['id']!r}, channel {channel['id']!r}"
        if "model" in channel or "acp" in channel:
            problems.append(f"{where}: a process of its own hosts scripted channels alone")
        for reaction in channel.get("on_event", []):
            extra = set(reaction) - _REACTION
            actions = reaction.get("do", [])
            calls = [a["call"] for a in actions if not a["call"].startswith("host.")]
            if extra or calls:
                problems.append(f"{where}: no gate nor action but a host verb: {extra or calls}")
    return problems


def _hosts(spec: Mapping[str, Any], channel_id: str) -> bool:
    """Whether *spec* declares *channel_id*, by name or by a prefix ending in ``*``."""
    return any(
        host == channel_id or (host.endswith("*") and channel_id.startswith(host[:-1]))
        for host in spec.get("hosts", [])
    )


class Apart:
    """One executor served on a thread, a loop and a backend node of its own."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        thread: threading.Thread,
        executor: TurnExecutor,
        backend: RedisTurnBackend,
    ) -> None:
        self._loop = loop
        self._thread = thread
        self._executor = executor
        self._backend = backend

    @classmethod
    async def start(
        cls, make: Callable[[], TurnExecutor], key_prefix: str, bounds: TurnTimeouts | None
    ) -> Apart:
        """Serves the executor *make* builds there, under *key_prefix*, where
        the kit's backend meets it."""
        loop = asyncio.new_event_loop()
        # A daemon: a run that fails before its close leaves no thread
        # keeping the test process alive.
        thread = threading.Thread(
            target=loop.run_forever, name="conformance-executor", daemon=True
        )
        thread.start()

        async def serve() -> tuple[TurnExecutor, RedisTurnBackend]:
            executor = make()
            backend = RedisTurnBackend(
                os.environ["REDIS_URL"], key_prefix=key_prefix, timeouts=bounds
            )
            await backend.serve(executor)
            return executor, backend

        try:
            executor, backend = await _on(loop, serve())
        except BaseException:
            await _stop(loop, thread)
            raise
        return cls(loop, thread, executor, backend)

    async def close(self) -> None:
        """Closes the executor and its backend there, then stops its loop."""

        async def shut() -> None:
            try:
                await self._executor.close()
            finally:
                await self._backend.close()

        try:
            await _on(self._loop, shut())
        finally:
            await _stop(self._loop, self._thread)


async def _on[T](loop: asyncio.AbstractEventLoop, work: Coroutine[Any, Any, T]) -> T:
    """Runs *work* on *loop*, another thread's, and waits for it here."""
    return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(work, loop))


async def _stop(loop: asyncio.AbstractEventLoop, thread: threading.Thread) -> None:
    loop.call_soon_threadsafe(loop.stop)
    await asyncio.to_thread(thread.join)
    loop.close()
