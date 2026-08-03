"""How an ACP connection reaches its agent.

:class:`ACPChannel` speaks the Agent Client Protocol; a transport decides how
those bytes travel. :class:`StdioACPTransport` — the default, and the only shape
the channel supported until now — spawns the agent as a local subprocess and
talks over its stdio. An agent running somewhere else (another machine, behind a
relay) needs the same protocol over a different pipe, which is what the ABC is
for.

The split is deliberately narrow: a transport opens the pipe, says whether it is
still alive, and tears it down. Everything the *protocol* does — ``initialize``,
version negotiation, ``authenticate``, sessions, prompts, permissions — stays in
the channel, so a new transport inherits all of it rather than reimplementing it.

Note the word is overloaded in RoomKit: a :class:`~roomkit.channels.transport.
TransportChannel` is a channel category (SMS, email — where messages reach
humans). *This* transport is the one the realtime voice layer also means, the
pipe to a remote peer.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from roomkit.channels._acp_client import _absolute_path, _load_sdk

logger = logging.getLogger("roomkit.channels.acp")

__all__ = ["ACPTransport", "StdioACPTransport"]


def _resolve_spawn_env(
    inherit_env: tuple[str, ...],
    env: Mapping[str, str] | None,
    environ: Mapping[str, str],
) -> dict[str, str] | None:
    """Build the env mapping handed to the ACP SDK's restricted spawn.

    ``inherit_env`` names are read from *environ* at spawn time (so a
    reconnect picks up a rotated value, e.g. a new SSH agent socket); unset
    names are skipped. Explicit ``env`` entries override inherited ones.
    Returns ``None`` when there is nothing to add — the SDK then uses its
    trimmed default environment unchanged.
    """
    merged = {name: environ[name] for name in inherit_env if name in environ}
    if env:
        merged.update(env)
    return merged or None


class ACPTransport(ABC):
    """The pipe an :class:`ACPChannel` speaks ACP over.

    Implement this to reach an agent the channel cannot spawn itself — one
    running on another machine, or behind a relay that carries its stdio.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Transport name, reported by ``ACPChannel.info["transport"]``."""
        ...

    @abstractmethod
    async def open(self, client: Any, *, queue: Any) -> Any:
        """Return a live ACP ``ClientSideConnection`` talking to the agent.

        *client* is the object the agent calls back into (permission
        requests, session updates) and *queue* the SDK message queue the
        channel drains notifications through; both are handed straight to
        the SDK connection. Raising is fine — the channel surfaces the
        failure and does not treat the transport as connected.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Undo what :meth:`open` set up. Must not raise.

        Called on teardown *and* on a failed handshake, so it has to
        tolerate being called when ``open`` never completed.
        """
        ...

    def is_alive(self) -> bool:
        """Whether the connection from the last :meth:`open` is still usable.

        The channel reconnects — and drops every session — when this turns
        false, so only answer when you know: a transport that cannot tell
        keeps the default. The cost of the two mistakes is asymmetric. A
        false ``False`` throws away live sessions; a false ``True`` merely
        lets the next request fail on a dead pipe, which the channel already
        reports.
        """
        return True


class StdioACPTransport(ACPTransport):
    """Spawn the agent locally and speak ACP over its stdio.

    The channel builds one of these for you when you pass ``command=``; name
    it explicitly only to hold a configured spawn on its own.

    Args:
        command: Executable and arguments. Run directly — no shell.
        cwd: Absolute working directory for the spawned process.
        env: Environment variables added to the SDK's restricted spawn env.
        inherit_env: Parent-process variable names to forward. The SDK trims
            the environment to ``HOME/LOGNAME/PATH/SHELL/TERM/USER`` (the MCP
            practice), which silently breaks tooling a coding agent relies on
            — without ``SSH_AUTH_SOCK`` every git-over-SSH operation prompts
            for key passphrases on the controlling terminal. Read at each
            spawn, so a reconnect picks up a rotated value; unset names are
            skipped; explicit ``env`` entries win.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path,
        env: Mapping[str, str] | None = None,
        inherit_env: Sequence[str] | None = None,
    ) -> None:
        if isinstance(command, str) or not command:
            raise ValueError("command must be a non-empty sequence of arguments")
        if any(not isinstance(arg, str) or not arg for arg in command):
            raise ValueError("every command argument must be a non-empty string")
        if env is not None and any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in env.items()
        ):
            raise ValueError("env keys and values must be strings")
        if inherit_env is not None and (
            isinstance(inherit_env, str)
            or any(not isinstance(name, str) or not name for name in inherit_env)
        ):
            raise ValueError("inherit_env must be a sequence of non-empty variable names")

        self._command = tuple(command)
        self._cwd = _absolute_path(cwd, field_name="cwd")
        self._env = dict(env) if env is not None else None
        self._inherit_env = tuple(inherit_env or ())
        self._context: Any = None
        self._process: Any = None
        self._stderr_task: asyncio.Task[None] | None = None

    @property
    def name(self) -> str:
        return "stdio"

    @property
    def process(self) -> Any:
        """The spawned process, or ``None`` before the first :meth:`open`."""
        return self._process

    async def open(self, client: Any, *, queue: Any) -> Any:
        sdk = _load_sdk()
        context = sdk.acp.spawn_agent_process(
            client,
            self._command[0],
            *self._command[1:],
            env=_resolve_spawn_env(self._inherit_env, self._env, os.environ),
            cwd=self._cwd,
            queue=queue,
        )
        # Nothing of ours exists to undo if entering fails: the SDK's own
        # context manager unwinds the half-started process, so record the
        # context only once it has actually yielded.
        connection, process = await context.__aenter__()
        self._context = context
        self._process = process
        self._stderr_task = asyncio.create_task(self._drain_stderr(process))
        return connection

    def is_alive(self) -> bool:
        return self._process is None or self._process.returncode is None

    async def close(self) -> None:
        context = self._context
        self._context = None
        self._process = None
        if context is not None:
            with contextlib.suppress(Exception):
                await context.__aexit__(None, None, None)
        await self._stop_stderr_task()

    async def _stop_stderr_task(self) -> None:
        task = self._stderr_task
        self._stderr_task = None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @staticmethod
    async def _drain_stderr(process: Any) -> None:
        stream = getattr(process, "stderr", None)
        if stream is None:
            return
        while await stream.read(8192):
            pass
