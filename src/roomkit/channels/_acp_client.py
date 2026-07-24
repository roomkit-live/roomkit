"""Internal ACP SDK loading, process lifecycle, callback adapter, and turn state."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from roomkit._version import __version__
from roomkit.models.streaming import StreamDelta

if TYPE_CHECKING:
    from roomkit.tools.external import ExternalToolHandler

logger = logging.getLogger("roomkit.channels.acp")

_STABLE_PROTOCOL_VERSION = 1


@dataclass(frozen=True, slots=True)
class _SDK:
    acp: ModuleType
    schema: ModuleType
    task: ModuleType
    version: str


@dataclass(slots=True)
class _ToolState:
    tool_id: str
    name: str = "tool"
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_output: Any = None
    content: Any = None
    started_at: float = field(default_factory=time.monotonic)
    started: bool = False
    finished: bool = False


@dataclass(slots=True)
class _TurnDone:
    error: BaseException | None = None


@dataclass(slots=True)
class _TurnState:
    room_id: str
    queue: asyncio.Queue[StreamDelta | _TurnDone] = field(default_factory=asyncio.Queue)
    tools: dict[str, _ToolState] = field(default_factory=dict)
    thinking_open: bool = False
    runner: asyncio.Task[None] | None = None


def _load_sdk() -> _SDK:
    """Load the optional official ACP SDK with an actionable error."""
    try:
        import acp
        import acp.schema
        import acp.task
    except ImportError as exc:
        raise ImportError(
            "ACPChannel requires the official Agent Client Protocol SDK. "
            "Install it with `pip install roomkit[acp]`."
        ) from exc

    try:
        sdk_version = metadata.version("agent-client-protocol")
    except metadata.PackageNotFoundError:
        sdk_version = "unknown"
    return _SDK(acp=acp, schema=acp.schema, task=acp.task, version=sdk_version)


def _absolute_path(value: str | Path, *, field_name: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be an absolute path")
    return str(path)


def _option_kind(option: Any) -> str:
    kind = getattr(option, "kind", "")
    return str(getattr(kind, "value", kind))


def _model_dump(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    if dump is not None:
        return dump(mode="json", by_alias=True, exclude_none=True)
    if isinstance(value, Mapping):
        return {str(key): _model_dump(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_model_dump(item) for item in value]
    return value


def _result_text(value: Any) -> str:
    value = _model_dump(value)
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


class _ACPClient:
    """ACP callbacks invoked by the external agent."""

    def __init__(self, channel: Any) -> None:
        self._channel = channel
        self._session_update_tasks: dict[str, set[asyncio.Task[Any]]] = {}

    async def request_permission(
        self,
        session_id: str,
        tool_call: Any,
        options: list[Any],
        **kwargs: Any,
    ) -> Any:
        return await self._channel._request_permission(session_id, tool_call, options)

    async def session_update(
        self,
        session_id: str,
        update: Any,
        **kwargs: Any,
    ) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._session_update_tasks.setdefault(session_id, set()).add(task)
        try:
            await self._channel._receive_update(session_id, update)
        finally:
            if task is not None:
                tasks = self._session_update_tasks.get(session_id)
                if tasks is not None:
                    tasks.discard(task)
                    if not tasks:
                        self._session_update_tasks.pop(session_id, None)

    async def drain_session_updates(self, session_id: str) -> None:
        """Wait for already-dispatched updates for one session to finish."""
        # The SDK schedules notification handlers as tasks. Yield once so every
        # handler dispatched by the message queue can register itself above.
        await asyncio.sleep(0)
        while tasks := tuple(self._session_update_tasks.get(session_id, ())):
            await asyncio.gather(*tasks, return_exceptions=True)

    async def write_text_file(self, *args: Any, **kwargs: Any) -> Any:
        raise self._unsupported("fs/write_text_file")

    async def read_text_file(self, *args: Any, **kwargs: Any) -> Any:
        raise self._unsupported("fs/read_text_file")

    async def create_terminal(self, *args: Any, **kwargs: Any) -> Any:
        raise self._unsupported("terminal/create")

    async def terminal_output(self, *args: Any, **kwargs: Any) -> Any:
        raise self._unsupported("terminal/output")

    async def release_terminal(self, *args: Any, **kwargs: Any) -> Any:
        raise self._unsupported("terminal/release")

    async def wait_for_terminal_exit(self, *args: Any, **kwargs: Any) -> Any:
        raise self._unsupported("terminal/wait_for_exit")

    async def kill_terminal(self, *args: Any, **kwargs: Any) -> Any:
        raise self._unsupported("terminal/kill")

    async def create_elicitation(self, *args: Any, **kwargs: Any) -> Any:
        raise self._unsupported("session/request_input")

    async def complete_elicitation(self, *args: Any, **kwargs: Any) -> None:
        logger.debug("Ignoring unsupported ACP elicitation completion")

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        raise self._unsupported(f"_{method}")

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        logger.debug("Ignoring unsupported ACP extension notification: %s", method)

    def on_connect(self, conn: Any) -> None:
        """Connection callback required by the ACP client interface."""

    def _unsupported(self, method: str) -> Exception:
        return self._channel._sdk().acp.RequestError.method_not_found(method)


class ACPConnectionMixin:
    """Own the ACP agent subprocess and the initialized stdio connection."""

    _client: _ACPClient
    _command: tuple[str, ...]
    _cwd: str
    _env: dict[str, str] | None
    _authentication_method: str | None
    _external_tool_handler: ExternalToolHandler | None
    _loaded_sdk: _SDK | None
    _connection: Any
    _process: Any
    _process_context: Any
    _message_queue: Any
    _stderr_task: asyncio.Task[None] | None
    _connect_lock: asyncio.Lock
    _sessions: dict[str, str]
    _session_rooms: dict[str, str]
    _agent_info: dict[str, Any] | None
    _handler_started: bool
    _closed: bool

    def _sdk(self) -> _SDK:
        if self._loaded_sdk is None:
            self._loaded_sdk = _load_sdk()
        return self._loaded_sdk

    def _create_process_context(self, sdk: _SDK) -> Any:
        self._message_queue = sdk.task.InMemoryMessageQueue()
        return sdk.acp.spawn_agent_process(
            self._client,
            self._command[0],
            *self._command[1:],
            env=self._env,
            cwd=self._cwd,
            queue=self._message_queue,
        )

    async def _drain_session_updates(self, session_id: str) -> None:
        # A prompt response is resolved directly by the SDK receive loop, while
        # preceding notifications are dispatched through this queue. Joining it
        # first guarantees those handlers exist before the client awaits them.
        if self._message_queue is not None:
            await self._message_queue.join()
        await self._client.drain_session_updates(session_id)

    async def _ensure_connection(self) -> Any:
        if self._closed:
            raise RuntimeError("ACPChannel is closed")
        if self._connection is not None and (
            self._process is None or self._process.returncode is None
        ):
            return self._connection

        async with self._connect_lock:
            if self._closed:
                raise RuntimeError("ACPChannel is closed")
            if self._connection is not None and (
                self._process is None or self._process.returncode is None
            ):
                return self._connection
            if self._process_context is not None:
                await self._close_process()
                self._sessions.clear()
                self._session_rooms.clear()

            sdk = self._sdk()
            if sdk.acp.PROTOCOL_VERSION != _STABLE_PROTOCOL_VERSION:
                raise RuntimeError(
                    "Unsupported ACP SDK protocol version "
                    f"{sdk.acp.PROTOCOL_VERSION}; RoomKit supports stable ACP v1"
                )

            process_context = self._create_process_context(sdk)
            try:
                connection, process = await process_context.__aenter__()
                self._stderr_task = asyncio.create_task(self._drain_stderr(process))
                response = await connection.initialize(
                    _STABLE_PROTOCOL_VERSION,
                    client_capabilities=sdk.schema.ClientCapabilities(
                        fs=sdk.schema.FileSystemCapabilities(
                            read_text_file=False,
                            write_text_file=False,
                        ),
                        terminal=False,
                    ),
                    client_info=sdk.schema.Implementation(
                        name="roomkit",
                        title="RoomKit",
                        version=__version__,
                    ),
                )
                if response.protocol_version != _STABLE_PROTOCOL_VERSION:
                    raise RuntimeError(
                        "ACP protocol negotiation failed: agent selected "
                        f"version {response.protocol_version}, expected "
                        f"{_STABLE_PROTOCOL_VERSION}"
                    )
                if self._authentication_method is not None:
                    await connection.authenticate(self._authentication_method)
                if self._external_tool_handler is not None and not self._handler_started:
                    await self._external_tool_handler.start()
                    self._handler_started = True
            except BaseException:
                await process_context.__aexit__(*sys.exc_info())
                await self._stop_stderr_task()
                raise

            self._process_context = process_context
            self._connection = connection
            self._process = process
            agent_info = getattr(response, "agent_info", None)
            self._agent_info = _model_dump(agent_info) if agent_info is not None else None
            return connection

    async def _close_process(self) -> None:
        process_context = self._process_context
        self._connection = None
        self._process = None
        self._process_context = None
        if process_context is not None:
            with contextlib.suppress(Exception):
                await process_context.__aexit__(None, None, None)
        self._message_queue = None
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
