"""Internal ACP SDK loading, connection lifecycle, callback adapter, and turn state."""

from __future__ import annotations

import asyncio
import json
import logging
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
    from roomkit.channels.acp_transport import ACPTransport
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
    started_at: float = field(default_factory=time.monotonic)
    segments: list[list[str]] = field(default_factory=list)
    """What the agent said this turn, the chunks of each stretch between tool calls.

    The queue hands each chunk to the consumer and forgets it, so the turn
    would otherwise end with no idea what it produced. A tool call closes the
    current stretch — the boundary ``segment_stream`` persists a MESSAGE on —
    and the chunks after it open the next. Chunks, joined once at the report:
    a coding agent streams whole files, and growing one string per chunk
    would copy the stretch on every one of them.
    """

    tokens: dict[str, int] = field(default_factory=dict)
    """Token counters the agent returned when the prompt ended."""

    context: dict[str, Any] = field(default_factory=dict)
    """Last context-window occupancy and running cost the agent announced."""

    completed: bool = False
    """Whether the turn reached its terminal item without an error.

    A stream closed from the outside never gets there: the agent is cancelled
    and its text is not delivered, so that turn is not a response.
    """


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


_TOKEN_FIELDS = (
    "total_tokens",
    "input_tokens",
    "output_tokens",
    "thought_tokens",
    "cached_read_tokens",
    "cached_write_tokens",
)


def _usage_tokens(usage: Any) -> dict[str, int]:
    """Read the ``Usage`` an agent returns when a prompt ends.

    Taken off the model by attribute rather than out of :func:`_model_dump`,
    whose camelCase aliases would not sit beside the counters an in-process
    provider reports under the same key.

    Relayed exactly as the agent sent it. The ACP schema annotates these
    fields as running session figures ("total input tokens across all turns")
    while the reference agent fills them per prompt — measured against it,
    ``cached_read_tokens`` is the whole prefix re-read on that turn, not a sum
    over turns. RoomKit cannot tell the two apart from one reading, and
    guessing wrong corrupts the number in the direction nobody can detect
    downstream, so it does no arithmetic on them at all.
    """
    counters: dict[str, int] = {}
    for name in _TOKEN_FIELDS:
        value = getattr(usage, name, None)
        if isinstance(value, int):
            counters[name] = value
    return counters


def _usage_context(update: Any) -> dict[str, Any]:
    """Read a usage notification: how full the context is, what it has cost.

    A different quantity from the token counters above — ``used``/``size``
    describe the window the session is living in, and ``cost`` is its running
    total. These really are cumulative, and observably so: they climb turn
    after turn.
    """
    context: dict[str, Any] = {}
    used = getattr(update, "used", None)
    if isinstance(used, int):
        context["context_used"] = used
    size = getattr(update, "size", None)
    if isinstance(size, int):
        context["context_size"] = size
    cost = getattr(update, "cost", None)
    amount = getattr(cost, "amount", None)
    if isinstance(amount, (int, float)):
        context["cost"] = float(amount)
    currency = getattr(cost, "currency", None)
    if isinstance(currency, str) and currency:
        context["currency"] = currency
    return context


def _usage_report(tokens: dict[str, int], context: dict[str, Any]) -> dict[str, Any]:
    """The accounting a finished turn carries: what the agent counted, and where.

    Two readings side by side under distinct keys — the token counters from
    the prompt's own response, and the session's context occupancy and running
    cost. Both are the agent's figures, unaltered; see :func:`_usage_tokens`
    for why nothing here is differenced.
    """
    return {**tokens, **context}


def _config_values(options: Any) -> dict[str, str | bool]:
    """Map ACP session config options onto ``{config_id: current value}``.

    Agents expose their tunables through this one list — ``model``, ``mode``,
    ``effort``, vendor-specific switches — as select (str) or boolean
    options. Dumps use camelCase aliases, so ``currentValue`` is the key;
    the snake_case spelling is tolerated for hand-built payloads.
    """
    values: dict[str, str | bool] = {}
    dumped = _model_dump(options)
    if not isinstance(dumped, list):
        return values
    for option in dumped:
        if not isinstance(option, Mapping):
            continue
        config_id = option.get("id")
        value = option.get("currentValue", option.get("current_value"))
        if isinstance(config_id, str) and config_id and isinstance(value, (str, bool)):
            values[config_id] = value
    return values


def _config_labels(options: Any) -> dict[str, str]:
    """Map config options onto ``{config_id: display name of current value}``.

    Values are identifiers, and an agent's default entry can be as opaque as
    ``"default"`` while its name reads ``"Default (recommended)"`` — a status
    bar wants the name. Options come flat or in groups; both are searched.
    Config ids without a matching entry are left out rather than guessed.
    """
    labels: dict[str, str] = {}
    dumped = _model_dump(options)
    if not isinstance(dumped, list):
        return labels
    for option in dumped:
        if not isinstance(option, Mapping):
            continue
        config_id = option.get("id")
        current = option.get("currentValue", option.get("current_value"))
        if not isinstance(config_id, str) or not isinstance(current, str):
            continue
        name = _entry_name(option.get("options"), current)
        if name:
            labels[config_id] = name
    return labels


def _entry_name(entries: Any, value: str) -> str | None:
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        if entry.get("value") == value:
            name = entry.get("name")
            return name if isinstance(name, str) and name else None
        nested = _entry_name(entry.get("options"), value)  # a select group
        if nested:
            return nested
    return None


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
    """Own the initialized ACP connection, whatever transport carries it."""

    _client: _ACPClient
    _transport: ACPTransport
    _authentication_method: str | None
    _external_tool_handler: ExternalToolHandler | None
    _loaded_sdk: _SDK | None
    _connection: Any
    _message_queue: Any
    _connect_lock: asyncio.Lock
    _sessions: dict[str, str]
    _session_rooms: dict[str, str]
    _session_options: dict[str, list[Any]]
    _prompted_index: dict[str, int]
    _agent_info: dict[str, Any] | None
    _handler_started: bool
    _closed: bool

    def _sdk(self) -> _SDK:
        if self._loaded_sdk is None:
            self._loaded_sdk = _load_sdk()
        return self._loaded_sdk

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
        if self._connection is not None and self._transport.is_alive():
            return self._connection

        async with self._connect_lock:
            if self._closed:
                raise RuntimeError("ACPChannel is closed")
            if self._connection is not None and self._transport.is_alive():
                return self._connection
            if self._connection is not None:
                # The agent behind the old connection is gone, and its
                # sessions with it — a reconnect never resumes them.
                await self._close_transport()
                self._sessions.clear()
                self._session_rooms.clear()
                self._session_options.clear()
                # Catch-up state belongs to the sessions that just died. The
                # replacement agent starts empty and must receive the visible
                # room history again on its first prompt.
                self._prompted_index.clear()

            sdk = self._sdk()
            if sdk.acp.PROTOCOL_VERSION != _STABLE_PROTOCOL_VERSION:
                raise RuntimeError(
                    "Unsupported ACP SDK protocol version "
                    f"{sdk.acp.PROTOCOL_VERSION}; RoomKit supports stable ACP v1"
                )

            self._message_queue = sdk.task.InMemoryMessageQueue()
            try:
                connection = await self._transport.open(self._client, queue=self._message_queue)
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
                # Covers a transport that never opened as well as a handshake
                # that failed on a live one: closing is defined to tolerate both.
                await self._close_transport()
                raise

            self._connection = connection
            agent_info = getattr(response, "agent_info", None)
            self._agent_info = _model_dump(agent_info) if agent_info is not None else None
            return connection

    async def _close_transport(self) -> None:
        self._connection = None
        self._message_queue = None
        await self._transport.close()
