"""The server's own results, kept under the MCP SDK's session (RFC §21.2).

The SDK checks a known method's result against the negotiated protocol
revision and trims the fields of later revisions before its caller sees it.
Where the RFC says a host keeps what the server sent, RoomKit reads the
result off the transport instead: the two streams between the transport and
the session pass through a :class:`WireCapture`, which keeps the raw result
of a request made under one of its tickets and hands the session what that
ticket shows it in its place. roomkit-rs does the same (``roomkit-mcp``'s
``raw.rs``).

The streams are wrapped, not relayed: a message reaches the transport when
the session's own write does, so a request abandoned before the transport
takes it is still never sent.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

Shown = Callable[[dict[str, Any]], dict[str, Any]]
"""What the session reads in place of a ticket's result, given that result."""


@dataclass
class Ticket:
    """One request whose result is read off the wire: the result, once it came."""

    shown: Shown
    result: dict[str, Any] | None = None
    request_id: Any = field(default=None, repr=False)


_TICKET: ContextVar[Ticket | None] = ContextVar("roomkit_mcp_ticket", default=None)


class WireCapture:
    """The raw results of the requests made under :meth:`ticket` on one
    connection."""

    def __init__(self) -> None:
        from mcp.shared.message import SessionMessage
        from mcp.types import JSONRPCRequest, JSONRPCResponse

        self._message = SessionMessage
        self._request = JSONRPCRequest
        self._response = JSONRPCResponse
        self._sent: dict[Any, Ticket] = {}

    def streams(self, read: Any, write: Any) -> tuple[Any, Any]:
        """The transport's *read* and *write* streams, as the session reads
        and writes them through this capture."""
        return _Incoming(read, self), _Outgoing(write, self)

    @contextmanager
    def ticket(self, shown: Shown) -> Generator[Ticket]:
        """Read off the wire the result of the first request sent in this
        block, the session reading ``shown(result)`` in its place.

        The block is the one task that sends the request: the SDK's session
        writes a request from its caller's task.
        """
        ticket = Ticket(shown)
        token = _TICKET.set(ticket)
        try:
            yield ticket
        finally:
            _TICKET.reset(token)
            if ticket.request_id is not None:
                self._sent.pop(ticket.request_id, None)

    def _outgoing(self, item: Any) -> None:
        ticket = _TICKET.get()
        message = getattr(item, "message", None)
        if ticket is None or ticket.request_id is not None:
            return
        if isinstance(message, self._request):
            ticket.request_id = message.id
            self._sent[message.id] = ticket

    def _incoming(self, item: Any) -> Any:
        message = getattr(item, "message", None)
        if not isinstance(message, self._response):
            return item
        ticket = self._sent.pop(message.id, None)
        if ticket is None:
            return item
        ticket.result = message.result
        shown = self._response(jsonrpc="2.0", id=message.id, result=ticket.shown(message.result))
        return self._message(message=shown, metadata=item.metadata)


class _Outgoing:
    """The transport's write stream, which shows the capture each message the
    session writes before writing it."""

    def __init__(self, stream: Any, capture: WireCapture) -> None:
        self._stream = stream
        self._capture = capture

    async def send(self, item: Any) -> None:
        self._capture._outgoing(item)
        await self._stream.send(item)

    async def __aenter__(self) -> _Outgoing:
        await self._stream.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        return await self._stream.__aexit__(*exc)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _Incoming:
    """The transport's read stream, which hands each message the transport
    reads to the capture before the session sees it."""

    def __init__(self, stream: Any, capture: WireCapture) -> None:
        self._stream = stream
        self._capture = capture

    async def receive(self) -> Any:
        return self._capture._incoming(await self._stream.receive())

    def __aiter__(self) -> _Incoming:
        return self

    async def __anext__(self) -> Any:
        return self._capture._incoming(await self._stream.__anext__())

    async def __aenter__(self) -> _Incoming:
        await self._stream.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        return await self._stream.__aexit__(*exc)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)
