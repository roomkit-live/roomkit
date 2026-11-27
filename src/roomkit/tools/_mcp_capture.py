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

The capture also takes out of each message the W3C trace context the SDK
writes into its ``_meta`` (``traceparent``, ``tracestate``, ``baggage``)
whenever the host has a current span: a request carries the trace keys its
caller wrote, and no other, as roomkit-rs's do.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

Shown = Callable[[dict[str, Any]], dict[str, Any]]
"""What the session reads in place of a ticket's result, given that result."""


_TRACE_KEYS = ("traceparent", "tracestate", "baggage")
"""The ``_meta`` keys the SDK writes a host's trace context into (SEP-414)."""


@dataclass
class Ticket:
    """One request whose result is read off the wire: the result, once it came.

    ``own_trace`` holds the trace keys the caller itself wrote in the
    request's ``_meta``, which it keeps as written. ``methods``, when given,
    makes the ticket follow each request of those methods sent in its block,
    not only the first: a negotiation that probes ``server/discover`` and
    falls back to ``initialize`` is read from whichever was answered."""

    shown: Shown
    own_trace: dict[str, Any] = field(default_factory=dict)
    methods: frozenset[str] | None = None
    result: dict[str, Any] | None = None
    request_id: Any = field(default=None, repr=False)

    def follows(self, method: str) -> bool:
        """Whether a request of *method* sent in the ticket's block is the
        one its result is read from."""
        if self.methods is None:
            return self.request_id is None
        return method in self.methods

    def taken(self) -> dict[str, Any]:
        """The result read off the wire, once the session's call returned.

        Raises:
            RuntimeError: No result was read for the request: the session
                answered it without the capture seeing its response.
        """
        if self.result is None:
            raise RuntimeError("the MCP server's result was not read off the wire")
        return self.result


def _correlated(request_id: Any) -> Any:
    """*request_id* as the SDK correlates it: a string that reads as an
    integer is that integer, so a server that echoes ``7`` as ``"7"`` still
    answers request 7 (the SDK's ``coerce_request_id``)."""
    if isinstance(request_id, str):
        try:
            return int(request_id)
        except ValueError:
            return request_id
    return request_id


_TICKET: ContextVar[Ticket | None] = ContextVar("roomkit_mcp_ticket", default=None)


class WireCapture:
    """The raw results of the requests made under :meth:`ticket` on one
    connection."""

    def __init__(self) -> None:
        from mcp.shared.message import SessionMessage
        from mcp.types import JSONRPCError, JSONRPCNotification, JSONRPCRequest, JSONRPCResponse

        self._message = SessionMessage
        self._request = JSONRPCRequest
        self._notification = JSONRPCNotification
        self._response = JSONRPCResponse
        self._error = JSONRPCError
        self._sent: dict[Any, Ticket] = {}

    def streams(self, read: Any, write: Any) -> tuple[Any, Any]:
        """The transport's *read* and *write* streams, as the session reads
        and writes them through this capture."""
        return _Incoming(read, self), _Outgoing(write, self)

    @contextmanager
    def ticket(
        self, shown: Shown, meta: Any = None, *, methods: frozenset[str] | None = None
    ) -> Generator[Ticket]:
        """Read off the wire the result of the first request sent in this
        block (with *methods*, of the last request of one of them sent in it),
        the session reading ``shown(result)`` in its place. *meta* is the
        ``_meta`` its caller wrote, whose trace keys the request keeps.

        The block is the one task that sends the request: the SDK's session
        writes a request from its caller's task.
        """
        own = (
            {key: meta[key] for key in _TRACE_KEYS if key in meta}
            if isinstance(meta, dict)
            else {}
        )
        ticket = Ticket(shown, own, methods)
        token = _TICKET.set(ticket)
        try:
            yield ticket
        finally:
            _TICKET.reset(token)
            if ticket.request_id is not None:
                self._sent.pop(ticket.request_id, None)

    def _outgoing(self, item: Any) -> Any:
        """*item* as it goes out: the SDK's trace keys taken out, and a
        request made under a ticket bound to it."""
        ticket = _TICKET.get()
        message = getattr(item, "message", None)
        binds = (
            ticket is not None
            and isinstance(message, self._request)
            and ticket.follows(message.method)
        )
        if binds and ticket is not None:
            if ticket.request_id is not None:
                self._sent.pop(ticket.request_id, None)
            ticket.request_id = _correlated(message.id)
            self._sent[ticket.request_id] = ticket
        if isinstance(message, (self._request, self._notification)):
            return self._untraced(item, message, ticket.own_trace if binds and ticket else {})
        return item

    def _untraced(self, item: Any, message: Any, own: dict[str, Any]) -> Any:
        params = message.params
        meta = params.get("_meta") if isinstance(params, dict) else None
        if not isinstance(meta, dict) or not any(key in meta for key in _TRACE_KEYS):
            return item
        kept = {key: value for key, value in meta.items() if key not in _TRACE_KEYS} | own
        rest = {key: value for key, value in params.items() if key != "_meta"}
        params = rest | {"_meta": kept} if kept else rest
        fields = message.model_dump(exclude={"params"}, exclude_unset=True)
        # No params at all rather than an empty or null one (RK-21.2-046).
        rewritten = type(message)(**fields, **({"params": params} if params else {}))
        return self._message(message=rewritten, metadata=item.metadata)

    def _incoming(self, item: Any) -> Any:
        message = getattr(item, "message", None)
        if isinstance(message, self._error):
            # An error answers the request: no result to read for it.
            self._sent.pop(_correlated(message.id), None)
            return item
        if not isinstance(message, self._response):
            return item
        ticket = self._sent.pop(_correlated(message.id), None)
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
        await self._stream.send(self._capture._outgoing(item))

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
