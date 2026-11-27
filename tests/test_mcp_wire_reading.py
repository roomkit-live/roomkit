"""What an MCP provider sends and reads on the wire under the SDK 2 (RFC §21.2, RMK-663).

The SDK writes the host's trace context into every request it sends and checks
a method it knows against its schema; the provider reads the server's words
off the wire instead, and sends only the trace keys its caller wrote.
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import baggage, context, trace
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags
from pydantic import ValidationError

from roomkit.tools import MCPToolProvider
from tests.conformance._mcp_peer import MCPPeer

_CALLER_TRACE = "00-0000000000000000000000000000abcd-000000000000beef-01"


@contextmanager
def _host_span() -> Iterator[None]:
    """A current span and baggage in the host, as its own telemetry sets them."""
    span_context = SpanContext(
        trace_id=0x1234, span_id=0x5678, is_remote=False, trace_flags=TraceFlags(1)
    )
    with trace.use_span(NonRecordingSpan(span_context)):
        token = context.attach(baggage.set_baggage("tenant", "t1"))
        try:
            yield
        finally:
            context.detach(token)


async def test_a_request_carries_no_trace_context_its_caller_did_not_write() -> None:
    replies = [
        {"method": "x/read", "result": {"n": 1}},
        {"method": "x/read", "params": {"_meta": {"traceparent": _CALLER_TRACE}}, "result": {}},
        {
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {}},
            "result": {"content": [{"type": "text", "text": "ok"}]},
        },
    ]
    async with MCPPeer({"replies": replies}) as peer:
        provider = MCPToolProvider(peer.url, discover=False)
        async with provider:
            with _host_span():
                assert await provider.request("x/read") == {"n": 1}
                await provider.request("x/read", {"_meta": {"traceparent": _CALLER_TRACE}})
                await provider.call_tool_result("echo", {})
    assert peer.problems == []


_HOST_TRACE = "00000000000000000000000000001234"
"""The trace id of :func:`_host_span`, as W3C writes it."""


@pytest.mark.parametrize(
    ("meta", "traced"),
    [({}, _HOST_TRACE), ({"traceparent": _CALLER_TRACE}, _CALLER_TRACE.split("-")[1])],
    ids=["host", "caller"],
)
async def test_a_provider_that_propagates_the_trace_context_sends_the_host_s(
    meta: dict[str, Any], traced: str, tmp_path: Path
) -> None:
    """``propagate_trace_context=True``: each request carries the trace
    context of the host's current span, unless its caller wrote one of its
    own, which goes whole: no host baggage beside the caller's traceparent."""
    script, log = tmp_path / "server.py", tmp_path / "received.jsonl"
    script.write_text(_RECORDING_SERVER)
    provider = MCPToolProvider.from_command(
        sys.executable, [str(script), str(log)], discover=False, propagate_trace_context=True
    )
    async with provider:
        with _host_span(), pytest.raises(TimeoutError):
            # The recording server answers nothing past the handshake.
            await provider.request("x/read", {"_meta": meta} if meta else None, timeout=0.05)
    sent = [json.loads(line) for line in log.read_text().splitlines()]
    request = next(message for message in sent if message.get("method") == "x/read")
    trace = request["params"]["_meta"]
    assert trace["traceparent"].split("-")[1] == traced
    assert ("baggage" in trace) == (not meta)


class _StringIdPeer(MCPPeer):
    """A server that echoes each request id as a string, ``"7"`` for ``7``."""

    def _reply(self, message: dict[str, Any]) -> dict[str, Any] | None:
        reply = super()._reply(message)
        if reply is not None:
            reply["id"] = str(reply["id"])
        return reply


async def test_a_server_echoing_ids_as_strings_is_read_off_the_wire() -> None:
    """The SDK correlates ``"7"`` with request 7; the provider reads that
    answer too, its handshake and its raw results (RFC §21.2)."""
    spec = {
        "server_info": {"name": "strid", "version": "1", "vendor": "x"},
        "instructions": "be nice",
        "replies": [{"method": "x/read", "result": {"value": 1}}],
    }
    async with _StringIdPeer(spec) as peer:
        provider = MCPToolProvider(peer.url, discover=False)
        async with provider:
            assert provider.server_info == {"name": "strid", "version": "1", "vendor": "x"}
            assert provider.instructions == "be nice"
            assert await provider.request("x/read") == {"value": 1}
    assert peer.problems == []


@pytest.mark.parametrize("method", ["resources/list", "prompts/list", "completion/complete"])
async def test_a_request_the_sdk_models_still_returns_what_the_server_sent(method: str) -> None:
    """``request`` returns the result as the server sent it, whichever method
    it names: the SDK's check of a method it knows does not reach it."""
    replies = [{"method": method, "result": {"echo": "outside the method's schema"}}]
    async with MCPPeer({"replies": replies}) as peer:
        provider = MCPToolProvider(peer.url, discover=False)
        async with provider:
            assert await provider.request(method) == {"echo": "outside the method's schema"}
    assert peer.problems == []


_RECORDING_SERVER = textwrap.dedent(
    """\
    import json, sys

    log = open(sys.argv[1], "a")
    for line in sys.stdin:
        log.write(line)
        log.flush()
        message = json.loads(line)
        if message.get("method") == "server/discover":
            error = {"code": -32601, "message": "Method not found"}
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": message["id"], "error": error}))
            sys.stdout.write("\\n")
            sys.stdout.flush()
        if message.get("method") == "initialize":
            result = {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "serverInfo": {"name": "recording", "version": "1"},
            }
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}))
            sys.stdout.write("\\n")
            sys.stdout.flush()
    """
)


@pytest.mark.parametrize("call", ["request", "call_tool", "read_resource"])
async def test_a_call_given_no_time_sends_nothing(call: str, tmp_path: Path) -> None:
    """A timeout of 0 expires before the call starts: no request goes out,
    hence no cancellation either. Over stdio, where a started call writes at
    once."""
    script, log = tmp_path / "server.py", tmp_path / "received.jsonl"
    script.write_text(_RECORDING_SERVER)
    provider = MCPToolProvider.from_command(
        sys.executable, [str(script), str(log)], discover=False
    )
    async with provider:
        with pytest.raises(TimeoutError):
            if call == "request":
                await provider.request("x/read", timeout=0)
            elif call == "call_tool":
                await provider.call_tool("echo", {}, timeout=0)
            else:
                await provider.read_resource("file:///a", timeout=0)
    received = [json.loads(line)["method"] for line in log.read_text().splitlines()]
    assert received == ["server/discover", "initialize", "notifications/initialized"]


async def test_a_listing_is_read_by_its_wire_names_only() -> None:
    """A tool listed with ``input_schema`` lacks MCP's ``inputSchema``: the
    listing is refused, as roomkit-rs and the SDK 1 refused it."""
    listing = {"tools": [{"name": "lookup", "input_schema": {"type": "object"}}]}
    async with MCPPeer({"replies": [{"method": "tools/list", "result": listing}]}) as peer:
        provider = MCPToolProvider(peer.url)
        with pytest.raises(ValidationError, match="inputSchema"):
            await provider.__aenter__()
        assert not provider.connected


async def test_a_request_sent_after_a_ticket_s_block_is_not_read_for_it() -> None:
    """A task started in a ticket's block inherits the ticket; a request it
    sends once the block is over is not bound to it, and its answer reaches
    the session as the server sent it."""
    from mcp.shared.message import SessionMessage
    from mcp.types import JSONRPCRequest, JSONRPCResponse

    from roomkit.tools._mcp_capture import WireCapture

    written: list[Any] = []
    answers: list[Any] = []

    class _Write:
        async def send(self, item: Any) -> None:
            written.append(item)

    class _Read:
        async def receive(self) -> Any:
            return answers.pop(0)

    capture = WireCapture()
    incoming, outgoing = capture.streams(_Read(), _Write())
    block_over = asyncio.Event()

    async def late_sender() -> None:
        await block_over.wait()
        await outgoing.send(
            SessionMessage(JSONRPCRequest(jsonrpc="2.0", id=9, method="initialize"))
        )

    with capture.ticket(
        lambda _result: {"shown": True}, methods=frozenset({"initialize"})
    ) as ticket:
        task = asyncio.create_task(late_sender())
    block_over.set()
    await task
    answers.append(SessionMessage(JSONRPCResponse(jsonrpc="2.0", id=9, result={"sent": 1})))
    assert (await incoming.receive()).message.result == {"sent": 1}
    assert ticket.result is None
