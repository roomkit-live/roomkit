"""What an MCP provider sends and reads on the wire under the SDK 2 (RFC §21.2, RMK-663).

The SDK writes the application's OpenTelemetry context into every request it
sends and checks a method it knows against its schema; the provider reads the
server's words off the wire instead, and sends only the trace keys its caller
wrote, or the kit's when it propagates the trace context.
"""

from __future__ import annotations

import asyncio
import copy
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

from roomkit import AIChannel, InboundMessage, RoomKit, TextContent
from roomkit.providers.ai import AIResponse, AITool, AIToolCall, MockAIProvider
from roomkit.telemetry import MockTelemetryProvider, get_current_span
from roomkit.tools import MCPToolProvider
from tests.conformance._mcp_peer import MCPPeer
from tests.test_framework import SimpleChannel

_CALLER_TRACE = "00-0000000000000000000000000000abcd-000000000000beef-01"

_TRACE_KEYS = ("traceparent", "tracestate", "baggage")


@contextmanager
def _application_span() -> Iterator[None]:
    """A current span and baggage of the application's own OpenTelemetry,
    which are not the kit's: the trace id ``aaaa``."""
    span_context = SpanContext(
        trace_id=0xAAAA, span_id=0xBBBB, is_remote=False, trace_flags=TraceFlags(1)
    )
    with trace.use_span(NonRecordingSpan(span_context)):
        token = context.attach(baggage.set_baggage("tenant", "app"))
        try:
            yield
        finally:
            context.detach(token)


@pytest.mark.parametrize("propagating", [False, True], ids=["local", "propagating"])
async def test_a_request_carries_no_trace_context_its_caller_did_not_write(
    propagating: bool,
) -> None:
    """Outside the kit's spans, even a provider that propagates the trace
    context sends none: the application's OpenTelemetry span is not the
    kit's, and the SDK's trace keys never go out."""
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
        provider = MCPToolProvider(peer.url, discover=False, propagate_trace_context=propagating)
        async with provider:
            with _application_span():
                assert await provider.request("x/read") == {"n": 1}
                await provider.request("x/read", {"_meta": {"traceparent": _CALLER_TRACE}})
                await provider.call_tool_result("echo", {})
    assert peer.problems == []


_KIT_TRACE = "00000000000000000000000000001234"
"""The trace id :class:`_W3CTelemetry` names the kit's spans under."""


class _W3CTelemetry(MockTelemetryProvider):
    """A kit's telemetry whose spans name themselves to another process as
    W3C does, with the host's baggage, and a key of another propagator."""

    def inject_context(self, span_id: str) -> dict[str, str]:
        return {
            "traceparent": f"00-{_KIT_TRACE}-{span_id}-01",
            "baggage": "tenant=t1",
            "b3": f"{_KIT_TRACE}-{span_id}-1",
        }


class _TracePeer(MCPPeer):
    """A server that sets aside the trace keys of each request after the
    negotiation, in :attr:`traces`, and reads the rest against its script."""

    def __init__(self, spec: dict[str, Any]) -> None:
        super().__init__(spec)
        self.traces: list[tuple[str, dict[str, Any]]] = []

    def _reply(self, message: dict[str, Any]) -> dict[str, Any] | None:
        if message["method"] not in ("initialize", "server/discover"):
            message = copy.deepcopy(message)
            params = message.get("params", {})
            meta = params.get("_meta", {})
            self.traces.append(
                (message["method"], {key: meta.pop(key) for key in _TRACE_KEYS if key in meta})
            )
            if "_meta" in params and not meta:
                del params["_meta"]
            if "params" in message and not params:
                del message["params"]
        return super()._reply(message)


@pytest.mark.parametrize("propagating", [False, True], ids=["local", "propagating"])
async def test_a_provider_that_propagates_the_trace_context_sends_the_host_s(
    propagating: bool,
) -> None:
    """A tool served to an agent sends its requests under the kit's current
    span. With ``propagate_trace_context=True`` each carries that span's
    trace context as the kit's telemetry names it to another process
    (``traceparent``, ``tracestate``, ``baggage`` only), unless its caller
    wrote one of its own, which goes whole: no host baggage beside the
    caller's traceparent. The application's OpenTelemetry span, current
    too, is never what goes out. roomkit-rs: ``roomkit-mcp``'s
    ``tests/mcp/trace.rs``."""
    replies = [
        {"method": "x/read", "result": {}},
        {"method": "x/read", "result": {}},
        {
            "method": "tools/call",
            "params": {"name": "lookup", "arguments": {}},
            "result": {"content": [{"type": "text", "text": "found"}]},
        },
    ]
    async with _TracePeer({"replies": replies}) as peer:
        server = MCPToolProvider(peer.url, discover=False, propagate_trace_context=propagating)
        current: list[str | None] = []

        async def relay(name: str, arguments: dict[str, Any]) -> str:
            current.append(get_current_span())
            with _application_span():
                await server.request("x/read")
                await server.request("x/read", {"_meta": {"traceparent": _CALLER_TRACE}})
                return await server.call_tool("lookup", {})

        kit = RoomKit(telemetry=_W3CTelemetry())
        kit.register_channel(SimpleChannel("ch1"))
        calling = AIResponse(
            content="", tool_calls=[AIToolCall(id="call_1", name="relay", arguments={})]
        )
        kit.register_channel(
            AIChannel(
                "assistant",
                provider=MockAIProvider(ai_responses=[calling, AIResponse(content="Done.")]),
                tool_handler=relay,
                tools=[AITool(name="relay", description="Relays to the server.")],
            )
        )
        async with server:
            room = await kit.create_room()
            await kit.attach_channel(room.id, "ch1")
            await kit.attach_channel(room.id, "assistant")
            message = InboundMessage(
                channel_id="ch1", sender_id="user1", content=TextContent(body="Look it up.")
            )
            await kit.process_inbound(message, room_id=room.id)
            await kit.close()
    assert peer.problems == []
    [span] = current
    assert span is not None
    host = {"traceparent": f"00-{_KIT_TRACE}-{span}-01", "baggage": "tenant=t1"}
    assert peer.traces == [
        ("x/read", host if propagating else {}),
        ("x/read", {"traceparent": _CALLER_TRACE}),
        ("tools/call", host if propagating else {}),
    ]


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
