"""What an MCP provider sends and reads on the wire under the SDK 2 (RFC §21.2, RMK-663).

The SDK writes the host's trace context into every request it sends and checks
a method it knows against its schema; the provider reads the server's words
off the wire instead, and sends only the trace keys its caller wrote.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from opentelemetry import baggage, context, trace
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

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
