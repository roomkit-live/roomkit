"""What a failed generation reads as, on every text AI provider (RMK-524, RFC
§6.7).

The same failure reads the same way whatever form the server gave it and
whichever provider carries it: an overload is retried whether it came as an
error status or as an error written into a 200 stream; 408, 409, 429 and
every 5xx are retried, a client error is final; a 200 whose body is not the
provider's format (a gateway's HTML page) is a final :class:`ProviderError`,
never an empty success.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from roomkit.providers.ai.base import AIContext, AIMessage, ProviderError
from tests.text_conformance import openai_wire
from tests.text_conformance.http_wire import OVERLOAD, HttpWire, http_wires

WIRES = http_wires()
BY_WIRE = pytest.mark.parametrize("wire", WIRES, ids=[wire.label for wire in WIRES])
CONTEXT = AIContext(messages=[AIMessage(role="user", content="go")])
OPENAI_LABELS = {wire.label for wire in openai_wire.wires()}


async def _failure(provider: Any, mode: str) -> ProviderError:
    """The ProviderError one generation raises, streamed or not."""
    try:
        if mode == "generate":
            response = await provider.generate(CONTEXT)
            raise AssertionError(f"no failure: {response.content!r}")
        events = [event async for event in provider.generate_structured_stream(CONTEXT)]
        raise AssertionError(f"no failure: {events!r}")
    except ProviderError as error:
        return error
    finally:
        await provider.close()


@BY_WIRE
async def test_an_overload_reads_the_same_as_a_status_and_inside_a_200_stream(
    wire: HttpWire,
) -> None:
    if wire.stream_drops_status:
        pytest.skip(f"{wire.label}'s SDK raises an in-stream error without its status")
    as_status = await _failure(wire.status(wire.overload_status), "stream")
    in_stream = await _failure(wire.build(wire.stream_error), "stream")

    assert (as_status.retryable, in_stream.retryable) == (True, True)
    if wire.stream_names_status:
        assert in_stream.status_code == as_status.status_code


@BY_WIRE
@pytest.mark.parametrize("mode", ["generate", "stream"])
@pytest.mark.parametrize(
    ("status", "retryable"),
    [
        (408, True),
        (409, True),
        (429, True),
        (500, True),
        (502, True),
        (503, True),
        (504, True),
        (529, True),
        (400, False),
        (401, False),
        (404, False),
    ],
)
async def test_a_status_is_retried_when_transient(
    wire: HttpWire, status: int, retryable: bool, mode: str
) -> None:
    if status in wire.lost_statuses:
        pytest.skip(f"{wire.label}'s SDK raises a {status} without its status")
    error = await _failure(wire.status(status), mode)

    assert (error.retryable, error.status_code) == (retryable, status)


@BY_WIRE
@pytest.mark.parametrize("mode", ["generate", "stream"])
async def test_a_200_html_page_is_a_final_provider_error(wire: HttpWire, mode: str) -> None:
    error = await _failure(wire.build(wire.html), mode)

    assert error.retryable is False


@BY_WIRE
async def test_a_200_stream_with_no_event_is_a_final_provider_error(wire: HttpWire) -> None:
    error = await _failure(wire.empty_stream(), "stream")

    assert error.retryable is False


@BY_WIRE
async def test_an_error_in_a_stream_that_names_no_status_is_final(wire: HttpWire) -> None:
    """Without a status, only a lost connection is retried: a message that
    happens to contain "rate" (in "generate") names no rate limit."""
    if wire.label == "ollama":
        pytest.skip("ollama's -1, a stream the server aborted, reads as a lost connection")
    error = await _failure(wire.unnamed_stream_error(), "stream")

    assert error.retryable is False


@pytest.mark.parametrize(
    "wire",
    [wire for wire in WIRES if wire.label in OPENAI_LABELS],
    ids=sorted(OPENAI_LABELS),
)
async def test_a_200_json_error_object_reads_as_its_status(wire: HttpWire) -> None:
    """A gateway answers a completion with 200 and an error object."""
    body = {"error": {"message": OVERLOAD, "type": "server_error", "code": 503}}
    provider = wire.build(lambda request: httpx.Response(200, json=body))

    error = await _failure(provider, "generate")

    assert (error.retryable, error.status_code) == (True, 503)
