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

import pytest

from roomkit.providers.ai.base import AIContext, AIMessage, ProviderError
from tests.text_conformance.http_wire import HttpWire, http_wires

WIRES = http_wires()
BY_WIRE = pytest.mark.parametrize("wire", WIRES, ids=[wire.label for wire in WIRES])
CONTEXT = AIContext(messages=[AIMessage(role="user", content="go")])


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
    as_status = await _failure(wire.status(wire.overload_status), "stream")
    in_stream = await _failure(wire.build(wire.stream_error), "stream")

    assert (as_status.retryable, in_stream.retryable) == (True, True)
    if wire.stream_names_status:
        assert in_stream.status_code == as_status.status_code


@BY_WIRE
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
    wire: HttpWire, status: int, retryable: bool
) -> None:
    if status in wire.lost_statuses:
        pytest.skip(f"{wire.label}'s SDK raises a {status} without its status")
    error = await _failure(wire.status(status), "stream")

    assert (error.retryable, error.status_code) == (retryable, status)


@BY_WIRE
@pytest.mark.parametrize("mode", ["generate", "stream"])
async def test_a_200_html_page_is_a_final_provider_error(wire: HttpWire, mode: str) -> None:
    error = await _failure(wire.build(wire.html), mode)

    assert error.retryable is False
