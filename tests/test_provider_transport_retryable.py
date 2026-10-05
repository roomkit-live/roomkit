"""A transport failure is recognised once, through the error's cause chain,
whichever SDK surfaces it (RMK-509)."""

from __future__ import annotations

import httpx
import httpx2
import pytest

from roomkit.providers.ai.base import is_transport_failure

_REFUSED = "[Errno 111] Connection refused"


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError(_REFUSED),
        httpx.ReadTimeout("timed out"),
        httpx2.RemoteProtocolError("peer closed connection"),
        ConnectionResetError(104, "Connection reset by peer"),
        TimeoutError(),
    ],
    ids=["httpx-connect", "httpx-timeout", "httpx2-protocol", "socket-reset", "timeout"],
)
def test_a_transport_failure_is_recognised_through_its_cause(exc: Exception) -> None:
    wrapped = RuntimeError("Connection error.")
    wrapped.__cause__ = exc

    assert is_transport_failure(exc) is True
    assert is_transport_failure(wrapped) is True


def test_an_error_that_is_not_a_transport_failure_is_not_one() -> None:
    context_only = ValueError("bad arguments")
    context_only.__context__ = httpx.ConnectError(_REFUSED)

    assert is_transport_failure(ValueError("bad arguments")) is False
    # Only what an error was raised *from* is its cause: one merely raised
    # while handling a transport failure is not that failure.
    assert is_transport_failure(context_only) is False
