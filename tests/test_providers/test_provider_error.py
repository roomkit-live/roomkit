"""The one reading of a failed generation every AI provider shares (RFC §6.7).

A provider hands its SDK's failure to ``provider_error`` (or a body it could
not take as an answer to ``error_body_failure``), and the ``ProviderError`` it
gets back carries the status the server described and whether it is retried:
the same for every provider, whatever form the server gave the failure. The
failures here are plain exceptions shaped as an SDK's are; no vendor's SDK is
involved.
"""

from __future__ import annotations

from typing import Any

import pytest

from roomkit.providers.ai.base import error_body_failure, provider_error


class _SDKError(Exception):
    """An SDK's failure: a message, and the status or body it may carry."""

    def __init__(self, message: str = "failed", *, status_code: Any = None, body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


@pytest.mark.parametrize(
    ("status", "retryable"),
    [
        (408, True),
        (409, True),
        (429, True),
        (500, True),
        (503, True),
        (529, True),
        (400, False),
        (401, False),
        (404, False),
        (422, False),
    ],
)
def test_a_status_decides_whether_a_failure_is_retried(status: int, retryable: bool) -> None:
    error = provider_error(_SDKError(status_code=status), provider="any")

    assert (error.status_code, error.retryable) == (status, retryable)


@pytest.mark.parametrize(
    ("body", "status", "retryable"),
    [
        ({"error": {"type": "rate_limit_error", "message": "slow down"}}, 429, True),
        ({"error": {"code": 400, "message": "bad request"}}, 400, False),
        ('{"error": {"status": "UNAVAILABLE"}}', 503, True),
    ],
)
def test_a_status_written_in_the_error_s_body_is_its_status(
    body: Any, status: int, retryable: bool
) -> None:
    """A stream that had already answered 200 carries its failure in an error
    event's body: the status read there decides as a status line would."""
    error = provider_error(_SDKError(body=body), provider="any")

    assert (error.status_code, error.retryable) == (status, retryable)


@pytest.mark.parametrize(
    ("failure", "retryable"),
    [
        (ConnectionError("connection reset by peer"), True),
        (TimeoutError("read timed out"), True),
        (ValueError("Expecting value: line 1 column 1"), False),
        (_SDKError("could not parse", status_code=200), False),
        (_SDKError("Rate limit reached, try again"), True),
        (_SDKError("upstream answered 503"), True),
        (_SDKError("the token limit of this model was reached"), False),
        (_SDKError("failed to generate"), False),
    ],
)
def test_without_a_status_the_failure_s_form_decides(failure: Exception, retryable: bool) -> None:
    """A lost connection is retried; an answer that could not be read is
    final; anything else is retried only when its message names a retried
    status, a rate limit or an overload, as whole words."""
    error = provider_error(failure, provider="any")

    assert (error.status_code, error.retryable) == (None, retryable)


@pytest.mark.parametrize(
    ("body", "status", "retryable"),
    [
        ({"error": {"code": 503, "message": "busy"}}, 503, True),
        ({"error": {"code": 400, "message": "no such model"}}, 400, False),
        ({"error": {"message": "something went wrong"}}, None, False),
    ],
)
def test_a_200_whose_body_is_an_error_object_reads_as_the_status_it_names(
    body: dict[str, Any], status: int | None, retryable: bool
) -> None:
    error = error_body_failure(body, provider="any")

    assert (error.status_code, error.retryable) == (status, retryable)
