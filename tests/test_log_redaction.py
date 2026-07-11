"""Tests for the central content-redaction policy."""

from __future__ import annotations

import logging

import pytest

from roomkit.telemetry.redaction import (
    content_logging_enabled,
    redact,
    set_content_logging,
)


@pytest.fixture(autouse=True)
def _reset_content_logging():
    """Content logging is a process-wide global — restore it after each test."""
    original = content_logging_enabled()
    yield
    set_content_logging(original)


def test_redacted_by_default() -> None:
    set_content_logging(False)
    assert content_logging_enabled() is False
    out = redact("secret transcript")
    assert "secret" not in out
    assert "17 chars" in out  # len("secret transcript") == 17


def test_full_content_when_enabled() -> None:
    set_content_logging(True)
    assert content_logging_enabled() is True
    assert redact("secret transcript") == "secret transcript"


def test_none_and_non_string() -> None:
    set_content_logging(False)
    assert redact(None) == "<none>"
    assert "5 chars" in redact(12345)  # str(12345) has length 5


def test_no_content_reaches_logs_by_default(caplog: pytest.LogCaptureFixture) -> None:
    set_content_logging(False)
    log = logging.getLogger("roomkit.voice")
    with caplog.at_level(logging.DEBUG, logger="roomkit.voice"):
        log.debug("Transcription: %s", redact("my SIN is 123-456-789"))
    assert "123-456-789" not in caplog.text
    assert "redacted" in caplog.text
