"""Gemini Live: fail-fast + exact close-code surfacing on permanent disconnects.

A close code like 1007 (invalid argument — e.g. a tool schema Gemini Live
rejects) or 1011 (quota) won't recover by reconnecting. The receive loop must
end the session immediately and fire the error callback with the exact code,
rather than burning 5 back-off retries on a doomed connection.
"""

from __future__ import annotations

import pytest

pytest.importorskip("google.genai", reason="google-genai not installed")

from roomkit.providers.gemini.realtime import _GeminiSessionState
from roomkit.voice.base import VoiceSession, VoiceSessionState


class _FakeAPIError(Exception):
    """Mimics google.genai APIError, which carries a numeric ``.code``."""

    def __init__(self, code: int) -> None:
        super().__init__(f"{code} None. Request contains an invalid argument.")
        self.code = code


class _FakeLiveSession:
    """A Gemini live session whose ``receive()`` immediately raises."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def receive(self):  # noqa: ANN202 - async generator
        raise self._exc
        yield  # pragma: no cover - makes this a generator


@pytest.fixture
def provider():
    from roomkit.providers.gemini.realtime import GeminiLiveProvider

    return GeminiLiveProvider(api_key="test-key")


def _seed_session(provider, *, code: int) -> VoiceSession:
    session = VoiceSession(
        id="sess-1", room_id="room-1", participant_id="user-1", channel_id="voice"
    )
    session.state = VoiceSessionState.ACTIVE
    state = _GeminiSessionState(
        session=session,
        live_session=_FakeLiveSession(_FakeAPIError(code)),
        started_at=0.0,
    )
    provider._sessions[session.id] = state
    return session


class TestNonRetryableFastFail:
    async def test_1007_fires_exact_code_and_ends(self, provider) -> None:
        session = _seed_session(provider, code=1007)
        errors: list[tuple[str, str]] = []
        provider.on_error(lambda sess, code, msg: errors.append((code, msg)))

        # Must return (not loop forever) without reconnecting.
        reconnects = 0

        async def _no_reconnect(_sess):
            nonlocal reconnects
            reconnects += 1

        provider._reconnect = _no_reconnect  # type: ignore[method-assign]

        await provider._receive_loop(session)

        assert reconnects == 0, "permanent failure must not reconnect"
        assert session.state == VoiceSessionState.ENDED
        assert len(errors) == 1
        code, msg = errors[0]
        assert code == "ws_1007"
        assert "invalid argument" in msg

    async def test_1011_quota_fires_exact_code(self, provider) -> None:
        session = _seed_session(provider, code=1011)
        errors: list[str] = []
        provider.on_error(lambda sess, code, msg: errors.append(code))
        provider._reconnect = lambda _sess: None  # type: ignore[assignment]

        await provider._receive_loop(session)

        assert errors == ["ws_1011"]
        assert session.state == VoiceSessionState.ENDED

    async def test_transient_close_does_not_fastfail(self, provider) -> None:
        """A transient code (1006) goes through the reconnect path, not fast-fail."""
        session = _seed_session(provider, code=1006)
        errors: list[str] = []
        provider.on_error(lambda sess, code, msg: errors.append(code))

        # Stop the loop after one reconnect attempt by ending the session.
        attempts = 0

        async def _reconnect(_sess):
            nonlocal attempts
            attempts += 1
            session.state = VoiceSessionState.ENDED

        provider._reconnect = _reconnect  # type: ignore[method-assign]

        await provider._receive_loop(session)

        assert attempts == 1, "transient close must attempt reconnect"
        assert errors == [], "transient close must not fire ws_<code> fast-fail"
