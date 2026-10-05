"""Gemini Live frees the calls a reconnect orphans before the new socket's
handshake, whatever asked for the reconnect (RFC §12.4).

A handler that finishes during the handshake then finds its call abandoned,
its result dropped as stale, rather than a connection that is not there.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from roomkit.providers.gemini.realtime import GeminiLiveProvider, _GeminiSessionState
from roomkit.voice.base import VoiceSession, VoiceSessionState


class FakeLive:
    def __init__(self) -> None:
        self.sent: list[Any] = []

    async def send_tool_response(self, **kw: Any) -> None:
        self.sent.append(kw)

    async def send_realtime_input(self, **kw: Any) -> None:
        pass

    async def receive(self):  # noqa: ANN201
        await asyncio.Event().wait()
        yield None


class FakeCtx:
    async def __aexit__(self, *a: Any) -> None:
        pass


async def _run(path: str) -> list[str]:
    provider = GeminiLiveProvider(api_key="dummy")
    log: list[str] = []
    started, release = asyncio.Event(), asyncio.Event()

    async def slow_open(live_config: Any) -> tuple[Any, Any]:
        log.append("handshake starts")
        started.set()
        await release.wait()
        log.append("handshake done")
        return FakeCtx(), FakeLive()

    provider._open_live_session = slow_open  # type: ignore[method-assign]
    provider.on_tool_call_cancelled(lambda s, ids: log.append(f"abandoned {ids}"))
    session = VoiceSession(
        id="s1", room_id="r1", participant_id="u", channel_id="rt", state=VoiceSessionState.ACTIVE
    )
    live_config = provider._build_config(
        system_prompt="p",
        voice=None,
        tools=[],
        temperature=None,
        provider_config={},
        server_vad=True,
        warned=set(),
    )
    state = _GeminiSessionState(
        session=session,
        live_session=FakeLive(),
        ctxmgr=FakeCtx(),
        live_config=live_config,
        started_at=time.monotonic(),
        system_prompt="p",
        voice=None,
        tools=[],
        temperature=None,
    )
    provider._book_tool_call(session, "c1", "lookup")
    provider._sessions[session.id] = state

    if path == "reconfigure":

        async def noop(_: Any) -> None:
            return None

        provider._receive_loop = noop  # type: ignore[method-assign]
        task = asyncio.create_task(provider.reconfigure(session, system_prompt="new"))
    else:
        state.live_session = None  # the socket dropped
        task = asyncio.create_task(provider._receive_loop(session))
    await started.wait()
    try:
        await provider.submit_tool_result(session, "c1", "found")
        log.append("result for c1 during handshake: dropped as stale, no error")
    except Exception as exc:
        log.append(f"result for c1 during handshake: raised {exc!r}")
    release.set()
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    return log


@pytest.mark.parametrize("path", ["receive loop reconnect", "reconfigure"])
async def test_a_reconnect_frees_its_calls_before_the_handshake(path: str) -> None:
    log = await _run(path)

    assert log == [
        "abandoned ['c1']",
        "handshake starts",
        "result for c1 during handshake: dropped as stale, no error",
        "handshake done",
    ]
