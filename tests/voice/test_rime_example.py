"""Run the Prism microphone example through its real channel and provider."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection

from roomkit import RoomKit
from roomkit.voice import VoiceSession, VoiceSessionState
from roomkit.voice.backends import LocalAudioBackend
from roomkit.voice.realtime import MockRealtimeTransport, RimeRealtimeConfig
from tests.voice.rime_wire import prism_handshake, receive, server


async def test_microphone_example_starts_and_closes_a_prism_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "examples"))
    from examples import realtime_voice_local_rime as example

    monkeypatch.setenv("RIME_API_KEY", "test-key")
    monkeypatch.setenv("RIME_VOICE", "astra")
    captured = asyncio.Event()
    sessions: list[VoiceSession] = []

    class Transport(MockRealtimeTransport):
        async def accept(self, session: VoiceSession, connection: Any) -> None:
            sessions.append(session)
            await super().accept(session, connection)

    transport = Transport()

    def audio_backend(**kwargs: Any) -> Transport:
        # Substitute the device, retaining the real constructor's contract.
        inspect.signature(LocalAudioBackend).bind(**kwargs)
        return transport

    async def stop(kit: RoomKit) -> None:
        await asyncio.wait_for(captured.wait(), 1)
        await kit.close()

    async def handler(ws: ServerConnection) -> None:
        await prism_handshake(ws)
        event = await receive(ws)
        assert event["type"] == "input_audio_buffer.append"
        captured.set()
        await ws.wait_closed()

    monkeypatch.setattr(example, "LocalAudioBackend", audio_backend)
    monkeypatch.setattr(example, "run_until_stopped", stop)
    async with server(handler) as url:
        monkeypatch.setattr(
            example,
            "RimeRealtimeConfig",
            lambda **kwargs: RimeRealtimeConfig(websocket_url=url, **kwargs),
        )
        await example.main()

    assert len(sessions) == 1 and sessions[0].state == VoiceSessionState.ENDED
    assert any(call.method == "close" for call in transport.calls)
