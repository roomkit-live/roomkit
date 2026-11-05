"""Run the Prism microphone example through its real channel and provider."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection

from roomkit import RoomKit
from roomkit.voice import VoiceSession, VoiceSessionState
from roomkit.voice.backends import LocalAudioBackend
from roomkit.voice.pipeline import MockAECProvider
from roomkit.voice.realtime import MockRealtimeTransport, RimeRealtimeConfig
from tests.voice.rime_wire import prism_handshake, receive, server


@pytest.mark.parametrize(
    ("aec_mode", "mute_mic", "expected_muted"),
    [(None, None, False), ("0", None, True), ("webrtc", "1", True), ("0", "0", False)],
)
@pytest.mark.parametrize("console_mode", ["0", "1"])
async def test_microphone_example_starts_and_closes_a_prism_session(
    monkeypatch: pytest.MonkeyPatch,
    aec_mode: str | None,
    mute_mic: str | None,
    expected_muted: bool,
    console_mode: str,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "examples"))
    from examples import realtime_voice_local_rime as example

    monkeypatch.setenv("RIME_API_KEY", "test-key")
    monkeypatch.setenv("RIME_VOICE", "astra")
    monkeypatch.setenv("CONSOLE", console_mode)
    dashboards = []

    class Dashboard:
        def __init__(self, kit: RoomKit) -> None:
            self.kit, self.closed = kit, False
            dashboards.append(self)

        async def stop(self) -> None:
            assert not self.closed
            self.closed = True

    monkeypatch.setattr("roomkit.console.RoomKitConsole", Dashboard)
    if aec_mode is None:
        monkeypatch.delenv("AEC", raising=False)
    else:
        monkeypatch.setenv("AEC", aec_mode)
    monkeypatch.delenv("AEC_DELAY_MS", raising=False)
    if mute_mic is None:
        monkeypatch.delenv("MUTE_MIC", raising=False)
    else:
        monkeypatch.setenv("MUTE_MIC", mute_mic)
    canceller = MockAECProvider()

    def webrtc_aec(*, sample_rate: int, **kwargs: Any) -> MockAECProvider:
        assert sample_rate == 16000
        return canceller

    monkeypatch.setattr("roomkit.voice.pipeline.aec.WebRTCAECProvider", webrtc_aec)
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
        assert kwargs["aec"] is (None if aec_mode == "0" else canceller)
        assert kwargs["mute_mic_during_playback"] is expected_muted
        return transport

    async def stop(
        kit: RoomKit,
        *,
        cleanup: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        await asyncio.wait_for(captured.wait(), 1)
        assert len(dashboards) == int(console_mode)
        assert all(dashboard.kit is kit and not dashboard.closed for dashboard in dashboards)
        if cleanup:
            await cleanup()
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
    assert all(dashboard.closed for dashboard in dashboards)
