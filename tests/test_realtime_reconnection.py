"""Concurrent microphone input and teardown during Gemini reconnection."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import Access, ChannelType
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.mock import MockRealtimeTransport
from tests.test_providers.test_gemini_realtime import _build_fake_genai


@pytest.fixture
def gemini():
    with patch.dict(sys.modules, _build_fake_genai()):
        from roomkit.providers.gemini.realtime import GeminiLiveProvider, _GeminiSessionState

        provider = GeminiLiveProvider(api_key="test")
        session = VoiceSession(id="s1", room_id="r", participant_id="p", channel_id="rt")
        state = _GeminiSessionState(
            session=session, live_config=SimpleNamespace(session_resumption=None)
        )
        provider._sessions[session.id] = state
        cm = AsyncMock()
        provider._client.aio.live.connect = MagicMock(return_value=cm)
        yield provider, session, state, cm


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize(
    ("access", "muted", "allowed"),
    [
        (Access.READ_WRITE, False, True),
        (Access.READ_ONLY, False, False),
        (Access.NONE, False, False),
        (Access.READ_WRITE, True, False),
    ],
)
async def test_channel_preserves_reconnect_audio_and_permissions(
    gemini, pipeline: bool, access: Access, muted: bool, allowed: bool
) -> None:
    provider, session, state, _ = gemini
    channel = RealtimeVoiceChannel("rt", provider=provider, transport=MockRealtimeTransport())
    channel._sessions[session.id] = session
    channel._session_bindings[session.id] = ChannelBinding(
        channel_id="rt", room_id="r", channel_type=ChannelType.VOICE, access=access, muted=muted
    )
    session.state = VoiceSessionState.CONNECTING
    try:
        if pipeline:
            channel._on_pipeline_processed_frame(session, AudioFrame(data=b"mic!"))
            await asyncio.sleep(0)
        else:
            await channel._forward_client_audio(session, b"mic!")
        assert state.pop_audio() == (b"mic!" if allowed else None)
        session.state = VoiceSessionState.ENDED
        await channel._forward_client_audio(session, b"late")
        await channel._forward_pipeline_frame(session, b"late", None, None, None)
        assert not state.audio_buffer
    finally:
        await channel.close()


async def test_replay_includes_audio_arriving_during_network_send(gemini) -> None:
    provider, session, state, cm = gemini
    entered, resume = asyncio.Event(), asyncio.Event()

    async def send(**kwargs):
        if not entered.is_set():
            entered.set()
            await resume.wait()

    live = cm.__aenter__.return_value
    live.send_realtime_input.side_effect = send
    state.buffer_audio(b"first")
    reconnect = asyncio.create_task(provider._reconnect(session))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await provider.send_audio(session, b"second")
        await provider.send_audio(session, b"third")
        resume.set()
        await asyncio.wait_for(reconnect, 1)
        assert [c.kwargs["audio"].data for c in live.send_realtime_input.await_args_list] == [
            b"first",
            b"second",
            b"third",
        ]
        assert not state.audio_buffer
        assert session.state == VoiceSessionState.ACTIVE
    finally:
        reconnect.cancel()
        await asyncio.gather(reconnect, return_exceptions=True)
        await provider.disconnect(session)


def test_buffer_bounds_memory_duration_and_age(gemini) -> None:
    _, _, state, _ = gemini
    state.input_sample_rate = 8000
    now = [10.0]
    with patch(
        "roomkit.providers.gemini.realtime.time", SimpleNamespace(monotonic=lambda: now[0])
    ):
        for _ in range(500):
            state.buffer_audio(b"x" * 16000)
        assert len(state.audio_buffer) == 2
        state.buffer_audio(b"y" * 100000)
        assert state.pop_audio() == b"y" * 32000
        assert state.pop_audio() is None
        state.buffer_audio(b"stale")
        now[0] += 3
        state.buffer_audio(b"recent")
        assert state.pop_audio() == b"recent"
        state.buffer_audio(b"expires while connecting")
        now[0] += 3
        assert state.pop_audio() is None


@pytest.mark.parametrize("during_replay", [False, True])
async def test_disconnect_during_reconnect_cannot_resurrect_session(gemini, during_replay) -> None:
    provider, session, state, cm = gemini
    entered, resume = asyncio.Event(), asyncio.Event()
    live = cm.__aenter__.return_value

    async def blocked(*args, **kwargs):
        entered.set()
        await resume.wait()
        return live

    if during_replay:
        state.buffer_audio(b"first")
        live.send_realtime_input.side_effect = blocked
    else:
        cm.__aenter__.side_effect = blocked
    reconnect = asyncio.create_task(provider._reconnect(session))
    await asyncio.wait_for(entered.wait(), 1)
    await provider.disconnect(session)
    resume.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(reconnect, 1)
    assert session.state == VoiceSessionState.ENDED
    assert not provider._sessions
    assert state.live_session is None
    assert not state.audio_buffer
    cm.__aexit__.assert_awaited_once()


async def test_failed_replay_closes_connection_and_allows_retry(gemini) -> None:
    provider, session, state, cm = gemini
    state.buffer_audio(b"first")
    state.buffer_audio(b"second")
    live = cm.__aenter__.return_value
    live.send_realtime_input.side_effect = ConnectionError("send failed")
    with pytest.raises(ConnectionError):
        await provider._reconnect(session)
    assert state.live_session is None
    assert session.state == VoiceSessionState.CONNECTING
    cm.__aexit__.assert_awaited_once()
    live.send_realtime_input.side_effect = None
    try:
        await provider._reconnect(session)
        assert live.send_realtime_input.await_args.kwargs["audio"].data == b"second"
        assert session.state == VoiceSessionState.ACTIVE
    finally:
        await provider.disconnect(session)
