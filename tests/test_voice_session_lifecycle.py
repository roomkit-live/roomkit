"""Cancellation races across SIP call setup and realtime handshakes."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.voice.backends.sip import SIPVoiceBackend
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.voice.backends.test_sip import (
    _make_mock_aiosipua_module,
    _make_mock_call_session,
    _make_mock_incoming_call,
    _make_mock_rtp_bridge_module,
)


@pytest.fixture
def sip() -> tuple[SIPVoiceBackend, MagicMock]:
    media = _make_mock_call_session()
    with (
        patch(
            "roomkit.voice.backends.sip.import_aiosipua", return_value=_make_mock_aiosipua_module()
        ),
        patch(
            "roomkit.voice.backends.sip.import_rtp_bridge",
            return_value=_make_mock_rtp_bridge_module(media),
        ),
    ):
        backend = SIPVoiceBackend(
            local_rtp_ip="10.0.0.5", rtp_port_start=10000, rtp_port_end=10002
        )
    return backend, media


@pytest.mark.parametrize("phase", ["accept", "provider"])
@pytest.mark.parametrize("stop", ["disconnect", "close"])
@pytest.mark.parametrize("swallow_cancel", [False, True])
async def test_realtime_handshake_cannot_outlive_transport(
    phase: str,
    stop: str,
    swallow_cancel: bool,
) -> None:
    entered = asyncio.Event()
    captured: list[VoiceSession] = []
    provider, transport = MockRealtimeProvider(), MockRealtimeTransport()
    owner = transport if phase == "accept" else provider
    method = "accept" if phase == "accept" else "connect"
    original = getattr(owner, method)

    async def blocked(session: VoiceSession, *args: Any, **kwargs: Any) -> None:
        captured.append(session)
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if not swallow_cancel:
                raise
        await original(session, *args, **kwargs)

    setattr(owner, method, blocked)
    channel = RealtimeVoiceChannel("rt", provider=provider, transport=transport)
    start = asyncio.create_task(channel.start_session("r", "p", object()))
    await entered.wait()
    if stop == "close":
        await channel.close()
    else:
        await transport.simulate_client_disconnect(captured[0])
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(start, 1)
    assert not channel.get_room_sessions("r")
    assert not channel._connecting_sessions
    assert not channel._preconnect_audio
    assert not channel._session_spans
    assert not channel._session_tools
    assert any(c.method == "disconnect" for c in transport.calls)
    await channel.close()


@pytest.mark.parametrize("stop", ["cancel", "close"])
async def test_inbound_setup_rollback(sip: tuple[SIPVoiceBackend, MagicMock], stop: str) -> None:
    backend, media = sip
    entered = asyncio.Event()

    async def start_media() -> None:
        entered.set()
        await asyncio.Event().wait()

    media.start.side_effect = start_media
    await backend.start()
    call = _make_mock_incoming_call()
    backend._spawn_invite_handler(call)
    task = next(iter(backend._invite_tasks))
    await entered.wait()
    if stop == "close":
        await backend.close()
    else:
        task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not backend._allocated_ports
    assert not backend._session_states
    assert not backend._reserved_session_ids
    media.close.assert_awaited_once()
    backend._uac.send_bye.assert_called_once()
    await backend.close()


@pytest.mark.parametrize("answered", [False, True])
async def test_outbound_cancel_releases_resources(
    sip: tuple[SIPVoiceBackend, MagicMock],
    answered: bool,
) -> None:
    backend, media = sip
    await backend.start()
    entered = asyncio.Event()
    out_call = MagicMock(call_id="out1")
    out_call.wait_answered = AsyncMock()

    async def block(**kwargs: Any) -> None:
        entered.set()
        await asyncio.Event().wait()

    if answered:
        media.start.side_effect = block
    else:
        out_call.wait_answered.side_effect = block
    backend._uac.send_invite.return_value = out_call
    with patch.dict(sys.modules, {"aiosipua": SimpleNamespace(build_sdp=lambda **kw: "sdp")}):
        task = asyncio.create_task(backend.dial("sip:a@test", "sip:b@test", ("10.0.0.1", 5060)))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not backend._allocated_ports
    assert not backend._call_to_session
    assert not backend._setup_tasks
    backend._uac.remove_call.assert_called_once_with("out1")
    if answered:
        media.close.assert_awaited_once()
        out_call.hangup.assert_called_once_with(backend._uac)
        out_call.cancel.assert_not_called()
    else:
        out_call.cancel.assert_called_once_with(backend._uac)
        out_call.hangup.assert_not_called()
    await backend.close()


async def test_repeated_cancel_waits_for_media_cleanup(
    sip: tuple[SIPVoiceBackend, MagicMock],
) -> None:
    backend, media = sip
    started, closing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def start_media() -> None:
        started.set()
        await asyncio.Event().wait()

    async def close_media() -> None:
        closing.set()
        await release.wait()

    media.start.side_effect, media.close.side_effect = start_media, close_media
    task = asyncio.create_task(backend._handle_invite(_make_mock_incoming_call()))
    await started.wait()
    task.cancel()
    await closing.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert backend._allocated_ports == {10000}
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not backend._allocated_ports
    media.close.assert_awaited_once()


async def test_close_rejects_new_invites_and_is_shared(
    sip: tuple[SIPVoiceBackend, MagicMock],
) -> None:
    backend, _ = sip
    await backend.start()
    await asyncio.gather(backend.close(), backend.close())
    call = _make_mock_incoming_call()
    backend._spawn_invite_handler(call)
    call.reject.assert_called_once_with(503, "Service Unavailable")
    backend._uas.stop.assert_awaited_once()
    with pytest.raises(RuntimeError, match="closing"):
        await backend.dial("sip:a@test", "sip:b@test", ("10.0.0.1", 5060))
